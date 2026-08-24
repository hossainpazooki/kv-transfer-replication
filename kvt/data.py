import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kvt.cache import get_layer_kv
from kvt.pairs import kv_shape
from kvt.rope import strip_rope_tokens_first


def iter_fineweb_sequences(tokenizer, n_seqs: int, seq_len: int, seed: int = 0) -> np.ndarray:
    """First n_seqs FineWeb-Edu docs (shuffled with `seed`) that tokenize to >= seq_len tokens, truncated."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    out = []
    for doc in ds:
        ids = tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= seq_len:
            out.append(ids[:seq_len])
            if len(out) == n_seqs:
                break
    return np.asarray(out, dtype=np.int64)


@torch.no_grad()
def dump_kv(model, seqs: np.ndarray, stride: int, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = kv_shape(model.config)
    n_seqs, seq_len = seqs.shape
    keep = np.arange(0, seq_len, stride)
    Ks = [[] for _ in range(shape.n_layers)]
    Vs = [[] for _ in range(shape.n_layers)]
    dev = next(model.parameters()).device
    for i in tqdm(range(n_seqs), desc="dump", ascii=True):
        ids = torch.tensor(seqs[i : i + 1], device=dev)
        out = model(input_ids=ids, use_cache=True)
        for l in range(shape.n_layers):
            k, v = get_layer_kv(out.past_key_values, l)                       # [1, n_kv, T, d_h]
            Ks[l].append(k[0, :, keep].transpose(0, 1).to(torch.float16).cpu().numpy())
            Vs[l].append(v[0, :, keep].transpose(0, 1).to(torch.float16).cpu().numpy())
    for l in range(shape.n_layers):
        np.savez(out_dir / f"layer{l:02d}.npz", K=np.concatenate(Ks[l]), V=np.concatenate(Vs[l]))
    np.savez(out_dir / "meta.npz",
             positions=np.tile(keep, n_seqs).astype(np.int64),
             seq_idx=np.repeat(np.arange(n_seqs), len(keep)).astype(np.int64))
    (out_dir / "meta.json").write_text(json.dumps({
        "model": getattr(model.config, "_name_or_path", "unknown"),
        "n_layers": shape.n_layers, "n_kv": shape.n_kv, "d_h": shape.d_h,
        "rope_theta": shape.rope_theta, "stride": stride, "seq_len": int(seq_len), "n_seqs": int(n_seqs),
    }, indent=2))


class KVDump:
    KINDS = ("K_rope", "K_stripped", "V")

    def __init__(self, root: Path, meta: dict, positions: np.ndarray, seq_idx: np.ndarray):
        self.root = root
        self.n_layers, self.n_kv, self.d_h = meta["n_layers"], meta["n_kv"], meta["d_h"]
        self.rope_theta, self.n_seqs, self.stride = meta["rope_theta"], meta["n_seqs"], meta["stride"]
        self.positions = torch.from_numpy(positions)
        self.seq_idx = seq_idx
        self._cache: dict[tuple[str, int], torch.Tensor] = {}

    @classmethod
    def load(cls, root) -> "KVDump":
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        m = np.load(root / "meta.npz")
        return cls(root, meta, m["positions"], m["seq_idx"])

    def get(self, kind: str, layer: int) -> torch.Tensor:
        """Lazily load and cache a key-value tensor.

        Returns a shared cached tensor: callers MUST treat it as read-only and must never
        mutate it in place. If a caller modifies the returned tensor, every subsequent call
        to get() for that (kind, layer) will return silently corrupted data, propagating
        errors through the entire downstream analysis.

        Cloning on every call is not an option: tensors are ~210 MB each at the real run size,
        and the ridge probe calls get() 784 times in its inner loop. Cloning would copy roughly
        164 GB per run. The caching by reference is deliberate and load-bearing for performance.

        The contract is: read the returned tensor, extract what you need, and do not modify.
        """
        assert kind in self.KINDS, kind
        key = (kind, layer)
        if key not in self._cache:
            with np.load(self.root / f"layer{layer:02d}.npz") as z:
                if kind == "V":
                    t = torch.from_numpy(z["V"].astype(np.float32))
                else:
                    t = torch.from_numpy(z["K"].astype(np.float32))
                    if kind == "K_stripped":
                        t = strip_rope_tokens_first(t, self.positions, self.rope_theta)
            self._cache[key] = t
        return self._cache[key]

    def split(self, holdout_frac: float):
        n_ho = math.ceil(holdout_frac * self.n_seqs)
        heldout = self.seq_idx >= (self.n_seqs - n_ho)
        return ~heldout, heldout
