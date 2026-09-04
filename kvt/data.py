import json
import math
from collections import OrderedDict
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
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)   # K/V unaffected; drops the [T, vocab] f32 logits (19.5 GB at 32k)
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
        self._cache: "OrderedDict[tuple[str, int], torch.Tensor]" = OrderedDict()
        self._cache_limit: int | None = None

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

        Cloning on every call is not an option: at the 50-sequence run size a tensor is 52.4 MB
        (12,800 stride-4 rows x 8 heads x 128 dims, float32) and the ridge probe calls get() 784
        times in its inner loop, so cloning would copy about 41 GB per run. At 420 sequences a
        tensor is 440 MB. The caching by reference is deliberate and load-bearing for performance.

        Corrected 2026-08-24: this docstring previously read "~210 MB each ... roughly 164 GB
        per run". That was the UN-STRIDED size -- it omitted the stride-4 subsampling, and was
        4x too high. Both figures above are measured from data/kv/qwen3-0.6b-to-1.7b/source.
        The decision is unchanged (41 GB of copying is still prohibitive); only its evidence
        was wrong. See docs/ledger.md, "What fired / what is blocked".

        The contract is: read the returned tensor, extract what you need, and do not modify.

        The cache is LRU-ordered and bounded by set_cache_limit(); eviction never changes a
        returned value, since a reloaded tensor is bit-identical.
        """
        assert kind in self.KINDS, kind
        key = (kind, layer)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with np.load(self.root / f"layer{layer:02d}.npz") as z:
            if kind == "V":
                t = torch.from_numpy(z["V"].astype(np.float32))
            else:
                t = torch.from_numpy(z["K"].astype(np.float32))
                if kind == "K_stripped":
                    t = strip_rope_tokens_first(t, self.positions, self.rope_theta)
        self._cache[key] = t
        self._evict()
        return t

    def split(self, holdout_frac: float):
        n_ho = math.ceil(holdout_frac * self.n_seqs)
        heldout = self.seq_idx >= (self.n_seqs - n_ho)
        return ~heldout, heldout

    def seq_range_mask(self, lo: int, hi: int) -> np.ndarray:
        """Boolean row mask over token rows whose sequence index is in [lo, hi).

        Unlike split(), the range is absolute, so a caller can grow a training prefix
        (0, n) while holding a FIXED held-out range (h0, h1) -- which is what makes the
        WP2 learning curve a curve in n rather than a curve in n and the held-out set at once.
        """
        if not (0 <= lo <= hi <= self.n_seqs):
            raise ValueError(
                f"invalid sequence range [{lo}, {hi}) for a dump with {self.n_seqs} sequences")
        return (self.seq_idx >= lo) & (self.seq_idx < hi)

    def clear_cache(self) -> None:
        """Drop every lazily-loaded tensor. At 420 sequences one cached layer is ~440 MB in
        float32; a k=8 fit holds 8 of them plus a 3.4 GB design matrix, so callers that walk
        many layers must evict between layers or exhaust RAM."""
        self._cache.clear()

    def set_cache_limit(self, n: int | None) -> None:
        """Bound the lazy cache to `n` tensors (None = unlimited, the default).

        Needed because fit_mapper/mapper_r2 walk every target layer and the cache would
        otherwise grow to 2*L_s + 2*L_t tensors with no eviction: 5.9 GB at 50 sequences
        (which is why the committed runs worked) but 49 GB at 420, on a 32 GB box.
        Eviction is invisible in the numbers -- an evicted tensor is reloaded from disk
        bit-identically -- so this trades a little I/O for a bounded footprint.
        """
        if n is not None and n < 1:
            raise ValueError(f"cache limit must be >= 1 or None, got {n}")
        self._cache_limit = n
        self._evict()

    def _evict(self) -> None:
        if self._cache_limit is None:
            return
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)      # drop least-recently-used
