import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from kvt.checkpoint import MANIFEST_FILE, CheckpointMismatch, record_checkpoint
from kvt.data import dump_kv
from kvt.models import DTYPES, load_model, pretrained_args
from kvt.pairs import PAIRS, ModelRef, check_matched_kv
from transformers import AutoConfig


PROBE_TOKENS = 64


@torch.no_grad()
def probe_forward(model, seq: np.ndarray, load_dtype: str) -> dict:
    """One short forward so a new box (Apple silicon, a smaller card) reports its device, dtype,
    attention kernel and peak memory before the hours-long dump commits to them."""
    dev = next(model.parameters()).device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    ids = torch.tensor(np.asarray(seq)[None, :PROBE_TOKENS], device=dev)
    model(input_ids=ids, use_cache=True, logits_to_keep=1)
    if dev.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(dev))
    elif dev.type == "mps":
        peak = int(torch.mps.driver_allocated_memory())
    else:
        peak = -1
    return {"device": dev.type, "dtype": load_dtype,
            "attn": model.config._attn_implementation, "peak_bytes": peak}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--which", required=True, choices=["source", "target"])
    ap.add_argument("--tokens", default=None, help="defaults to data/tokens/<pair>_n50_len1024_seed0.npy")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output dir; defaults to data/kv/<pair>/<which>. Pass an explicit path "
                         "when dumping a DIFFERENT sequence count: the default path holds the "
                         "50-sequence dumps that back Runs 1/2/4, and overwriting them would "
                         "silently change what every re-verify line in docs/ledger.md recomputes "
                         "(KVDump.split(0.2) holds out the last 20%%, which moves with n_seqs).")
    ap.add_argument("--rope-scaling", default=None,
                    help="JSON in the HF form, e.g. '{\"rope_type\": \"yarn\", \"factor\": 2.5, "
                         "\"original_max_position_embeddings\": 32768}' (linear-ceiling E9-long). The model is "
                         "loaded with this scaling, the dump's meta.json records the RoPE spec the model applied "
                         "(frequencies + attention factor), and KVDump strips with it. Omit for the native RoPE.")
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES),
                    help="dtype the model is loaded and run in; K/V are saved float16 either way.")
    ap.add_argument("--probe", action="store_true",
                    help="before the dump, run one forward of the first sequence (64 tokens) and print "
                         "device/dtype/attention kernel/peak memory; also recorded under 'probe' in meta.json.")
    ap.add_argument("--revision", default=None,
                    help="HF revision (branch, tag or commit) for the --which model. The commit it resolves to "
                         "is hashed, loaded, and recorded in meta.json. Exclusive with --local-path.")
    ap.add_argument("--local-path", default=None,
                    help="load the --which model from this local checkpoint directory (e.g. a training "
                         "checkpoint) instead of the Hub; the pair's id is kept as the recorded model_id. "
                         "Exclusive with --revision.")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    rope_scaling = None
    if a.rope_scaling:
        rope_scaling = json.loads(a.rope_scaling)
    pair = PAIRS[a.pair]
    refs = {w: pair.model_ref(w) for w in ("source", "target")}
    if a.revision is not None or a.local_path is not None:
        base = refs[a.which]
        if base.revision is not None or base.local_path is not None:
            raise SystemExit(f"pair {a.pair} already pins its {a.which}; refusing to override it from the CLI")
        try:
            refs[a.which] = ModelRef(base.model_id, revision=a.revision,
                                     local_path=Path(a.local_path) if a.local_path is not None else None)
        except (ValueError, TypeError) as e:
            raise SystemExit(str(e))
    # Hash the checkpoint BEFORE loading it; the model is then loaded from exactly what was hashed
    # (an HF ref is pinned to the resolved commit) and re-verified after the dump.
    record = record_checkpoint(refs[a.which])
    refs[a.which] = record.load_ref()
    check_matched_kv(*(AutoConfig.from_pretrained(p, **kw) for p, kw in
                       (pretrained_args(refs["source"]), pretrained_args(refs["target"]))))
    tokens = Path(a.tokens or f"data/tokens/{a.pair}_n50_len1024_seed0.npy")
    seqs = np.load(tokens)
    out_dir = Path(a.out) if a.out else Path("data/kv") / a.pair / a.which
    if out_dir.exists() and (out_dir / "meta.json").exists():
        prev = json.loads((out_dir / "meta.json").read_text())
        if int(prev.get("n_seqs", -1)) != int(seqs.shape[0]):
            raise SystemExit(
                f"refusing to overwrite {out_dir}: it holds a {prev.get('n_seqs')}-sequence dump "
                f"and this run would write {seqs.shape[0]}. A different n_seqs changes what "
                f"KVDump.split() holds out, so every number recomputed from this directory would "
                f"change silently. Pass --out with a different path.")
        prev_digest = (prev.get("checkpoint") or {}).get("manifest_digest")
        if prev_digest is not None and prev_digest != record.manifest["digest"]:
            raise SystemExit(
                f"refusing to overwrite {out_dir}: it was dumped from checkpoint manifest {prev_digest}, "
                f"this run's checkpoint is {record.manifest['digest']}. Pass --out with a different path.")
    model = load_model(refs[a.which], rope_scaling=rope_scaling, dtype=a.dtype)
    probe = None
    if a.probe:
        probe = probe_forward(model, seqs[0], a.dtype)
        print(f"probe device={probe['device']} dtype={probe['dtype']} attn={probe['attn']} "
              f"peak_bytes={probe['peak_bytes']}")
    t0 = time.time()
    dump_kv(model, seqs, a.stride, out_dir, load_dtype=a.dtype, checkpoint=record)
    try:
        record.verify()
    except CheckpointMismatch as e:
        (out_dir / "meta.json").rename(out_dir / "meta.json.INVALID")   # KVDump.load now refuses this dir
        raise SystemExit(f"checkpoint changed during the dump; {out_dir} invalidated: {e}")
    if probe is not None:   # added after the fact so meta.json's default key set is unchanged without --probe
        meta_path = out_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["probe"] = probe
        meta_path.write_text(json.dumps(meta, indent=2))
    rope = getattr(model.config, "rope_parameters", {})
    print(f"wrote {out_dir} n_seqs={seqs.shape[0]} stride={a.stride} rope={rope.get('rope_type', 'default')} "
          f"in {time.time() - t0:.0f}s; checkpoint {record.manifest['digest'][:12]} "
          f"({len(record.manifest['files'])} files, {MANIFEST_FILE})")


if __name__ == "__main__":
    main()
