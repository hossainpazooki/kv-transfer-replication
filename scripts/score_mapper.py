"""Score an EXISTING mapper on a pair of dumps, without refitting.

Added for linear-ceiling's E8 (its ledger entry 0016): `kvt.mapper.mapper_r2` was
library-only, so there was no way to evaluate a fitted mapper on KV states from a different
text distribution by subprocess. Held out by sequence exactly as `fit_mapper.py` does
(`KVDump.split(holdout_frac)` keeps the LAST ceil(frac * n_seqs) sequences), so scoring the
archived dumps with the archived mapper reproduces the archived `r2.json` held-out numbers.

Refuses (never writes a partial file) when the dumps' KV shapes disagree with the mapper, or
when the held-out mask is empty.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from kvt.data import KVDump
from kvt.mapper import Mapper, mapper_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", required=True, help="path without suffix, e.g. mappers/<pair>/k1")
    ap.add_argument("--src", required=True, help="source-model dump directory")
    ap.add_argument("--tgt", required=True, help="target-model dump directory")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--out", required=True, help="r2.json to write")
    a = ap.parse_args()
    m = Mapper.load(a.mapper)
    src, tgt = KVDump.load(a.src), KVDump.load(a.tgt)
    if src.n_seqs != tgt.n_seqs or src.stride != tgt.stride:
        raise SystemExit(f"dumps disagree: src n_seqs={src.n_seqs} stride={src.stride}, "
                         f"tgt n_seqs={tgt.n_seqs} stride={tgt.stride}")
    if tgt.n_kv != m.n_kv or tgt.d_h != m.d_h or tgt.n_layers != len(m.W_K):
        raise SystemExit(f"mapper/target shape mismatch: mapper n_kv={m.n_kv} d_h={m.d_h} "
                         f"L={len(m.W_K)}, target n_kv={tgt.n_kv} d_h={tgt.d_h} L={tgt.n_layers}")
    tr, ho = src.split(a.holdout_frac)
    if not ho.any() or not tr.any():
        raise SystemExit("empty training or held-out mask; refusing to score")
    t0 = time.time()
    r2_ho = mapper_r2(m, src, tgt, ho)
    r2_tr = mapper_r2(m, src, tgt, tr)
    rec = {
        "mapper": str(Path(a.mapper)), "src": str(Path(a.src)), "tgt": str(Path(a.tgt)),
        "holdout_frac": a.holdout_frac, "n_seqs": int(src.n_seqs), "stride": int(src.stride),
        "n_train_tokens": int(tr.sum()), "n_heldout_tokens": int(ho.sum()),
        "k": m.k, "space": getattr(m, "space", "content"),
        "K_r2_train_layer_mean": float(np.mean(r2_tr["K"])),
        "K_r2_heldout_layer_mean": float(np.mean(r2_ho["K"])),
        "V_r2_train_layer_mean": float(np.mean(r2_tr["V"])),
        "V_r2_heldout_layer_mean": float(np.mean(r2_ho["V"])),
        "K_r2_heldout_per_layer": [float(x) for x in r2_ho["K"]],
        "V_r2_heldout_per_layer": [float(x) for x in r2_ho["V"]],
        "seconds": time.time() - t0,
    }
    for k, v in rec.items():
        if isinstance(v, float) and not np.isfinite(v):
            raise SystemExit(f"non-finite {k}; refusing to write")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}: k={m.k} heldout K={rec['K_r2_heldout_layer_mean']:.4f} "
          f"V={rec['V_r2_heldout_layer_mean']:.4f} in {rec['seconds']:.0f}s")


if __name__ == "__main__":
    main()
