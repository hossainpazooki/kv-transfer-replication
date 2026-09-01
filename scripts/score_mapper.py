"""Score an EXISTING mapper on a pair of dumps, without refitting.

Added for linear-ceiling's E8 (its ledger entry 0016): `kvt.mapper.mapper_r2` was
library-only, so there was no way to evaluate a fitted mapper on KV states from a different
text distribution by subprocess. Held out by sequence exactly as `fit_mapper.py` does
(`KVDump.split(holdout_frac)` keeps the LAST ceil(frac * n_seqs) sequences), so scoring the
archived dumps with the archived mapper reproduces the archived `r2.json` held-out numbers.

`--per-token` (linear-ceiling entry 0023, tau calibration): additionally write the held-out
per-token, per-layer, per-head squared deviations K_sq, V_sq and reference norms ref_K, ref_V
[n_heldout, L, n_kv] float32, with sst_K, sst_V [L, n_kv] float64, so a reader can recompute the
head-averaged held-out R^2 from them; the json then also carries the A5-pooled-over-heads
per-layer R^2 as a labelled diagnostic (it is NOT the reported figure). The per-token path
recomputes the held-out R^2 independently of `mapper_r2` and refuses if the two disagree.

Refuses (never writes a partial file) when the dumps' KV shapes disagree with the mapper, or
when the held-out mask is empty.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from kvt.data import KVDump
from kvt.mapper import Mapper, build_features, mapper_r2
from kvt.pertoken import moments, pooled_r2
from kvt.ridge import _np, predict


def per_token_heldout(m: Mapper, src: KVDump, tgt: KVDump, mask: np.ndarray) -> tuple[dict, dict]:
    """Per-token squares on the masked rows; returns (arrays, r2) where r2 carries the
    head-averaged per-layer R^2 recomputed from the squares and the pooled diagnostic."""
    n = int(mask.sum())
    sq, ref, sst, head_mean, pooled = {}, {}, {}, {}, {}
    for kind, Ws, bs, key in (("K_stripped", m.W_K, m.b_K, "K"), ("V", m.W_V, m.b_V, "V")):
        sq[key], ref[key], sst[key], head_mean[key], pooled[key] = [], [], [], [], []
        for lt in range(len(Ws)):
            X = build_features(src, m.selected[lt], kind, mask)
            Yhat = predict(X, Ws[lt], bs[lt])
            Y = _np(tgt.get(kind, lt))[mask].reshape(n, -1)
            rec, s, r = moments(Y, Yhat, m.n_kv, m.d_h)
            sq[key].append(s)
            ref[key].append(r)
            sst[key].append(rec["sst"])
            head_mean[key].append(rec["r2_head_mean"])
            pooled[key].append(pooled_r2(Y, Yhat))
    arrays = {"K_sq": np.stack(sq["K"], 1).astype(np.float32), "V_sq": np.stack(sq["V"], 1).astype(np.float32),
              "ref_K": np.stack(ref["K"], 1).astype(np.float32), "ref_V": np.stack(ref["V"], 1).astype(np.float32),
              "sst_K": np.asarray(sst["K"], dtype=np.float64), "sst_V": np.asarray(sst["V"], dtype=np.float64),
              "n_heldout": np.asarray(n, dtype=np.int64)}
    return arrays, {"head_mean": head_mean, "pooled": pooled}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", required=True, help="path without suffix, e.g. mappers/<pair>/k1")
    ap.add_argument("--src", required=True, help="source-model dump directory")
    ap.add_argument("--tgt", required=True, help="target-model dump directory")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--out", required=True, help="r2.json to write")
    ap.add_argument("--per-token", default=None, help="optional .npz for held-out per-token squares (entry 0023)")
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
    }
    if a.per_token:
        arrays, r2pt = per_token_heldout(m, src, tgt, ho)
        for key in ("K", "V"):
            if not np.allclose(r2pt["head_mean"][key], r2_ho[key], rtol=0, atol=1e-9):
                raise SystemExit(f"per-token path disagrees with mapper_r2 on held-out {key}; refusing to write")
        for name, arr in arrays.items():
            if not np.isfinite(arr).all():
                raise SystemExit(f"non-finite per-token value in {name}; refusing to write")
        pt = Path(a.per_token)
        pt.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pt, **arrays)
        rec["per_token"] = {"path": pt.name, "sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
                            "dtype": "float32 squares; float64 sst",
                            "layout": "[n_heldout, n_layers, n_kv]; sse[l][h] == sum over tokens (float64)"}
        rec["K_r2_heldout_pooled_over_heads_per_layer"] = r2pt["pooled"]["K"]
        rec["V_r2_heldout_pooled_over_heads_per_layer"] = r2pt["pooled"]["V"]
        rec["K_r2_heldout_pooled_over_heads_layer_mean"] = float(np.mean(r2pt["pooled"]["K"]))
        rec["V_r2_heldout_pooled_over_heads_layer_mean"] = float(np.mean(r2pt["pooled"]["V"]))
    rec["seconds"] = time.time() - t0
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
