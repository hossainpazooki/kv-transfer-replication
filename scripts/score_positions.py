"""Score KV agreement at matched positions across two prefills, and a mapper's transfer to them.

Added for linear-ceiling's E9 (its ledger entry 0019). Inputs are three stride-1 single-sequence
dumps made by `dump_kv.py` and a pairs file of matched token positions:

  --same-src   receiver model prefilling the SENDER context S
  --same-tgt   receiver model prefilling the RECEIVER prompt R
  --cross-src  source model prefilling S (optional; enables the cross measurement)
  --pairs      .npz with `pairs` [n, 2] int64: (p_S, p_R) per matched token

E9-same: the receiver's content-space K (K_stripped) and V at rows p_S of `same-src` are
compared against its own rows p_R of `same-tgt` -- how much of a matched token's KV survives a
different preceding context. E9-cross: the mapper's prediction from the source model's rows p_S
compared against the receiver's rows p_R (content space; the mapper is applied exactly as
`fit_mapper` scored it, feature layout via `build_features`).

Per target layer and per head this writes SSE and SST alongside R² (definition A5 per head,
head-averaged per layer, layer-averaged to one scalar), so a reader can recompute every R² from
the recorded moments. Refuses -- writes nothing -- on empty pairs, shape mismatches, an
out-of-range position, or any non-finite number.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from kvt.data import KVDump
from kvt.mapper import Mapper, build_features
from kvt.ridge import _np, predict

CACHE_LIMIT = 8


def _rows(dump: KVDump, kind: str, layer: int, pos: np.ndarray) -> np.ndarray:
    t = _np(dump.get(kind, layer))            # [T, n_kv, d_h], stride-1 single sequence
    if pos.max() >= t.shape[0]:
        raise SystemExit(f"position {int(pos.max())} out of range for dump with T={t.shape[0]}")
    return t[pos].reshape(len(pos), -1)       # [n, n_kv*d_h]


def _moments(Y: np.ndarray, Yhat: np.ndarray, n_kv: int, d_h: int) -> dict:
    sse, sst, r2 = [], [], []
    Y64, H64 = Y.astype(np.float64), Yhat.astype(np.float64)
    for h in range(n_kv):
        y = Y64[:, h * d_h:(h + 1) * d_h]
        e = float(((y - H64[:, h * d_h:(h + 1) * d_h]) ** 2).sum())
        t = float(((y - y.mean(0)) ** 2).sum())
        sse.append(e)
        sst.append(t)
        r2.append(1.0 - e / t)
    return {"sse": sse, "sst": sst, "r2_head_mean": float(np.mean(r2))}


def score_same(src: KVDump, tgt: KVDump, p_s: np.ndarray, p_r: np.ndarray) -> dict:
    out = {"K": [], "V": []}
    for l in range(tgt.n_layers):
        for kind, key in (("K_stripped", "K"), ("V", "V")):
            Yhat = _rows(src, kind, l, p_s)
            Y = _rows(tgt, kind, l, p_r)
            out[key].append(_moments(Y, Yhat, tgt.n_kv, tgt.d_h))
    return out


def score_cross(m: Mapper, cross_src: KVDump, tgt: KVDump, p_s: np.ndarray, p_r: np.ndarray) -> dict:
    mask = np.zeros(int(cross_src.positions.shape[0]), dtype=bool)
    mask[p_s] = True
    order = np.argsort(p_s)                   # build_features returns rows in position order
    inv = np.argsort(order)
    out = {"K": [], "V": []}
    for lt in range(len(m.W_K)):
        for kind, W, b, key in (("K_stripped", m.W_K[lt], m.b_K[lt], "K"),
                                ("V", m.W_V[lt], m.b_V[lt], "V")):
            X = build_features(cross_src, m.selected[lt], kind, mask)[inv]
            Yhat = predict(X, W, b)
            Y = _rows(tgt, kind, lt, p_r)
            out[key].append(_moments(Y, Yhat, tgt.n_kv, tgt.d_h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--same-src", required=True)
    ap.add_argument("--same-tgt", required=True)
    ap.add_argument("--cross-src", default=None)
    ap.add_argument("--mapper", default=None, help="required with --cross-src")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = np.load(a.pairs)["pairs"]
    if pairs.size == 0:
        raise SystemExit("empty pairs; nothing to score")
    p_s, p_r = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
    s_src, s_tgt = KVDump.load(a.same_src), KVDump.load(a.same_tgt)
    for d in (s_src, s_tgt):
        d.set_cache_limit(CACHE_LIMIT)
        if d.stride != 1 or d.n_seqs != 1:
            raise SystemExit(f"{d.root}: E9 dumps must be stride-1 single sequences "
                             f"(got stride={d.stride}, n_seqs={d.n_seqs})")
    if (s_src.n_layers, s_src.n_kv, s_src.d_h) != (s_tgt.n_layers, s_tgt.n_kv, s_tgt.d_h):
        raise SystemExit("same-src and same-tgt dumps disagree in shape; not the same model?")
    t0 = time.time()
    rec = {"n_pairs": int(pairs.shape[0]),
           "same": score_same(s_src, s_tgt, p_s, p_r)}
    if a.cross_src:
        if not a.mapper:
            raise SystemExit("--cross-src requires --mapper")
        m = Mapper.load(a.mapper)
        c_src = KVDump.load(a.cross_src)
        c_src.set_cache_limit(CACHE_LIMIT)
        if c_src.stride != 1 or c_src.n_seqs != 1:
            raise SystemExit(f"{c_src.root}: E9 dumps must be stride-1 single sequences")
        rec["cross"] = score_cross(m, c_src, s_tgt, p_s, p_r)
        rec["mapper"] = {"path": str(Path(a.mapper)), "k": m.k, "space": getattr(m, "space", "content")}
    for part in ("same", "cross"):
        if part not in rec:
            continue
        for key in ("K", "V"):
            rec[f"{part}_{key}_r2_layer_mean"] = float(np.mean([l["r2_head_mean"] for l in rec[part][key]]))
    rec["seconds"] = time.time() - t0

    def walk(o):
        if isinstance(o, float) and not np.isfinite(o):
            raise SystemExit("non-finite value; refusing to write")
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(rec)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec))
    print(f"wrote {out}: n_pairs={rec['n_pairs']} same K={rec['same_K_r2_layer_mean']:.4f} "
          f"V={rec['same_V_r2_layer_mean']:.4f}"
          + (f" cross K={rec['cross_K_r2_layer_mean']:.4f} V={rec['cross_V_r2_layer_mean']:.4f}" if a.cross_src else "")
          + f" in {rec['seconds']:.0f}s")


if __name__ == "__main__":
    main()
