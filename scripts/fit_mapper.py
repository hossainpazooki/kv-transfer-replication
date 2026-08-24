import argparse
import json
import time
from pathlib import Path

import numpy as np

from kvt.data import KVDump
from kvt.mapper import Mapper, fit_mapper, mapper_r2, select_top_k
from kvt.pairs import PAIRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--k", nargs="+", default=["1", "4", "8"], help="ints or 'all'")
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    a = ap.parse_args()
    src = KVDump.load(Path("data/kv") / a.pair / "source")
    tgt = KVDump.load(Path("data/kv") / a.pair / "target")
    probe = Path("results/probe") / a.pair
    r2_sel = 0.5 * (np.load(probe / "r2_K_stripped_train.npy") + np.load(probe / "r2_V_train.npy"))
    tr, ho = src.split(a.holdout_frac)
    res_dir = Path("results/mapper") / a.pair
    res_dir.mkdir(parents=True, exist_ok=True)
    report = {"pair": a.pair, "lam": a.lam, "n_train_tokens": int(tr.sum()), "n_heldout_tokens": int(ho.sum()), "k": {}}
    for k in a.k:
        kk = "all" if k == "all" else int(k)
        t0 = time.time()
        sel = select_top_k(r2_sel, kk)
        m = fit_mapper(src, tgt, sel, a.lam, tr)
        m.save(Path("mappers") / a.pair / f"k{k}")
        r_tr, r_ho = mapper_r2(m, src, tgt, tr), mapper_r2(m, src, tgt, ho)
        p = sel.shape[1] * src.n_kv * src.d_h
        report["k"][str(k)] = {
            "p_features": int(p), "p_over_n_train": p / int(tr.sum()),
            "n_weight_params": m.n_weight_params(),
            "formula_params": Mapper.formula_params(tgt.n_layers, tgt.n_kv, sel.shape[1], tgt.d_h),
            "K_r2_train_layer_mean": float(r_tr["K"].mean()), "K_r2_heldout_layer_mean": float(r_ho["K"].mean()),
            "V_r2_train_layer_mean": float(r_tr["V"].mean()), "V_r2_heldout_layer_mean": float(r_ho["V"].mean()),
            "K_r2_heldout_per_layer": r_ho["K"].tolist(), "V_r2_heldout_per_layer": r_ho["V"].tolist(),
            "selected": sel.tolist(), "seconds": round(time.time() - t0, 1),
        }
        print(f"k={k}: p={p} p/n={p / int(tr.sum()):.3f} K r2 train={r_tr['K'].mean():.3f} "
              f"heldout={r_ho['K'].mean():.3f} | V train={r_tr['V'].mean():.3f} heldout={r_ho['V'].mean():.3f}")
    (res_dir / "r2.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {res_dir / 'r2.json'}")


if __name__ == "__main__":
    main()
