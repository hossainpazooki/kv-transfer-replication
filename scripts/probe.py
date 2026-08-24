import argparse
import json
import time
from pathlib import Path

import numpy as np

from kvt.data import KVDump
from kvt.pairs import PAIRS
from kvt.ridge import probe_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    a = ap.parse_args()
    src = KVDump.load(Path("data/kv") / a.pair / "source")
    tgt = KVDump.load(Path("data/kv") / a.pair / "target")
    out = Path("results/probe") / a.pair
    out.mkdir(parents=True, exist_ok=True)
    tr, ho = src.split(a.holdout_frac)
    summary = {"pair": a.pair, "n_seqs": src.n_seqs, "stride": src.stride, "holdout_frac": a.holdout_frac,
               "n_train_tokens": int(tr.sum()), "n_heldout_tokens": int(ho.sum()),
               "p_over_n_probe": src.d_h / int(tr.sum()), "kinds": {}}
    for kind in KVDump.KINDS:
        t0 = time.time()
        r = probe_r2(src, tgt, kind, a.holdout_frac)
        for split in ("train", "heldout"):
            np.save(out / f"r2_{kind}_{split}.npy", r[split])
        best = np.unravel_index(np.argmax(r["heldout"]), r["heldout"].shape)
        L = min(src.n_layers, tgt.n_layers)
        summary["kinds"][kind] = {
            "best_heldout_cell": {"src_layer": int(best[0]), "tgt_layer": int(best[1]),
                                  "train": float(r["train"][best]), "heldout": float(r["heldout"][best])},
            "max_train": float(r["train"].max()),
            "diag_mean_train": float(np.mean([r["train"][i, i] for i in range(L)])),
            "diag_mean_heldout": float(np.mean([r["heldout"][i, i] for i in range(L)])),
            "mean_gap_train_minus_heldout": float((r["train"] - r["heldout"]).mean()),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{kind}: best heldout cell {best} train={r['train'][best]:.3f} heldout={r['heldout'][best]:.3f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
