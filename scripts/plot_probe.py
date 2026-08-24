"""Figure 2 reproduction: 3 kinds x {train, heldout} heatmaps of head-averaged R^2, shared color scale."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kvt.data import KVDump
from kvt.pairs import PAIRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    a = ap.parse_args()
    root = Path("results/probe") / a.pair
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for j, kind in enumerate(KVDump.KINDS):
        for i, split in enumerate(("train", "heldout")):
            r = np.load(root / f"r2_{kind}_{split}.npy")
            ax = axes[i, j]
            im = ax.imshow(r.T, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
            ax.set_title(f"{kind} / {split}  (max {r.max():.2f})")
            ax.set_xlabel("source layer"); ax.set_ylabel("target layer")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="head-averaged R^2")
    fig.suptitle(f"Single-source OLS probe, {a.pair}")
    fig.savefig(root / "figure2.png", dpi=130)
    print(f"wrote {root / 'figure2.png'}")


if __name__ == "__main__":
    main()
