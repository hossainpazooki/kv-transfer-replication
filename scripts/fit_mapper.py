import argparse
import json
import time
from pathlib import Path

import numpy as np

from kvt.data import KVDump
from kvt.mapper import Mapper, fit_mapper, mapper_r2, select_top_k
from kvt.pairs import PAIRS


def resolve_dump_root(dump_root, pair: str) -> Path:
    """Where the source/ and target/ dumps live."""
    return Path(dump_root) if dump_root else Path("data/kv") / pair


def resolve_out_dirs(pair: str, tag):
    """(mapper_dir, results_dir). Both carry the tag, so two differently-tagged runs
    cannot silently overwrite each other's artifact or report."""
    m = Path("mappers") / pair
    r = Path("results/mapper") / pair
    if tag:
        m, r = m / tag, r / tag
    return m, r


def check_tag_required(tag, dump_root, n_train, space):
    """Refuse to write the untagged output path from a run whose options change the numbers."""
    if tag is not None:
        return
    nondefault = []
    if dump_root is not None:
        nondefault.append("--dump-root")
    if n_train is not None:
        nondefault.append("--n-train")
    if space != "content":
        nondefault.append("--space")
    if nondefault:
        raise ValueError(
            f"refusing to write the untagged output path with {', '.join(nondefault)} set: "
            "mappers/<pair>/ and results/mapper/<pair>/ hold the published artifacts behind "
            "docs/ledger.md, and this run would overwrite them with different numbers under "
            "the same filenames. Pass --tag to write somewhere else.")


def resolve_masks(src, n_train, holdout, holdout_frac):
    """Training/held-out row masks. A training prefix REQUIRES a fixed held-out range:
    without one the test set would move as n_train grows, so a learning curve would
    confound 'more training data' with 'different test data'.

    An empty training or held-out set is refused rather than silently accepted: kvt.ridge.r2_score
    on an empty array returns NaN instead of raising, which would let a degenerate run report
    "success" while writing K_r2_heldout_layer_mean: NaN into r2.json."""
    if (n_train is None) != (holdout is None):
        raise ValueError("--n-train and --holdout must be given together: a training prefix "
                         "without a fixed held-out range would move the test set with n")
    if n_train is None:
        return src.split(holdout_frac)
    if n_train < 1:
        raise ValueError(f"--n-train must be >= 1, got {n_train}: a training set of size 0 "
                         "cannot fit anything, and its R^2 would silently come back as NaN")
    lo, hi = holdout
    if hi <= lo:
        raise ValueError(f"--holdout [{lo}, {hi}) is empty (hi must be > lo): a held-out set "
                         "of size 0 cannot be scored, and its R^2 would silently come back as NaN")
    if n_train > lo:
        raise ValueError(f"training prefix [0, {n_train}) overlaps held-out [{lo}, {hi})")
    tr, ho = src.seq_range_mask(0, n_train), src.seq_range_mask(lo, hi)
    if not tr.any():
        raise ValueError(f"training range [0, {n_train}) selects zero rows in a dump with "
                         f"{src.n_seqs} sequences")
    if not ho.any():
        raise ValueError(f"held-out range [{lo}, {hi}) selects zero rows in a dump with "
                         f"{src.n_seqs} sequences")
    return tr, ho


def curve_point(n_train, k, holdout, p_over_n, K_train, K_heldout, V_train, V_heldout,
                 n_train_tokens, n_heldout_tokens, space, dump_root) -> dict:
    """Build one results/curve/<pair>/n{n_train}_k{k}.json point from values already computed
    for r2.json, so the curve file and r2.json cannot disagree about what a fit measured. Schema
    matches exactly what scripts.summarize_curve.load_curve_points expects."""
    lo, hi = holdout
    return {
        "n_train": int(n_train), "k": k, "holdout": [int(lo), int(hi)],
        "p_over_n": float(p_over_n),
        "K_train": float(K_train), "K_heldout": float(K_heldout),
        "V_train": float(V_train), "V_heldout": float(V_heldout),
        "n_train_tokens": int(n_train_tokens), "n_heldout_tokens": int(n_heldout_tokens),
        "space": space, "dump_root": str(dump_root),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--k", nargs="+", default=["1", "4", "8"], help="ints or 'all'")
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--space", choices=["content", "rope"], default="content",
                    help="content: strip source RoPE, map, re-apply target RoPE (the paper's design). "
                         "rope: map rotated keys directly.")
    ap.add_argument("--n-train", type=int, default=None,
                    help="training prefix = sequences [0, N). Default: the dump's own 80/20 split.")
    ap.add_argument("--holdout", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="FIXED held-out sequence range [LO, HI). Required with --n-train so a "
                         "learning curve varies n_train only.")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output directory, e.g. n200 or rope")
    ap.add_argument("--dump-root", default=None,
                    help="directory holding source/ and target/ dumps; defaults to "
                         "data/kv/<pair>. Point this at a dump with a different sequence "
                         "count (e.g. data/kv/qwen3-0.6b-to-1.7b-n420) rather than "
                         "overwriting the default one.")
    a = ap.parse_args()
    try:
        check_tag_required(a.tag, a.dump_root, a.n_train, a.space)
    except ValueError as e:
        raise SystemExit(str(e))
    root = resolve_dump_root(a.dump_root, a.pair)
    src = KVDump.load(root / "source")
    tgt = KVDump.load(root / "target")
    probe = Path("results/probe") / a.pair
    r2_sel = 0.5 * (np.load(probe / "r2_K_stripped_train.npy") + np.load(probe / "r2_V_train.npy"))
    try:
        tr, ho = resolve_masks(src, a.n_train, a.holdout, a.holdout_frac)
    except ValueError as e:
        raise SystemExit(str(e))
    out_dir, res_dir = resolve_out_dirs(a.pair, a.tag)
    res_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "pair": a.pair, "lam": a.lam, "n_train_tokens": int(tr.sum()), "n_heldout_tokens": int(ho.sum()),
        "space": a.space, "dump_root": str(root), "n_train_seqs": a.n_train,
        "holdout": list(a.holdout) if a.holdout is not None else None,
        "n_seqs_in_dump": src.n_seqs, "k": {},
    }
    for k in a.k:
        kk = "all" if k == "all" else int(k)
        t0 = time.time()
        sel = select_top_k(r2_sel, kk)
        m = fit_mapper(src, tgt, sel, a.lam, tr, space=a.space)
        m.save(out_dir / f"k{k}")
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

    if a.n_train is not None:
        curve_dir = Path("results/curve") / a.pair
        curve_dir.mkdir(parents=True, exist_ok=True)
        for k in a.k:
            entry = report["k"][str(k)]
            kk = "all" if k == "all" else int(k)
            pt = curve_point(
                n_train=a.n_train, k=kk, holdout=a.holdout,
                p_over_n=entry["p_over_n_train"],
                K_train=entry["K_r2_train_layer_mean"], K_heldout=entry["K_r2_heldout_layer_mean"],
                V_train=entry["V_r2_train_layer_mean"], V_heldout=entry["V_r2_heldout_layer_mean"],
                n_train_tokens=report["n_train_tokens"], n_heldout_tokens=report["n_heldout_tokens"],
                space=report["space"], dump_root=report["dump_root"],
            )
            curve_path = curve_dir / f"n{a.n_train}_k{k}.json"
            curve_path.write_text(json.dumps(pt, indent=2))
            print(f"wrote {curve_path}")


if __name__ == "__main__":
    main()
