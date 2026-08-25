"""WP2: recompute the learning-curve table from per-point JSON. Fails closed on a
curve whose points do not share one held-out range -- that would not be a curve in n."""
import argparse
import json
from pathlib import Path


def load_curve_points(root: Path) -> list[dict]:
    files = sorted(Path(root).glob("*.json"))
    pts = [json.loads(p.read_text()) for p in files]
    if not pts:
        raise ValueError(f"no curve points found in {root}; nothing to summarize")
    if len(pts) < 2:
        raise ValueError(
            f"only {len(pts)} curve point(s) in {root}; a learning curve needs at least 2. "
            "A single point cannot show a trend and its held-out-consistency check is vacuous.")
    holdouts = {tuple(p["holdout"]) for p in pts}
    if len(holdouts) != 1:
        raise ValueError(
            f"curve points use different held-out ranges {sorted(holdouts)}; a learning curve "
            "must vary ONLY n_train, or the points are not comparable")
    # .get(...) (not p[...]): older/hand-written points may lack these keys entirely, which must
    # not be conflated with an actual mismatch -- only an ACTUAL disagreement in value is refused.
    spaces = {p.get("space") for p in pts}
    if len(spaces) != 1:
        raise ValueError(
            f"curve points use different feature spaces {sorted(spaces, key=str)}; mixing "
            "content-space and rope-space points is not a curve in n")
    dump_roots = {p.get("dump_root") for p in pts}
    if len(dump_roots) != 1:
        raise ValueError(
            f"curve points come from different dump root(s) {sorted(dump_roots, key=str)}; "
            "points fitted against different dumps are not comparable")
    return sorted(pts, key=lambda p: (p["n_train"], p["k"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    args = ap.parse_args()
    root = Path("results/curve") / args.pair
    pts = load_curve_points(root)
    lines = ["| n_train | k | p/n | K train | K held-out | V train | V held-out |",
             "|---|---|---|---|---|---|---|"]
    for p in pts:
        lines.append(f"| {p['n_train']} | {p['k']} | {p['p_over_n']:.3f} | {p['K_train']:.4f} "
                     f"| {p['K_heldout']:.4f} | {p['V_train']:.4f} | {p['V_heldout']:.4f} |")
    table = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
