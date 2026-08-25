"""Recompute every HellaSwag number from the raw per-example JSONL. Never restate from stdout."""
import argparse
import json
from pathlib import Path

from kvt.hellaswag import summarize_records
from kvt.pairs import PAIRS
from kvt.stats import correctness_by_idx, mcnemar_exact


def load_and_validate_records(root: Path, expect_n: int | None = None) -> dict[str, list[dict]]:
    """Load every condition's *.jsonl under root and enforce the like-for-like guarantees
    that retention numbers depend on. Raises ValueError (never returns NaN or a silent
    truncation) if any of those guarantees don't hold."""
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise ValueError(f"no .jsonl files found in {root}; nothing to summarize")

    records = {}
    for p in files:
        recs = []
        for lineno, line in enumerate(p.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"malformed JSON in {p} at line {lineno}: {e}") from e
        records[p.stem] = recs

    if "native" not in records:
        raise ValueError(
            f"native.jsonl is missing from {root} (found conditions: {sorted(records)}); "
            "retention is defined relative to the native condition and cannot be computed without it"
        )

    # Zero-record conditions must halt loudly here: an unconditional check, not something the
    # count/idx-set comparisons below could accidentally let through (empty vs. empty compares
    # "equal" to those checks and would otherwise flow straight into a full-NaN summary).
    empty_conditions = sorted(cond for cond, recs in records.items() if len(recs) == 0)
    if empty_conditions:
        raise ValueError(
            f"condition(s) in {root} produced zero scored examples: {empty_conditions}; "
            "a run that scored nothing must not be summarized as a valid empty result"
        )

    # Duplicate idx within one condition can collapse to a set that matches another condition's
    # idx set even though the record count differs from the distinct-idx count. Check this before
    # the set-equality comparison below, which is blind to duplicates.
    for cond, recs in records.items():
        idxs = [r["idx"] for r in recs]
        if len(idxs) != len(set(idxs)):
            seen = set()
            dups = []
            for i in idxs:
                if i in seen and i not in dups:
                    dups.append(i)
                seen.add(i)
            raise ValueError(
                f"condition '{cond}' has duplicate idx values, which would corrupt the "
                f"like-for-like comparison against native: {dups[:5]}{' ...' if len(dups) > 5 else ''}"
            )

    native_recs = records["native"]
    native_n = len(native_recs)

    if expect_n is not None:
        bad = {cond: len(recs) for cond, recs in records.items() if len(recs) != expect_n}
        if bad:
            raise ValueError(
                f"--expect-n={expect_n} but these conditions have a different record count: {bad}"
            )
    else:
        bad = {cond: len(recs) for cond, recs in records.items()
               if cond != "native" and len(recs) != native_n}
        if bad:
            raise ValueError(
                f"conditions have mismatched record counts (native has {native_n} records): {bad}; "
                "a truncated run must not be reported as a smaller n with no complaint"
            )

    native_idx = [r["idx"] for r in native_recs]
    native_idx_set = set(native_idx)
    for cond, recs in records.items():
        if cond == "native":
            continue
        idx_set = {r["idx"] for r in recs}
        if idx_set != native_idx_set:
            missing = sorted(native_idx_set - idx_set)
            extra = sorted(idx_set - native_idx_set)
            raise ValueError(
                f"condition '{cond}' scored a different set of examples than 'native' "
                f"(native n={len(native_idx_set)}, {cond} n={len(idx_set)}); "
                f"missing from '{cond}' (present in native): {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}; "
                f"extra in '{cond}' (absent from native): {extra[:5]}"
                f"{' ...' if len(extra) > 5 else ''}"
            )

    return records


KNOWN_ORDER = ["native", "native-injected", "source", "identity"]


def render_table(s: dict) -> str:
    """Render EVERY condition in `s`, known ones first in a fixed order and the rest sorted.

    The previous version iterated a hardcoded list plus `mapped-*`, so any condition named
    anything else (composed-*, rope-*) was computed into summary.json and then silently
    dropped from the human-read summary.md. A report that omits a condition without saying
    so is the same fail-open class as emitting a NaN row.
    """
    rest = sorted(c for c in s if c not in KNOWN_ORDER)
    lines = ["| condition | n | acc | acc_norm | 95% CI | retention raw | retention floor-norm |",
             "|---|---|---|---|---|---|---|"]
    for cond in [c for c in KNOWN_ORDER if c in s] + rest:
        r = s[cond]
        lines.append(
            f"| {cond} | {r['n']} | {r['acc']:.3f} | {r['acc_norm']:.3f} "
            f"| [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] "
            f"| {r.get('retention_raw', float('nan')):.3f} "
            f"| {r.get('retention_floor_norm', float('nan')):.3f} |")
    return "\n".join(lines) + "\n"


def build_comparisons(records: dict[str, list[dict]], pairs: list[tuple[str, str]]) -> list[dict]:
    """One McNemar-exact comparison per (a, b) pair in `pairs`, using acc_norm-based
    correctness (kvt.stats.correctness_by_idx(norm=True)) to match the repo's headline
    metric. `records` comes from load_and_validate_records, which already guarantees every
    condition scored the identical example set -- the precondition mcnemar_exact enforces
    for a paired test. Refuses (never silently skips) a pair naming an absent condition."""
    present = sorted(records)
    out = []
    for a_name, b_name in pairs:
        missing = [c for c in (a_name, b_name) if c not in records]
        if missing:
            raise ValueError(
                f"--compare {a_name} {b_name}: condition(s) {missing} not present in {sorted(records)}; "
                f"conditions present: {present}")
        corr_a = correctness_by_idx(records[a_name], norm=True)
        corr_b = correctness_by_idx(records[b_name], norm=True)
        r = mcnemar_exact(corr_a, corr_b)
        out.append({"a": a_name, "b": b_name, "acc_a": r["acc_a"], "acc_b": r["acc_b"],
                     "b_count": r["b"], "c_count": r["c"], "n_discordant": r["n_discordant"],
                     "p": r["p"]})
    return out


def render_comparisons_table(comparisons: list[dict]) -> str:
    lines = [
        "",
        "Paired comparisons (McNemar exact test; correctness is acc_norm-based -- NOT raw accuracy):",
        "| a | b | acc_a | acc_b | b (a right, b wrong) | c (a wrong, b right) | n discordant | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c['a']} | {c['b']} | {c['acc_a']:.3f} | {c['acc_b']:.3f} "
            f"| {c['b_count']} | {c['c_count']} | {c['n_discordant']} | {c['p']:.4g} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--chance", type=float, default=0.25)
    ap.add_argument("--expect-n", type=int, default=None,
                     help="require every condition to have exactly this many records")
    ap.add_argument("--compare", nargs=2, action="append", default=None, metavar=("A", "B"),
                    help="paired McNemar-exact comparison between two condition names, e.g. "
                         "--compare composed-BA-k1 mapped-C-k1. Repeatable.")
    a = ap.parse_args()
    root = Path("results/hellaswag") / a.pair
    records = load_and_validate_records(root, a.expect_n)
    s = summarize_records(records, a.chance)
    table = render_table(s)
    md = table
    output = dict(s)
    if a.compare:
        try:
            comparisons = build_comparisons(records, [tuple(p) for p in a.compare])
        except ValueError as e:
            raise SystemExit(str(e))
        output["comparisons"] = comparisons
        md = table + render_comparisons_table(comparisons)
    (root / "summary.json").write_text(json.dumps(output, indent=2))
    (root / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
