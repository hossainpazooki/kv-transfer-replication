"""WP3: recompute pooled perplexity from raw per-window JSONL, and enforce the
native-injected == native correctness gate that scripts/eval_perplexity.py writes records
for but never itself checks. Never restate a number from stdout."""
import argparse
import json
import math
from pathlib import Path


def load_and_validate_records(root: Path) -> dict[str, list[dict]]:
    """Load every condition's *.jsonl under root (one prefix-length directory) and enforce
    the like-for-like guarantees pooled perplexity depends on. Raises ValueError (never
    returns a partial or silently-truncated result) if any guarantee doesn't hold."""
    files = sorted(Path(root).glob("*.jsonl"))
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
            "perplexity and the correctness gate are both defined relative to native")

    if "native-injected" not in records:
        raise ValueError(
            f"native-injected.jsonl is missing from {root}; the native-injected == native "
            "correctness gate cannot be skipped -- rerun eval_perplexity.py or point at a "
            "root that has it")

    empty = sorted(cond for cond, recs in records.items() if len(recs) == 0)
    if empty:
        raise ValueError(
            f"condition(s) in {root} produced zero scored windows: {empty}; a run that "
            "scored nothing must not be summarized as a valid empty result")

    for cond, recs in records.items():
        wids = [r["window"] for r in recs]
        if len(wids) != len(set(wids)):
            raise ValueError(f"condition '{cond}' in {root} has duplicate window ids")

    native_wids = {r["window"] for r in records["native"]}
    for cond, recs in records.items():
        if cond == "native":
            continue
        wids = {r["window"] for r in recs}
        if wids != native_wids:
            missing = sorted(native_wids - wids)
            extra = sorted(wids - native_wids)
            raise ValueError(
                f"condition '{cond}' in {root} scored a different window set than 'native' "
                f"(native n={len(native_wids)}, {cond} n={len(wids)}); "
                f"missing from '{cond}': {missing[:5]}{' ...' if len(missing) > 5 else ''}; "
                f"extra in '{cond}': {extra[:5]}{' ...' if len(extra) > 5 else ''}")

    return records


def _pooled_nll(recs: list[dict]) -> tuple[float, int]:
    return sum(r["sum_nll"] for r in recs), sum(r["n_tokens"] for r in recs)


def pooled_perplexity(recs: list[dict]) -> float:
    """exp(sum_nll / sum_tokens) pooled over windows. NOT the mean of per-window
    perplexities -- that would overweight short windows and is a different, wrong quantity."""
    total_nll, total_tokens = _pooled_nll(recs)
    if total_tokens == 0:
        raise ValueError("zero total tokens across windows; cannot compute perplexity")
    return math.exp(total_nll / total_tokens)


def enforce_native_injected_gate(records: dict[str, list[dict]], rtol: float = 1e-4):
    """native-injected must reproduce native to float noise. This is WP3's stated
    invariant, made a hard gate rather than a pair of numbers nobody compares."""
    nat_nll, _ = _pooled_nll(records["native"])
    inj_nll, _ = _pooled_nll(records["native-injected"])
    diff_ok = (inj_nll == 0) if nat_nll == 0 else (abs(inj_nll - nat_nll) / abs(nat_nll) <= rtol)
    if not diff_ok:
        raise ValueError(
            f"native-injected != native beyond rtol={rtol}: native total sum_nll={nat_nll!r}, "
            f"native-injected total sum_nll={inj_nll!r}; the injected cache is not reproducing "
            "a plain forward pass, so nothing else in this run is interpretable")


def summarize_prefix_dir(root: Path) -> dict[str, dict]:
    records = load_and_validate_records(root)
    enforce_native_injected_gate(records)
    ppls = {cond: pooled_perplexity(recs) for cond, recs in records.items()}
    native_ppl = ppls["native"]
    return {
        cond: {
            "perplexity": ppl,
            "pct_degradation_vs_native": (ppl - native_ppl) / native_ppl * 100.0,
        }
        for cond, ppl in ppls.items()
    }


def render_table(summary: dict[str, dict[str, dict]]) -> str:
    lines = ["| prefix_len | condition | perplexity | % degradation vs native |",
             "|---|---|---|---|"]
    for P in sorted(summary, key=int):
        rows = summary[P]
        for cond in ["native"] + sorted(c for c in rows if c != "native"):
            r = rows[cond]
            lines.append(f"| {P} | {cond} | {r['perplexity']:.4f} "
                         f"| {r['pct_degradation_vs_native']:.2f}% |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    a = ap.parse_args()
    root = Path("results/perplexity") / a.pair
    p_dirs = sorted((d for d in root.iterdir() if d.is_dir() and d.name.startswith("P")),
                    key=lambda d: int(d.name[1:]))
    if not p_dirs:
        raise SystemExit(f"no P<n> directories found under {root}")

    summary = {}
    for d in p_dirs:
        P = d.name[1:]
        try:
            summary[P] = summarize_prefix_dir(d)
        except ValueError as e:
            raise SystemExit(str(e))

    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    table = render_table(summary)
    (root / "summary.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
