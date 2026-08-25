"""Paired significance testing for condition-vs-condition comparisons.

Wilson intervals (kvt.hellaswag._wilson) are UNPAIRED: at n=500 they are about +/-4 pp
and two conditions differing by 3 pp will overlap regardless of how consistent the
difference is per example. Every condition here scores the SAME examples, so the paired
test is both available and far more powerful. Use this for condition-vs-condition;
use Wilson for reporting a single condition's uncertainty.
"""
import math


def correctness_by_idx(records: list[dict], norm: bool = True) -> dict[int, bool]:
    """Map example idx -> was this condition correct on it.

    norm=True reproduces acc_norm (byte-length-normalized argmax), which is the headline
    metric in docs/ledger.md; norm=False reproduces raw acc.
    """
    out: dict[int, bool] = {}
    for r in records:
        idx = int(r["idx"])
        if idx in out:
            raise ValueError(f"duplicate idx {idx} in records; cannot pair conditions safely")
        lp = [float(x) for x in r["logprobs"]]
        if norm:
            nb = [float(x) for x in r["nbytes"]]
            if any(x <= 0 for x in nb):
                raise ValueError(f"non-positive byte length in record idx {idx}: {nb}")
            score = [a / b for a, b in zip(lp, nb)]
        else:
            score = lp
        out[idx] = score.index(max(score)) == int(r["gold"])
    return out


def mcnemar_exact(a: dict[int, bool], b: dict[int, bool]) -> dict:
    """Two-sided exact McNemar test over paired per-example correctness.

    b_count = examples a got right and b got wrong; c_count = the reverse. Under the null
    each discordant pair is a fair coin, so the p-value is an exact binomial tail. Returns
    p=1.0 when there are no discordant pairs (nothing to distinguish the conditions).
    """
    if set(a) != set(b):
        raise ValueError(
            f"conditions scored different example sets: {len(set(a) - set(b))} only in a, "
            f"{len(set(b) - set(a))} only in b; a paired test is undefined")
    b_count = sum(1 for i in a if a[i] and not b[i])
    c_count = sum(1 for i in a if b[i] and not a[i])
    n = b_count + c_count
    if n == 0:
        p = 1.0
    else:
        k = min(b_count, c_count)
        tail = sum(math.comb(n, i) for i in range(k + 1))
        p = min(1.0, 2.0 * tail / (2 ** n))
    return {"n_pairs": len(a), "b": b_count, "c": c_count, "n_discordant": n, "p": p,
            "acc_a": sum(a.values()) / len(a) if a else float("nan"),
            "acc_b": sum(b.values()) / len(b) if b else float("nan")}
