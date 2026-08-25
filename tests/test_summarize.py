from scripts.summarize_hellaswag import render_table


def _s(*conds):
    return {c: {"n": 10, "acc": 0.5, "acc_norm": 0.5, "ci95": [0.4, 0.6],
                "retention_raw": 1.0, "retention_floor_norm": 1.0} for c in conds}


def test_table_includes_every_condition_even_unknown_ones():
    s = _s("native", "native-injected", "mapped-k1", "composed-BA-k1", "rope-k1", "weird")
    table = render_table(s)
    for cond in s:
        assert f"| {cond} |" in table, f"{cond} missing from the rendered table"


def test_table_orders_known_conditions_first():
    s = _s("mapped-k1", "native", "aaa-custom")
    # The separator row ("|---|---|...") has no space after its leading pipe, so it never
    # matches "| " and is already excluded from this filter; only the header row does. Skip
    # just the header (index 1), not header+separator (index 2).
    rows = [l for l in render_table(s).splitlines() if l.startswith("| ")][1:]
    assert rows[0].startswith("| native |")


def test_table_row_count_matches_condition_count():
    s = _s("native", "source", "composed-BA-k1", "rope-k1")
    rows = [l for l in render_table(s).splitlines() if l.startswith("| ")][1:]
    assert len(rows) == len(s)
