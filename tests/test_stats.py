import pytest

from kvt.stats import correctness_by_idx, mcnemar_exact


def _recs(rows):
    """rows: list of (idx, logprobs, nbytes, gold)."""
    return [{"idx": i, "logprobs": lp, "nbytes": nb, "gold": g} for i, lp, nb, g in rows]


def test_correctness_by_idx_uses_length_normalized_argmax():
    recs = _recs([(0, [-1.0, -2.0], [1, 10], 1), (1, [-1.0, -2.0], [1, 1], 0)])
    c = correctness_by_idx(recs, norm=True)
    assert c == {0: True, 1: True}
    raw = correctness_by_idx(recs, norm=False)
    assert raw == {0: False, 1: True}


def test_correctness_by_idx_rejects_duplicates():
    recs = _recs([(0, [-1.0, -2.0], [1, 1], 0), (0, [-1.0, -2.0], [1, 1], 0)])
    with pytest.raises(ValueError, match="duplicate"):
        correctness_by_idx(recs)


def test_mcnemar_rejects_mismatched_example_sets():
    a = {0: True, 1: False}
    b = {0: True, 2: False}
    with pytest.raises(ValueError, match="different example sets"):
        mcnemar_exact(a, b)


def test_mcnemar_identical_conditions_give_p_one():
    a = {i: i % 2 == 0 for i in range(20)}
    r = mcnemar_exact(a, dict(a))
    assert r["n_discordant"] == 0 and r["p"] == 1.0


def test_mcnemar_all_discordant_one_way_is_significant():
    a = {i: True for i in range(10)}
    b = {i: False for i in range(10)}
    r = mcnemar_exact(a, b)
    assert r["b"] == 10 and r["c"] == 0
    assert r["p"] == pytest.approx(2.0 / 2 ** 10)
    assert r["acc_a"] == 1.0 and r["acc_b"] == 0.0


def test_mcnemar_known_value():
    """b=8, c=2, n=10: two-sided exact p = 2*P(X<=2) = 2*(1+10+45)/1024."""
    a, b = {}, {}
    for i in range(8):
        a[i], b[i] = True, False
    for i in range(8, 10):
        a[i], b[i] = False, True
    for i in range(10, 30):
        a[i], b[i] = True, True
    r = mcnemar_exact(a, b)
    assert r["b"] == 8 and r["c"] == 2 and r["n_discordant"] == 10
    assert r["p"] == pytest.approx(2.0 * (1 + 10 + 45) / 1024)


def test_mcnemar_p_is_capped_at_one():
    a, b = {}, {}
    for i in range(5):
        a[i], b[i] = True, False
    for i in range(5, 10):
        a[i], b[i] = False, True
    r = mcnemar_exact(a, b)
    assert r["p"] == 1.0
