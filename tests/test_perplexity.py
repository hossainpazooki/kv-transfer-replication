import numpy as np
import pytest
import torch

from kvt.hellaswag import providers
from kvt.perplexity import nll_native, nll_with_provider, slice_for_prefix


@torch.no_grad()
def test_native_injected_reproduces_native(tiny_tgt):
    """The WP3 gate at tiny scale: injecting the target's OWN cache must reproduce
    a plain forward pass. If this fails nothing else in WP3 is interpretable."""
    ids = np.random.default_rng(0).integers(0, 256, size=48, dtype=np.int64)
    P = 16
    s_nat, n_nat = nll_native(tiny_tgt, ids, P)
    prov = providers(tiny_tgt, tiny_tgt, mappers={})
    s_inj, n_inj = nll_with_provider(tiny_tgt, ids, P, prov["native-injected"])
    assert n_nat == n_inj == len(ids) - P
    assert abs(s_nat - s_inj) < 1e-3


@torch.no_grad()
def test_returns_sum_and_count_not_a_mean(tiny_tgt):
    ids = np.random.default_rng(1).integers(0, 256, size=40, dtype=np.int64)
    s, n = nll_native(tiny_tgt, ids, 8)
    assert n == 32
    assert s > 0  # NLL of a random-init model on random tokens is large and positive


@torch.no_grad()
def test_rejects_prefix_longer_than_window(tiny_tgt):
    ids = np.random.default_rng(2).integers(0, 256, size=20, dtype=np.int64)
    with pytest.raises(ValueError, match="prefix_len"):
        nll_native(tiny_tgt, ids, 20)


@torch.no_grad()
def test_identity_differs_from_native(tiny_src3, tiny_tgt):
    """A control: a foreign cache must NOT reproduce the native score, or the harness
    is not actually using the injected cache."""
    ids = np.random.default_rng(3).integers(0, 256, size=48, dtype=np.int64)
    P = 16
    s_nat, _ = nll_native(tiny_tgt, ids, P)
    prov = providers(tiny_src3, tiny_tgt, mappers={})
    s_id, _ = nll_with_provider(tiny_tgt, ids, P, prov["identity"])
    assert abs(s_nat - s_id) > 1e-2


def test_windows_are_deterministic_and_disjoint():
    from kvt.perplexity import _chunk
    ids = np.arange(100, dtype=np.int64)
    w = _chunk(ids, n_windows=3, total_len=30)
    assert w.shape == (3, 30)
    assert np.array_equal(w[0], np.arange(0, 30))
    assert np.array_equal(w[1], np.arange(30, 60))
    assert np.array_equal(_chunk(ids, 3, 30), w)


def test_chunk_refuses_when_not_enough_tokens():
    from kvt.perplexity import _chunk
    with pytest.raises(ValueError, match="only enough for"):
        _chunk(np.arange(50, dtype=np.int64), n_windows=3, total_len=30)


def test_fixed_continuation_windows_sizes_for_the_largest_prefix(monkeypatch):
    """windows are drawn once at max_prefix + cont_len, not re-chunked per P."""
    from kvt import perplexity

    captured = {}

    def fake_wikitext_windows(tokenizer, n_windows, total_len):
        captured["total_len"] = total_len
        captured["n_windows"] = n_windows
        return np.tile(np.arange(total_len, dtype=np.int64), (n_windows, 1))

    monkeypatch.setattr(perplexity, "wikitext_windows", fake_wikitext_windows)
    windows, max_prefix = perplexity.fixed_continuation_windows(
        tokenizer=None, n_windows=5, prefix_lens=[512, 1024, 2048], cont_len=256)
    assert max_prefix == 2048
    assert windows.shape == (5, 2304)
    assert captured["total_len"] == 2304
    assert captured["n_windows"] == 5


def test_slice_for_prefix_shares_the_same_continuation_across_prefix_lengths():
    """The property the fix exists to guarantee: the final cont_len tokens are identical
    regardless of which P is sliced out."""
    max_prefix, cont_len = 2048, 256
    window = np.arange(max_prefix + cont_len, dtype=np.int64)
    tail = None
    for P in (512, 1024, 2048):
        sliced = slice_for_prefix(window, max_prefix, P)
        assert len(sliced) == P + cont_len
        this_tail = sliced[-cont_len:]
        if tail is None:
            tail = this_tail
        else:
            assert np.array_equal(tail, this_tail)


def test_slice_for_prefix_rejects_out_of_range_prefix():
    window = np.arange(2048 + 256, dtype=np.int64)
    with pytest.raises(ValueError, match="prefix_len"):
        slice_for_prefix(window, max_prefix=2048, prefix_len=0)
    with pytest.raises(ValueError, match="prefix_len"):
        slice_for_prefix(window, max_prefix=2048, prefix_len=2049)
