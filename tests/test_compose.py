import numpy as np
import pytest
import torch

from kvt.mapper import Mapper, apply_mapper, compose


def _rand_mapper(L_t, k, n_kv, d_h, src_theta, tgt_theta, seed, L_s=None):
    rng = np.random.default_rng(seed)
    L_s = L_s if L_s is not None else L_t
    D = n_kv * d_h
    sel = np.stack([rng.permutation(L_s)[:k] for _ in range(L_t)]).astype(np.int64)
    m = Mapper(k=k, selected=sel, n_kv=n_kv, d_h=d_h, src_theta=src_theta,
               tgt_theta=tgt_theta, lam=0.01)
    for _ in range(L_t):
        m.W_K.append((rng.standard_normal((k * D, D)) / np.sqrt(k * D)).astype(np.float32))
        m.b_K.append(rng.standard_normal(D).astype(np.float32) * 0.01)
        m.W_V.append((rng.standard_normal((k * D, D)) / np.sqrt(k * D)).astype(np.float32))
        m.b_V.append(rng.standard_normal(D).astype(np.float32) * 0.01)
    return m


def _rand_kvs(L, B, n_kv, T, d_h, seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(B, n_kv, T, d_h, generator=g),
             torch.randn(B, n_kv, T, d_h, generator=g)) for _ in range(L)]


def test_compose_shapes_and_metadata():
    a = _rand_mapper(L_t=3, k=2, n_kv=2, d_h=8, src_theta=1e4, tgt_theta=1e4, seed=0, L_s=4)
    b = _rand_mapper(L_t=5, k=2, n_kv=2, d_h=8, src_theta=1e4, tgt_theta=1e4, seed=1, L_s=3)
    c = compose(a, b)
    assert c.k == 4
    assert c.selected.shape == (5, 4)
    assert c.W_K[0].shape == (4 * 2 * 8, 2 * 8)
    assert c.src_theta == a.src_theta and c.tgt_theta == b.tgt_theta
    assert c.selected.max() < 4  # indexes into SOURCE layers, not middle


@pytest.mark.parametrize("k_a,k_b", [(1, 1), (2, 1), (1, 2), (2, 3)])
def test_closed_form_equals_operational_composition(k_a, k_b):
    """H-C3: apply_mapper(b, apply_mapper(a, x)) == apply_mapper(compose(a, b), x)."""
    n_kv, d_h, T, B, L_s, L_m, L_t = 2, 8, 6, 1, 4, 3, 5
    a = _rand_mapper(L_t=L_m, k=k_a, n_kv=n_kv, d_h=d_h, src_theta=1e4, tgt_theta=1e4,
                     seed=2, L_s=L_s)
    b = _rand_mapper(L_t=L_t, k=k_b, n_kv=n_kv, d_h=d_h, src_theta=1e4, tgt_theta=1e4,
                     seed=3, L_s=L_m)
    kvs = _rand_kvs(L_s, B, n_kv, T, d_h, seed=4)
    pos = torch.arange(T)
    mid = apply_mapper(a, kvs, pos)
    two_step = apply_mapper(b, mid, pos)
    one_step = apply_mapper(compose(a, b), kvs, pos)
    for (k2, v2), (k1, v1) in zip(two_step, one_step):
        assert torch.max(torch.abs(k2 - k1)).item() < 1e-3
        assert torch.max(torch.abs(v2 - v1)).item() < 1e-3


def test_compose_rejects_theta_mismatch():
    a = _rand_mapper(L_t=3, k=1, n_kv=2, d_h=8, src_theta=1e4, tgt_theta=1e4, seed=5)
    b = _rand_mapper(L_t=3, k=1, n_kv=2, d_h=8, src_theta=1e6, tgt_theta=1e6, seed=6)
    with pytest.raises(ValueError, match="must equal"):
        compose(a, b)


def test_compose_rejects_shape_mismatch():
    a = _rand_mapper(L_t=3, k=1, n_kv=2, d_h=8, src_theta=1e4, tgt_theta=1e4, seed=7)
    b = _rand_mapper(L_t=3, k=1, n_kv=4, d_h=8, src_theta=1e4, tgt_theta=1e4, seed=8)
    with pytest.raises(ValueError, match="shape mismatch"):
        compose(a, b)


def test_compose_rejects_selection_out_of_range():
    """b selects a middle layer that a does not produce -- silent garbage if unchecked."""
    a = _rand_mapper(L_t=2, k=1, n_kv=2, d_h=8, src_theta=1e4, tgt_theta=1e4, seed=9)
    b = _rand_mapper(L_t=3, k=1, n_kv=2, d_h=8, src_theta=1e4, tgt_theta=1e4, seed=10, L_s=5)
    b.selected[0, 0] = 4
    with pytest.raises(ValueError, match="only produces"):
        compose(a, b)


@pytest.mark.parametrize("k_a,k_b", [(1, 1), (2, 3)])
def test_compose_k_matches_selected_width(k_a, k_b):
    """c.k feeds Mapper.formula_params, a parameter-count claim used in the Appendix D
    comparison -- it must always equal the actual composed width, not be derived separately
    from a.k * b.k, which could silently disagree if either input's k ever drifted from its
    own selected.shape[1]."""
    a = _rand_mapper(L_t=3, k=k_a, n_kv=2, d_h=8, src_theta=1e4, tgt_theta=1e4, seed=20, L_s=4)
    b = _rand_mapper(L_t=2, k=k_b, n_kv=2, d_h=8, src_theta=1e4, tgt_theta=1e4, seed=21, L_s=3)
    c = compose(a, b)
    assert c.k == c.selected.shape[1]


def test_compose_handles_duplicate_middle_layer():
    """b selecting the same middle layer twice for one target layer is unreachable from
    select_top_k (which always picks distinct layers) but valid input -- the duplicated
    source layers must still be emitted and pair with the right weight blocks."""
    n_kv, d_h, T, B, L_s, L_m, L_t = 2, 8, 6, 1, 4, 3, 2
    a = _rand_mapper(L_t=L_m, k=1, n_kv=n_kv, d_h=d_h, src_theta=1e4, tgt_theta=1e4,
                     seed=11, L_s=L_s)
    b = _rand_mapper(L_t=L_t, k=2, n_kv=n_kv, d_h=d_h, src_theta=1e4, tgt_theta=1e4,
                     seed=12, L_s=L_m)
    b.selected[0] = np.array([1, 1])  # duplicate middle layer for target layer 0
    kvs = _rand_kvs(L_s, B, n_kv, T, d_h, seed=13)
    pos = torch.arange(T)
    mid = apply_mapper(a, kvs, pos)
    two_step = apply_mapper(b, mid, pos)
    one_step = apply_mapper(compose(a, b), kvs, pos)
    for (k2, v2), (k1, v1) in zip(two_step, one_step):
        assert torch.max(torch.abs(k2 - k1)).item() < 1e-3
        assert torch.max(torch.abs(v2 - v1)).item() < 1e-3
