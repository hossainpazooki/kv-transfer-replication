import numpy as np
import pytest
import torch

from kvt.hellaswag import providers
from kvt.mapper import Mapper


def _ident_mapper(L_t, n_kv, d_h, theta):
    D = n_kv * d_h
    m = Mapper(k=1, selected=np.arange(L_t, dtype=np.int64).reshape(L_t, 1), n_kv=n_kv,
               d_h=d_h, src_theta=theta, tgt_theta=theta, lam=0.0, space="rope")
    for _ in range(L_t):
        m.W_K.append(np.eye(D, dtype=np.float32)); m.b_K.append(np.zeros(D, dtype=np.float32))
        m.W_V.append(np.eye(D, dtype=np.float32)); m.b_V.append(np.zeros(D, dtype=np.float32))
    return m


@torch.no_grad()
def test_extra_conditions_get_their_exact_names(tiny_src3, tiny_tgt):
    m = _ident_mapper(3, 2, 16, 10000.0)
    p = providers(tiny_src3, tiny_tgt, mappers={}, extra={"composed-BA-k1": (tiny_src3, m),
                                                          "rope-k1": (tiny_src3, m)})
    assert "composed-BA-k1" in p and "rope-k1" in p
    assert "mapped-composed-BA-k1" not in p


@torch.no_grad()
def test_extra_provider_returns_target_shaped_cache(tiny_src3, tiny_tgt):
    m = _ident_mapper(3, 2, 16, 10000.0)
    p = providers(tiny_src3, tiny_tgt, mappers={}, extra={"c": (tiny_src3, m)})
    ids = torch.randint(0, 256, (2, 5))
    kvs = p["c"](ids)
    assert len(kvs) == tiny_tgt.config.num_hidden_layers
    assert kvs[0][0].shape == (2, 2, 5, 16)


@torch.no_grad()
def test_extra_name_collision_is_rejected(tiny_src3, tiny_tgt):
    m = _ident_mapper(3, 2, 16, 10000.0)
    with pytest.raises(ValueError, match="collides"):
        providers(tiny_src3, tiny_tgt, mappers={"k1": m}, extra={"mapped-k1": (tiny_src3, m)})


@torch.no_grad()
def test_identity_still_refuses_unequal_depth(tiny_src, tiny_tgt):
    """WP1 correction: 0.6B->4B is 28->36, so there is no identity control there."""
    p = providers(tiny_src, tiny_tgt, mappers={})
    with pytest.raises(ValueError, match="equal layer counts"):
        p["identity"](torch.randint(0, 256, (1, 4)))
