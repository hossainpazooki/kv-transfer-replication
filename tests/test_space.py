import json

import numpy as np
import pytest
import torch

from kvt.mapper import Mapper, apply_mapper, fit_mapper, mapper_r2


def test_default_space_is_content():
    m = Mapper(k=1, selected=np.zeros((2, 1), dtype=np.int64), n_kv=2, d_h=8,
               src_theta=1e4, tgt_theta=1e4, lam=0.01)
    assert m.space == "content"


def test_load_of_a_legacy_mapper_without_space_defaults_to_content(tmp_path):
    """The mappers already on disk (k1/k4/k8) have no 'space' key. They must still load."""
    m = Mapper(k=1, selected=np.zeros((1, 1), dtype=np.int64), n_kv=2, d_h=4,
               src_theta=1e4, tgt_theta=1e4, lam=0.01)
    m.W_K.append(np.eye(8, dtype=np.float32)); m.b_K.append(np.zeros(8, dtype=np.float32))
    m.W_V.append(np.eye(8, dtype=np.float32)); m.b_V.append(np.zeros(8, dtype=np.float32))
    p = tmp_path / "legacy"
    m.save(p)
    meta = json.loads(p.with_suffix(".json").read_text())
    del meta["space"]
    p.with_suffix(".json").write_text(json.dumps(meta))
    assert Mapper.load(p).space == "content"


def test_space_round_trips_through_save_and_load(tmp_path):
    m = Mapper(k=1, selected=np.zeros((1, 1), dtype=np.int64), n_kv=2, d_h=4,
               src_theta=1e4, tgt_theta=1e4, lam=0.01, space="rope")
    m.W_K.append(np.eye(8, dtype=np.float32)); m.b_K.append(np.zeros(8, dtype=np.float32))
    m.W_V.append(np.eye(8, dtype=np.float32)); m.b_V.append(np.zeros(8, dtype=np.float32))
    p = tmp_path / "rope"
    m.save(p)
    assert Mapper.load(p).space == "rope"


def test_rope_space_apply_skips_strip_and_reapply():
    """An identity rope-space mapper returns its input keys unchanged; an identity
    content-space mapper does not (it strips at src_theta and re-applies at tgt_theta)."""
    n_kv, d_h, T = 2, 8, 5
    D = n_kv * d_h
    def _ident(space, src_theta, tgt_theta):
        m = Mapper(k=1, selected=np.zeros((1, 1), dtype=np.int64), n_kv=n_kv, d_h=d_h,
                   src_theta=src_theta, tgt_theta=tgt_theta, lam=0.0, space=space)
        m.W_K.append(np.eye(D, dtype=np.float32)); m.b_K.append(np.zeros(D, dtype=np.float32))
        m.W_V.append(np.eye(D, dtype=np.float32)); m.b_V.append(np.zeros(D, dtype=np.float32))
        return m
    g = torch.Generator().manual_seed(0)
    kvs = [(torch.randn(1, n_kv, T, d_h, generator=g), torch.randn(1, n_kv, T, d_h, generator=g))]
    pos = torch.arange(T)
    k_rope = apply_mapper(_ident("rope", 1e4, 1e4), kvs, pos)[0][0]
    assert torch.max(torch.abs(k_rope - kvs[0][0])).item() < 1e-5
    k_content = apply_mapper(_ident("content", 1e4, 1e6), kvs, pos)[0][0]
    assert torch.max(torch.abs(k_content - kvs[0][0])).item() > 1e-3


def test_fit_in_rope_space_uses_rotated_keys(tmp_path):
    """A rope-space fit must consume K_rope; a content-space fit must consume K_stripped.
    With a self-mapping dump the two fits produce different weights unless theta is 0."""
    from kvt.data import KVDump
    root = tmp_path / "d"
    root.mkdir()
    rng = np.random.default_rng(0)
    n_rows = 40
    for l in range(2):
        np.savez(root / f"layer{l:02d}.npz",
                 K=rng.standard_normal((n_rows, 2, 8)).astype(np.float16),
                 V=rng.standard_normal((n_rows, 2, 8)).astype(np.float16))
    np.savez(root / "meta.npz",
             positions=np.tile(np.arange(10), 4).astype(np.int64),
             seq_idx=np.repeat(np.arange(4), 10).astype(np.int64))
    (root / "meta.json").write_text(json.dumps(
        {"n_layers": 2, "n_kv": 2, "d_h": 8, "rope_theta": 10000.0,
         "n_seqs": 4, "stride": 1, "seq_len": 10, "model": "x"}))
    d = KVDump.load(root)
    sel = np.zeros((2, 1), dtype=np.int64)
    mask = d.seq_range_mask(0, 3)
    m_content = fit_mapper(d, d, sel, 0.01, mask, space="content")
    m_rope = fit_mapper(d, d, sel, 0.01, mask, space="rope")
    assert m_content.space == "content" and m_rope.space == "rope"
    assert not np.allclose(m_content.W_K[0], m_rope.W_K[0], atol=1e-4)
    # H-G3: V has no RoPE, so the V weights must be IDENTICAL between the two spaces.
    assert np.allclose(m_content.W_V[0], m_rope.W_V[0], atol=1e-6)
    r_content = mapper_r2(m_content, d, d, mask)
    r_rope = mapper_r2(m_rope, d, d, mask)
    assert np.allclose(r_content["V"], r_rope["V"], atol=1e-6)
