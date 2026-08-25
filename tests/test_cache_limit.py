import json

import numpy as np
import torch

from kvt.data import KVDump


def _dump(tmp_path, n_layers=6, n_seqs=4, per_seq=5):
    root = tmp_path / "d"
    root.mkdir()
    rng = np.random.default_rng(0)
    rows = n_seqs * per_seq
    for l in range(n_layers):
        np.savez(root / f"layer{l:02d}.npz",
                 K=rng.standard_normal((rows, 2, 8)).astype(np.float16),
                 V=rng.standard_normal((rows, 2, 8)).astype(np.float16))
    np.savez(root / "meta.npz",
             positions=np.tile(np.arange(per_seq), n_seqs).astype(np.int64),
             seq_idx=np.repeat(np.arange(n_seqs), per_seq).astype(np.int64))
    (root / "meta.json").write_text(json.dumps(
        {"n_layers": n_layers, "n_kv": 2, "d_h": 8, "rope_theta": 10000.0,
         "n_seqs": n_seqs, "stride": 1, "seq_len": per_seq, "model": "x"}))
    return KVDump.load(root)


def test_default_cache_is_unlimited(tmp_path):
    """The committed results were produced with unlimited caching; defaults must not change."""
    d = _dump(tmp_path)
    for l in range(6):
        d.get("V", l)
    assert len(d._cache) == 6


def test_cache_limit_bounds_the_cache(tmp_path):
    d = _dump(tmp_path)
    d.set_cache_limit(3)
    for l in range(6):
        d.get("V", l)
    assert len(d._cache) == 3


def test_eviction_returns_bit_identical_values(tmp_path):
    """Eviction must be invisible in the numbers -- a reloaded tensor equals the evicted one."""
    d = _dump(tmp_path)
    first = d.get("V", 0).clone()          # clone so we keep a copy after eviction
    d.set_cache_limit(2)
    for l in range(1, 6):
        d.get("V", l)
    assert ("V", 0) not in d._cache        # it really was evicted
    assert torch.equal(d.get("V", 0), first)


def test_lru_keeps_the_recently_used_entry(tmp_path):
    d = _dump(tmp_path)
    d.set_cache_limit(2)
    d.get("V", 0)
    d.get("V", 1)
    d.get("V", 0)                          # touch 0 so 1 becomes least-recent
    d.get("V", 2)
    assert ("V", 0) in d._cache
    assert ("V", 1) not in d._cache


def test_setting_a_smaller_limit_evicts_immediately(tmp_path):
    d = _dump(tmp_path)
    for l in range(6):
        d.get("V", l)
    d.set_cache_limit(2)
    assert len(d._cache) == 2


def test_limit_none_restores_unlimited(tmp_path):
    d = _dump(tmp_path)
    d.set_cache_limit(2)
    d.set_cache_limit(None)
    for l in range(6):
        d.get("V", l)
    assert len(d._cache) == 6


def test_zero_or_negative_limit_is_rejected(tmp_path):
    d = _dump(tmp_path)
    for bad in (0, -1):
        try:
            d.set_cache_limit(bad)
        except ValueError:
            continue
        raise AssertionError(f"set_cache_limit({bad}) should have raised ValueError")


def test_get_still_returns_by_reference_not_a_clone(tmp_path):
    """The no-clone contract is load-bearing for performance and must survive this change."""
    d = _dump(tmp_path)
    assert d.get("V", 0) is d.get("V", 0)
