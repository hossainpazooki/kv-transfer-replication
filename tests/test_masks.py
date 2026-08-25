import numpy as np
import pytest
import torch

from kvt.data import KVDump


def _dump(n_seqs: int = 6, per_seq: int = 4) -> KVDump:
    meta = {"n_layers": 2, "n_kv": 2, "d_h": 16, "rope_theta": 10000.0,
            "n_seqs": n_seqs, "stride": 1}
    positions = np.tile(np.arange(per_seq), n_seqs).astype(np.int64)
    seq_idx = np.repeat(np.arange(n_seqs), per_seq).astype(np.int64)
    return KVDump(root=None, meta=meta, positions=positions, seq_idx=seq_idx)


def test_seq_range_mask_selects_exactly_that_range():
    d = _dump(n_seqs=6, per_seq=4)
    m = d.seq_range_mask(2, 5)
    assert m.sum() == 3 * 4
    assert set(d.seq_idx[m].tolist()) == {2, 3, 4}


def test_seq_range_mask_is_disjoint_and_covering():
    d = _dump(n_seqs=6, per_seq=4)
    train, heldout = d.seq_range_mask(0, 4), d.seq_range_mask(4, 6)
    assert not (train & heldout).any()
    assert (train | heldout).all()


def test_nested_prefixes_share_the_same_heldout():
    """The WP2 invariant: growing the training prefix must not move the held-out set."""
    d = _dump(n_seqs=10, per_seq=4)
    heldout = d.seq_range_mask(8, 10)
    for n_train in (2, 4, 6, 8):
        train = d.seq_range_mask(0, n_train)
        assert not (train & heldout).any()
        assert train.sum() == n_train * 4


def test_seq_range_mask_rejects_out_of_bounds():
    d = _dump(n_seqs=6, per_seq=4)
    with pytest.raises(ValueError, match="invalid sequence range"):
        d.seq_range_mask(0, 7)
    with pytest.raises(ValueError, match="invalid sequence range"):
        d.seq_range_mask(4, 2)


def test_clear_cache_frees_entries_and_get_still_works(tmp_path):
    root = tmp_path / "dump"
    root.mkdir()
    rng = np.random.default_rng(0)
    for l in range(2):
        np.savez(root / f"layer{l:02d}.npz",
                 K=rng.standard_normal((24, 2, 16)).astype(np.float16),
                 V=rng.standard_normal((24, 2, 16)).astype(np.float16))
    np.savez(root / "meta.npz",
             positions=np.tile(np.arange(4), 6).astype(np.int64),
             seq_idx=np.repeat(np.arange(6), 4).astype(np.int64))
    (root / "meta.json").write_text(
        '{"n_layers": 2, "n_kv": 2, "d_h": 16, "rope_theta": 10000.0, '
        '"n_seqs": 6, "stride": 1, "seq_len": 4, "model": "x"}')
    d = KVDump.load(root)
    first = d.get("V", 0)
    assert len(d._cache) == 1
    d.clear_cache()
    assert len(d._cache) == 0
    again = d.get("V", 0)
    assert torch.equal(first, again)
