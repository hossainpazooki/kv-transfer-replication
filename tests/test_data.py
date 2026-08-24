import json
import numpy as np
import torch

from kvt.data import KVDump, dump_kv
from kvt.rope import apply_rope_tokens_first
from kvt.cache import get_layer_kv
from kvt.ridge import probe_r2


@torch.no_grad()
def test_dump_roundtrip_and_stride(tmp_path, tiny_tgt, tiny_tokens):
    seqs = tiny_tokens(n_seqs=3, seq_len=16)
    dump_kv(tiny_tgt, seqs, stride=4, out_dir=tmp_path)
    d = KVDump.load(tmp_path)
    assert d.n_layers == 3 and d.n_kv == 2 and d.d_h == 16 and d.n_seqs == 3
    assert d.positions.tolist() == [0, 4, 8, 12] * 3
    assert d.seq_idx.tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert d.get("K_rope", 0).shape == (12, 2, 16)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["stride"] == 4 and meta["seq_len"] == 16


@torch.no_grad()
def test_dump_matches_live_cache_and_strip_is_consistent(tmp_path, tiny_tgt, tiny_tokens):
    seqs = tiny_tokens(n_seqs=1, seq_len=16)
    dump_kv(tiny_tgt, seqs, stride=4, out_dir=tmp_path)
    d = KVDump.load(tmp_path)
    out = tiny_tgt(input_ids=torch.tensor(seqs[:1]), use_cache=True)
    k_live, v_live = get_layer_kv(out.past_key_values, 1)              # [1, n_kv, 16, d_h]
    k_live = k_live[0, :, ::4].transpose(0, 1)                         # [4, n_kv, d_h]
    assert torch.allclose(d.get("K_rope", 1), k_live, atol=1e-3, rtol=2e-3)       # fp16 storage tolerance
    assert torch.allclose(d.get("V", 1), v_live[0, :, ::4].transpose(0, 1), atol=1e-3, rtol=2e-3)
    re_roped = apply_rope_tokens_first(d.get("K_stripped", 1), d.positions, d.rope_theta)
    assert torch.allclose(re_roped, d.get("K_rope", 1), atol=1e-5)


def test_split_is_by_sequence(tmp_path, tiny_tgt, tiny_tokens):
    seqs = tiny_tokens(n_seqs=5, seq_len=8)
    dump_kv(tiny_tgt, seqs, stride=2, out_dir=tmp_path)
    d = KVDump.load(tmp_path)
    tr, ho = d.split(holdout_frac=0.2)
    assert tr.sum() == 16 and ho.sum() == 4                            # 4 tokens/seq; last 1 of 5 seqs held out
    assert set(d.seq_idx[ho].tolist()) == {4}
    assert not (tr & ho).any()


@torch.no_grad()
def test_consumers_do_not_mutate_the_cached_dump_tensors(tmp_path, tiny_tgt, tiny_tokens):
    """Verify that consumers (e.g. probe_r2) do not mutate cached tensors returned by get()."""
    # Build a small dump using the fixtures.
    seqs = tiny_tokens(n_seqs=3, seq_len=12)
    dump_kv(tiny_tgt, seqs, stride=3, out_dir=tmp_path)
    d = KVDump.load(tmp_path)

    # Record copies of tensors returned by get() for a couple of (kind, layer) combinations.
    k_rope_0_before = d.get("K_rope", 0).clone()
    k_stripped_1_before = d.get("K_stripped", 1).clone()
    v_0_before = d.get("V", 0).clone()

    # Run the ridge probe consumer, which reads tensors in bulk and is a realistic
    # high-volume consumer of the cached data.
    probe_r2(d, d, "K_stripped", holdout_frac=0.2)

    # Assert that the cached tensors returned by get() are still exactly equal
    # to the recorded copies. If the consumer had mutated any in-place, this would fail.
    assert torch.allclose(d.get("K_rope", 0), k_rope_0_before, atol=0, rtol=0)
    assert torch.allclose(d.get("K_stripped", 1), k_stripped_1_before, atol=0, rtol=0)
    assert torch.allclose(d.get("V", 0), v_0_before, atol=0, rtol=0)
