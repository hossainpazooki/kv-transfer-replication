import numpy as np
import torch

from kvt.cache import build_cache, cache_to_arrays, forward_with_cache
from kvt.data import KVDump, dump_kv
from kvt.mapper import Mapper, apply_mapper, fit_mapper, mapper_r2, select_top_k
from kvt.ridge import probe_r2


def test_select_top_k_orders_by_r2_and_supports_all():
    r2 = np.array([[0.1, 0.9], [0.5, 0.2], [0.7, 0.4]])      # [L_s=3, L_t=2]
    sel = select_top_k(r2, 2)
    assert sel.tolist() == [[2, 1], [0, 2]]
    assert select_top_k(r2, "all").shape == (2, 3)


def test_formula_matches_paper_table12():
    # Qwen3 14B -> 32B, k=8: 2 * 64 * 8 * (8*8*128) * 128 = 1.07 B (Appendix D, Table 12)
    assert Mapper.formula_params(L_t=64, n_kv=8, k=8, d_h=128) == 1_073_741_824
    # our pair, k=8
    assert Mapper.formula_params(L_t=28, n_kv=8, k=8, d_h=128) == 469_762_048


@torch.no_grad()
def test_self_mapper_is_a_noop_through_inject(tmp_path, tiny_tgt, tiny_tokens):
    """Gate: fit target->target at k=1, lam=0. Top-1 selection must pick the same layer, W ~ I,
    and strip-RoPE -> map -> re-RoPE -> inject must reproduce native logits."""
    seqs = tiny_tokens(n_seqs=12, seq_len=64)
    dump_kv(tiny_tgt, seqs, stride=1, out_dir=tmp_path)
    d = KVDump.load(tmp_path)
    tr, _ = d.split(0.2)
    r2_sel = 0.5 * (probe_r2(d, d, "K_stripped", 0.2)["train"] + probe_r2(d, d, "V", 0.2)["train"])
    sel = select_top_k(r2_sel, 1)
    assert sel[:, 0].tolist() == [0, 1, 2]
    m = fit_mapper(d, d, sel, lam=0.0, train_mask=tr)
    assert m.n_weight_params() == Mapper.formula_params(3, 2, 1, 16)
    r = mapper_r2(m, d, d, ~tr)
    assert np.allclose(r["K"], 1.0, atol=1e-3) and np.allclose(r["V"], 1.0, atol=1e-3)

    ids = torch.randint(0, 256, (1, 40)); P = 30
    native = tiny_tgt(input_ids=ids).logits
    pre = tiny_tgt(input_ids=ids[:, :P], use_cache=True)
    kvs = cache_to_arrays(pre.past_key_values, 3)
    mapped = apply_mapper(m, kvs, torch.arange(P))
    suffix = forward_with_cache(tiny_tgt, build_cache(mapped), ids[:, P:], past_len=P)
    assert torch.allclose(suffix, native[:, P:], atol=1e-2)


@torch.no_grad()
def test_self_mapper_is_a_noop_at_k_gt_1_pinning_feature_order(tmp_path, tiny_tgt, tiny_tokens):
    """Gate: fit target->target at k=3 (all 3 source layers), lam=0. With k=1 the layer ordering
    is trivially a no-op (there's only one layer to order), so that test can't detect a fit-vs-
    inference feature-layout mismatch. At k>1, build_features() (fit time) and apply_mapper's
    feats() (inference time) must concatenate the selected source layers in the SAME order, or
    the learned weights get applied to a column layout that doesn't match what they were fit on
    -- silently, with no error, just garbage predictions. Using all 3 layers here makes the test
    sensitive to both a layer-order mismatch and a head-order mismatch between the two feature
    builders."""
    seqs = tiny_tokens(n_seqs=12, seq_len=64)
    dump_kv(tiny_tgt, seqs, stride=1, out_dir=tmp_path)
    d = KVDump.load(tmp_path)
    tr, _ = d.split(0.2)
    r2_sel = 0.5 * (probe_r2(d, d, "K_stripped", 0.2)["train"] + probe_r2(d, d, "V", 0.2)["train"])
    sel = select_top_k(r2_sel, 3)
    assert sel.shape == (3, 3)  # [L_t=3, k=3]: all source layers, per target layer
    m = fit_mapper(d, d, sel, lam=0.0, train_mask=tr)
    assert m.n_weight_params() == Mapper.formula_params(3, 2, 3, 16)
    r = mapper_r2(m, d, d, ~tr)
    assert np.allclose(r["K"], 1.0, atol=1e-3) and np.allclose(r["V"], 1.0, atol=1e-3)

    ids = torch.randint(0, 256, (1, 40)); P = 30
    native = tiny_tgt(input_ids=ids).logits
    pre = tiny_tgt(input_ids=ids[:, :P], use_cache=True)
    kvs = cache_to_arrays(pre.past_key_values, 3)
    mapped = apply_mapper(m, kvs, torch.arange(P))
    suffix = forward_with_cache(tiny_tgt, build_cache(mapped), ids[:, P:], past_len=P)
    assert torch.allclose(suffix, native[:, P:], atol=1e-2)


@torch.no_grad()
def test_cross_model_mapper_shapes_and_save_load(tmp_path, tiny_src, tiny_tgt, tiny_tokens):
    seqs = tiny_tokens(n_seqs=8, seq_len=48)
    dump_kv(tiny_src, seqs, stride=1, out_dir=tmp_path / "s")
    dump_kv(tiny_tgt, seqs, stride=1, out_dir=tmp_path / "t")
    s, t = KVDump.load(tmp_path / "s"), KVDump.load(tmp_path / "t")
    tr, ho = s.split(0.25)
    sel = select_top_k(np.ones((2, 3)), 2)
    m = fit_mapper(s, t, sel, lam=0.01, train_mask=tr)
    assert m.W_K[0].shape == (2 * 2 * 16, 2 * 16)
    m.save(tmp_path / "m")
    m2 = Mapper.load(tmp_path / "m")
    assert np.array_equal(m2.selected, sel) and np.allclose(m2.W_V[2], m.W_V[2])
    ids = torch.randint(0, 256, (2, 9))
    kvs = cache_to_arrays(tiny_src(input_ids=ids, use_cache=True).past_key_values, 2)
    out = apply_mapper(m2, kvs, torch.arange(9))
    assert len(out) == 3 and out[0][0].shape == (2, 2, 9, 16)
