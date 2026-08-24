import torch

from kvt.cache import build_cache, cache_to_arrays, forward_with_cache, get_layer_kv


@torch.no_grad()
def test_layer_kv_shapes(tiny_tgt):
    ids = torch.randint(0, 256, (1, 12))
    out = tiny_tgt(input_ids=ids, use_cache=True)
    k, v = get_layer_kv(out.past_key_values, 0)
    cfg = tiny_tgt.config
    assert k.shape == (1, cfg.num_key_value_heads, 12, cfg.head_dim) and v.shape == k.shape


@torch.no_grad()
def test_injecting_own_cache_reproduces_native_logits(tiny_tgt):
    """Gate H4: the cache -> tensors -> cache -> forward path must be exact."""
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 30))
    P = 21
    native = tiny_tgt(input_ids=ids).logits                                     # [1, 30, V]
    pre = tiny_tgt(input_ids=ids[:, :P], use_cache=True)
    kvs = cache_to_arrays(pre.past_key_values, tiny_tgt.config.num_hidden_layers)
    cache = build_cache(kvs)
    suffix = forward_with_cache(tiny_tgt, cache, ids[:, P:], past_len=P)       # [1, 9, V]
    assert torch.allclose(suffix, native[:, P:], atol=1e-4)


@torch.no_grad()
def test_build_cache_expands_batch(tiny_tgt):
    ids = torch.randint(0, 256, (1, 10))
    pre = tiny_tgt(input_ids=ids, use_cache=True)
    kvs = cache_to_arrays(pre.past_key_values, tiny_tgt.config.num_hidden_layers)
    cache = build_cache(kvs, batch=4)
    k, _ = get_layer_kv(cache, 0)
    assert k.shape[0] == 4
    suffix = forward_with_cache(tiny_tgt, cache, torch.randint(0, 256, (4, 3)), past_len=10)
    assert suffix.shape == (4, 3, tiny_tgt.config.vocab_size)
