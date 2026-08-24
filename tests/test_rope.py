import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from kvt.pairs import kv_shape
from kvt.rope import apply_rope, rope_cos_sin, strip_rope, strip_rope_tokens_first, apply_rope_tokens_first


def test_strip_is_exact_inverse_of_apply():
    torch.manual_seed(0)
    T, d_h = 37, 16
    x = torch.randn(2, T, d_h)
    pos = torch.arange(T)
    cos, sin = rope_cos_sin(pos, d_h, theta=10000.0)
    back = strip_rope(apply_rope(x, cos, sin), cos, sin)
    assert torch.allclose(back, x, atol=1e-6)


def test_apply_matches_hf_qwen3_rotary(tiny_src):
    torch.manual_seed(1)
    cfg = tiny_src.config
    T, d_h = 20, cfg.head_dim
    x = torch.randn(1, cfg.num_key_value_heads, T, d_h)
    pos = torch.arange(T)[None]
    cos_hf, sin_hf = tiny_src.model.rotary_emb(x, pos)         # [1, T, d_h]
    _, k_hf = apply_rotary_pos_emb(x, x, cos_hf, sin_hf)
    cos, sin = rope_cos_sin(pos[0], d_h, theta=kv_shape(cfg).rope_theta)
    k_mine = apply_rope(x, cos, sin)
    assert torch.allclose(k_mine, k_hf, atol=1e-5)


def test_tokens_first_helpers_roundtrip_with_strided_positions():
    torch.manual_seed(2)
    k = torch.randn(10, 2, 16)                      # [T, n_kv, d_h]
    pos = torch.arange(0, 40, 4)                    # stride-4 positions, as stored in dumps
    k_rope = apply_rope_tokens_first(k, pos, theta=10000.0)
    assert not torch.allclose(k_rope, k)
    assert torch.allclose(strip_rope_tokens_first(k_rope, pos, theta=10000.0), k, atol=1e-6)
