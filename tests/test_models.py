import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import kvt.models as models
from kvt.data import get_layer_kv


def _tiny(attn: str) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      rope_theta=10000.0, tie_word_embeddings=True, attn_implementation=attn)
    return Qwen3ForCausalLM(cfg).eval()


def test_repeat_kv_attention_is_registered_and_matches_eager():
    """The registered implementation must reproduce eager attention (up to float32 rounding) on a
    GQA model, and must be what load_model asks for."""
    assert models.ATTN_IMPLEMENTATION == "sdpa_repeat_kv"
    eager, ours = _tiny("eager"), _tiny(models.ATTN_IMPLEMENTATION)
    ours.load_state_dict(eager.state_dict())
    assert ours.config._attn_implementation == models.ATTN_IMPLEMENTATION
    ids = torch.randint(0, 256, (1, 40))
    with torch.no_grad():
        a = eager(input_ids=ids, use_cache=True)
        b = ours(input_ids=ids, use_cache=True, logits_to_keep=1)
    assert torch.allclose(a.logits[:, -1:], b.logits, atol=1e-5)
    assert b.logits.shape[1] == 1
    for l in range(2):
        ka, va = get_layer_kv(a.past_key_values, l)
        kb, vb = get_layer_kv(b.past_key_values, l)
        assert torch.allclose(ka, kb, atol=1e-5) and torch.allclose(va, vb, atol=1e-5)
