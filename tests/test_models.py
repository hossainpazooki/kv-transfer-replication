import pytest
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


def _avail(monkeypatch, cuda: bool, mps: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.delenv("KVT_DEVICE", raising=False)


def test_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    _avail(monkeypatch, cuda=True, mps=True)
    assert models.device().type == "cuda"
    _avail(monkeypatch, cuda=False, mps=True)
    assert models.device().type == "mps"
    _avail(monkeypatch, cuda=False, mps=False)
    assert models.device().type == "cpu"


def test_device_env_override_wins_over_availability(monkeypatch):
    _avail(monkeypatch, cuda=True, mps=True)
    monkeypatch.setenv("KVT_DEVICE", "cpu")
    assert models.device().type == "cpu"


def test_device_env_override_rejects_unknown_value(monkeypatch):
    _avail(monkeypatch, cuda=False, mps=False)
    monkeypatch.setenv("KVT_DEVICE", "tpu")
    with pytest.raises(ValueError, match="cuda, mps, cpu"):
        models.device()


def test_load_model_dtype_bfloat16_and_rejects_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("KVT_DEVICE", "cpu")
    _tiny("eager").save_pretrained(tmp_path)
    m = models.load_model(str(tmp_path), dtype="bfloat16")
    assert all(p.dtype == torch.bfloat16 for p in m.parameters())
    assert models.load_model(str(tmp_path)).lm_head.weight.dtype == torch.float32
    with pytest.raises(ValueError, match="int8"):
        models.load_model(str(tmp_path), dtype="int8")
