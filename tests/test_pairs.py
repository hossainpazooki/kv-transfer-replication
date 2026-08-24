import pytest
import types
from transformers import Qwen3Config

from kvt.pairs import PAIRS, KVShape, check_matched_kv, kv_shape


def _cfg(layers, hidden, heads, kv, d_h):
    return Qwen3Config(num_hidden_layers=layers, hidden_size=hidden, num_attention_heads=heads,
                       num_key_value_heads=kv, head_dim=d_h, rope_theta=1_000_000.0)


def test_registry_has_default_pair():
    p = PAIRS["qwen3-0.6b-to-1.7b"]
    assert p.source == "Qwen/Qwen3-0.6B" and p.target == "Qwen/Qwen3-1.7B"


def test_kv_shape_reads_config():
    assert kv_shape(_cfg(28, 1024, 16, 8, 128)) == KVShape(28, 8, 128, 1_000_000.0)


def test_matched_kv_passes_when_kv_heads_and_dim_agree_even_if_layers_differ():
    check_matched_kv(_cfg(28, 2048, 16, 8, 128), _cfg(36, 2560, 32, 8, 128))


def test_matched_kv_rejects_mismatch():
    with pytest.raises(ValueError, match="kv heads"):
        check_matched_kv(_cfg(28, 1024, 16, 8, 128), _cfg(28, 1024, 16, 4, 128))
    with pytest.raises(ValueError, match="head dim"):
        check_matched_kv(_cfg(28, 1024, 16, 8, 128), _cfg(28, 1024, 16, 8, 64))


def test_tiny_fixtures_are_matched_kv(tiny_src, tiny_tgt):
    check_matched_kv(tiny_src.config, tiny_tgt.config)
    assert kv_shape(tiny_src.config).n_layers == 2 and kv_shape(tiny_tgt.config).n_layers == 3


def test_rope_theta_fallback_to_plain_attribute():
    """Test _rope_theta fallback path: plain config.rope_theta attribute (older transformers)."""
    config = types.SimpleNamespace(
        num_hidden_layers=28,
        num_key_value_heads=8,
        head_dim=128,
        num_attention_heads=16,
        rope_theta=500_000.0
    )
    # Ensure rope_parameters is NOT present
    assert not hasattr(config, "rope_parameters")
    # Should read rope_theta directly
    assert kv_shape(config).rope_theta == 500_000.0


def test_rope_theta_raises_when_neither_present():
    """Test _rope_theta raises ValueError when neither rope_parameters nor rope_theta exists."""
    config = types.SimpleNamespace(
        num_hidden_layers=28,
        num_key_value_heads=8,
        head_dim=128,
        num_attention_heads=16
    )
    # Ensure neither attribute is present
    assert not hasattr(config, "rope_parameters")
    assert not hasattr(config, "rope_theta")
    # Should raise ValueError
    with pytest.raises(ValueError, match="cannot determine rope_theta"):
        kv_shape(config)
