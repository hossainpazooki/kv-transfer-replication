"""RoPE spec (linear-ceiling E9-long, 2026-09-09): a scaled receiver (YaRN) must round-trip through
dump -> KVDump.K_stripped exactly, and the spec must reproduce the model's own rotary embedding."""
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from kvt.cache import get_layer_kv
from kvt.data import KVDump, dump_kv
from kvt.models import load_model, scaled_config
from kvt.rope import (RopeSpec, apply_rope_spec_tokens_first, check_rope_spec_against_model, rope_cos_sin,
                      strip_rope_spec_tokens_first, strip_rope_tokens_first)

YARN = {"rope_type": "yarn", "factor": 2.5, "original_max_position_embeddings": 32768}
FAR_POSITIONS = [0, 1, 1000, 32767, 32768, 40960, 65535, 80000, 81919]


def _qwen_like(rope_scaling=None, layers=2):
    torch.manual_seed(0)
    kw = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
              num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=40960,
              rope_theta=1e6, tie_word_embeddings=True, attn_implementation="eager")
    if rope_scaling:
        kw["rope_scaling"] = dict(rope_scaling)
        kw["max_position_embeddings"] = int(rope_scaling["original_max_position_embeddings"] * rope_scaling["factor"])
    return Qwen3ForCausalLM(Qwen3Config(**kw)).eval()


def test_default_spec_equals_plain_theta_rotation():
    spec = RopeSpec.default(16, 1e6)
    assert spec.attention_scaling == 1.0 and len(spec.inv_freq) == 8
    pos = torch.tensor(FAR_POSITIONS)
    cos, sin = rope_cos_sin(pos, 16, 1e6)
    c2, s2 = spec.cos_sin(pos)
    assert torch.equal(cos, c2) and torch.equal(sin, s2)
    k = torch.randn(len(pos), 2, 16)
    assert torch.allclose(strip_rope_spec_tokens_first(k, pos, spec), strip_rope_tokens_first(k, pos, 1e6))


def test_yarn_spec_strip_is_exact_inverse_at_scaled_positions():
    model = _qwen_like(YARN)
    spec = RopeSpec.from_model(model)
    assert spec.attention_scaling == pytest.approx(0.1 * np.log(2.5) + 1.0)    # YaRN mscale
    assert spec.parameters["rope_type"] == "yarn" and spec.parameters["factor"] == 2.5
    pos = torch.tensor(FAR_POSITIONS)
    k = torch.randn(len(pos), 2, 16)
    k_rope = apply_rope_spec_tokens_first(k, pos, spec)
    assert not torch.allclose(k_rope, k)
    assert torch.allclose(strip_rope_spec_tokens_first(k_rope, pos, spec), k, atol=1e-5)
    # the plain-theta strip is WRONG under YaRN: different frequencies and a factor m left in
    assert not torch.allclose(strip_rope_tokens_first(k_rope, pos, 1e6), k, atol=1e-2)


@pytest.mark.parametrize("scaling", [None, YARN])
def test_spec_reproduces_hf_rotary_at_far_positions(scaling):
    model = _qwen_like(scaling)
    spec = RopeSpec.from_model(model)
    pos = torch.tensor(FAR_POSITIONS)
    worst = check_rope_spec_against_model(model, spec, pos, atol=1e-5)
    assert worst <= 1e-5
    # and against a rotary embedding built from the config alone, at every position of an 82k window
    emb = Qwen3RotaryEmbedding(model.config)
    allpos = torch.arange(0, 81920, 7)
    cos_hf, sin_hf = emb(torch.zeros(1, 1), allpos[None])
    cos, sin = spec.cos_sin(allpos)
    assert float((cos_hf[0] - cos * spec.attention_scaling).abs().max()) <= 1e-5
    assert float((sin_hf[0] - sin * spec.attention_scaling).abs().max()) <= 1e-5


def test_check_halts_on_a_wrong_spec():
    model = _qwen_like(YARN)
    good = RopeSpec.from_model(model)
    wrong = RopeSpec(good.inv_freq, 1.0, good.parameters)          # attention factor dropped
    with pytest.raises(ValueError, match="does not reproduce"):
        check_rope_spec_against_model(model, wrong, torch.tensor([0, 5, 100]), atol=1e-5)
    wrong2 = RopeSpec.default(16, 1e6)                             # plain-theta frequencies
    with pytest.raises(ValueError, match="does not reproduce"):
        check_rope_spec_against_model(model, wrong2, torch.tensor([0, 5, 100]), atol=1e-5)


@torch.no_grad()
def test_yarn_dump_records_spec_and_strips_to_content(tmp_path):
    model = _qwen_like(YARN)
    rng = np.random.default_rng(0)
    seqs = rng.integers(0, 256, size=(1, 24), dtype=np.int64)
    dump_kv(model, seqs, stride=1, out_dir=tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["rope"]["parameters"]["rope_type"] == "yarn" and meta["rope"]["attention_scaling"] > 1.0
    assert meta["rope"]["check_max_abs"] <= meta["rope"]["check_atol"] and meta["rope"]["max_position_embeddings"] == 81920
    d = KVDump.load(tmp_path)
    assert d.rope_spec.attention_scaling == pytest.approx(meta["rope"]["attention_scaling"])
    # K_rope is what the model wrote; K_stripped re-roped under the recorded spec reproduces it
    re_roped = apply_rope_spec_tokens_first(d.get("K_stripped", 1), d.positions, d.rope_spec)
    assert torch.allclose(re_roped, d.get("K_rope", 1), atol=1e-4)
    # and the content-space K is the model's pre-rotation key: k_norm(W_k h), computed by hand
    out = model(input_ids=torch.tensor(seqs), use_cache=True, output_hidden_states=True)
    h = out.hidden_states[1]                                                      # input to layer 1
    layer = model.model.layers[1]
    x = layer.input_layernorm(h)
    k_pre = layer.self_attn.k_norm(layer.self_attn.k_proj(x).view(1, 24, 2, 16))[0]   # [T, n_kv, d_h]
    assert torch.allclose(d.get("K_stripped", 1), k_pre, atol=2e-3, rtol=2e-3)     # fp16 storage tolerance


@torch.no_grad()
def test_native_dump_meta_gains_the_spec_and_strips_as_before(tmp_path, tiny_tgt, tiny_tokens):
    seqs = tiny_tokens(n_seqs=1, seq_len=16)
    dump_kv(tiny_tgt, seqs, stride=4, out_dir=tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["rope"]["attention_scaling"] == 1.0 and meta["rope"]["parameters"]["rope_type"] == "default"
    d = KVDump.load(tmp_path)
    old = strip_rope_tokens_first(d.get("K_rope", 1), d.positions, d.rope_theta)
    assert torch.allclose(d.get("K_stripped", 1), old, atol=1e-6)
    # an archived dump without the block loads and strips identically (backward compatibility)
    meta.pop("rope")
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    d2 = KVDump.load(tmp_path)
    assert d2.rope_spec.attention_scaling == 1.0
    assert torch.allclose(d2.get("K_stripped", 1), old, atol=1e-6)


def test_scaled_config_and_load_model_apply_yarn(tmp_path):
    native = _qwen_like(None)
    native.save_pretrained(tmp_path / "m")
    cfg = scaled_config(str(tmp_path / "m"), YARN)
    assert cfg.rope_parameters["rope_type"] == "yarn" and cfg.rope_parameters["rope_theta"] == 1e6
    assert cfg.max_position_embeddings == 81920
    m = load_model(str(tmp_path / "m"), rope_scaling=YARN)
    assert m.model.rotary_emb.attention_scaling == pytest.approx(0.1 * np.log(2.5) + 1.0)
    assert RopeSpec.from_model(m).parameters["factor"] == 2.5
    plain = load_model(str(tmp_path / "m"))
    assert plain.model.rotary_emb.attention_scaling == 1.0
    with pytest.raises(ValueError, match="rope_scaling needs"):
        scaled_config(str(tmp_path / "m"), {"rope_type": "yarn", "factor": 2.5})
    with pytest.raises(ValueError, match="extend the window"):
        scaled_config(str(tmp_path / "m"), {**YARN, "factor": 1.0})
