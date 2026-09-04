import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import repeat_kv


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sdpa_repeat_kv_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                           is_causal=None, **kwargs):
    """SDPA with grouped-query heads expanded by `repeat_kv`, never by `enable_gqa=True`.

    transformers' stock "sdpa" passes `enable_gqa=True` whenever there is no mask. In float32 that
    path has no fused kernel (flash needs fp16/bf16; the memory-efficient kernel does not take
    `enable_gqa`), so PyTorch silently uses the math kernel and materializes the [heads, T, T]
    scores -- 51.5 GiB at T = 29,391 for Qwen3-1.7B, an OOM on any single card. With the KV heads
    repeated up front the memory-efficient kernel is eligible: peak 32.0 GiB at T = 32,123
    (16.7 GiB with `logits_to_keep=1`), measured 2026-09-04 on an H100 MIG 3g.40gb. K/V differ
    from the math kernel only at float32 rounding (max |dK| 9.2e-4 on a scale of 423 at T = 1024).
    Body mirrors `sdpa_attention_forward` (transformers 5.15/5.16) minus the GQA branch and the
    NPU / position-bias cases this package never hits. linear-ceiling ledger entry 0026.
    """
    if getattr(module, "num_key_value_groups", 1) > 1:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    q_length, kv_length = query.shape[2], key.shape[2]
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = q_length > 1 and attention_mask is None and is_causal
    if is_causal and q_length > 1 and kv_length > q_length:           # static-cache prefill case, kept for parity
        key, value = key[:, :, :q_length, :], value[:, :, :q_length, :]
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=is_causal)
    return attn_output.transpose(1, 2).contiguous(), None


ATTN_IMPLEMENTATION = "sdpa_repeat_kv"
AttentionInterface.register(ATTN_IMPLEMENTATION, sdpa_repeat_kv_forward)


def load_model(model_id: str):
    # transformers 5 deprecated `torch_dtype=` in favor of `dtype=`.
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation=ATTN_IMPLEMENTATION)
    return m.to(device()).eval()


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)


def assert_shared_tokenizer(tok_a, tok_b) -> None:
    """Matched-KV pairs must share a tokenizer; compare vocab maps, not names."""
    va, vb = tok_a.get_vocab(), tok_b.get_vocab()
    if va != vb:
        raise ValueError(f"tokenizers differ: {len(va)} vs {len(vb)} entries or different mapping")
