import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_id: str):
    # transformers 5 deprecated `torch_dtype=` in favor of `dtype=`.
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    return m.to(device()).eval()


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)


def assert_shared_tokenizer(tok_a, tok_b) -> None:
    """Matched-KV pairs must share a tokenizer; compare vocab maps, not names."""
    va, vb = tok_a.get_vocab(), tok_b.get_vocab()
    if va != vb:
        raise ValueError(f"tokenizers differ: {len(va)} vs {len(vb)} entries or different mapping")
