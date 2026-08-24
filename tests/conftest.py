import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

VOCAB = 256


def _tiny(hidden: int, layers: int, heads: int, seed: int) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture
def tiny_src():
    return _tiny(hidden=64, layers=2, heads=4, seed=0)


@pytest.fixture
def tiny_tgt():
    return _tiny(hidden=96, layers=3, heads=6, seed=1)


@pytest.fixture
def tiny_src3():
    """Source with the SAME layer count as tiny_tgt (like the real 28->28 pair) for identity-control tests."""
    return _tiny(hidden=64, layers=3, heads=4, seed=2)


@pytest.fixture
def tiny_tokens():
    def make(n_seqs: int = 6, seq_len: int = 64, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, VOCAB, size=(n_seqs, seq_len), dtype=np.int64)
    return make
