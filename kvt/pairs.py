from dataclasses import dataclass


@dataclass(frozen=True)
class KVShape:
    n_layers: int
    n_kv: int
    d_h: int
    rope_theta: float


@dataclass(frozen=True)
class Pair:
    name: str
    source: str
    target: str


PAIRS: dict[str, Pair] = {
    "qwen3-0.6b-to-1.7b": Pair("qwen3-0.6b-to-1.7b", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"),
    "qwen3-1.7b-to-4b": Pair("qwen3-1.7b-to-4b", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"),  # [STRETCH]
    "qwen3-0.6b-to-4b": Pair("qwen3-0.6b-to-4b", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"),  # [STRETCH] WP1 direct
    # linear-ceiling entry 0039 (provisional; its drafts/README.md allocates the number at staging,
    # and the ledger ends at 0038): second model family, cross-release.
    # Matched-KV under paper Sec. 2.1: both sides num_key_value_heads=8, head_dim=128 (the 8B declares
    # no head_dim; kv_shape reaches 128 via hidden_size // num_attention_heads), so check_matched_kv
    # passes without relaxation. The two sides carry DIFFERENT llama3 RoPE scaling (3.2: factor 32.0;
    # 3.1: factor 8.0), which is why this pair requires 063f4023 or later: KVDump strips with each
    # dump's own recorded RopeSpec, not with plain rope_theta.
    # The key names the receiver IN FULL: the short form would read "llama3.2-3b-to-8b", asserting a
    # Llama-3.2-8B that does not exist, and this string keys mappers/, data/kv/, data/tokens/ and
    # every report.
    "llama3.2-3b-to-llama3.1-8b": Pair("llama3.2-3b-to-llama3.1-8b",
                                       "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"),
}


def _rope_theta(config) -> float:
    """Read rope_theta from a Qwen3Config.

    transformers 5 moved RoPE settings into a dict (`config.rope_parameters`);
    the plain `config.rope_theta` attribute no longer exists there. We must
    raise rather than default a missing value -- a silently wrong theta
    corrupts every RoPE operation downstream and a consistently-wrong value
    would not be caught by later tests (the error cancels when both sides
    share it).
    """
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    if hasattr(config, "rope_theta"):
        return float(config.rope_theta)
    raise ValueError("cannot determine rope_theta from config")


def kv_shape(config) -> KVShape:
    d_h = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return KVShape(int(config.num_hidden_layers), int(config.num_key_value_heads), int(d_h),
                   _rope_theta(config))


def check_matched_kv(src_config, tgt_config) -> None:
    """Matched-KV (paper Sec. 2.1): same KV head count and per-head dim. Layers/hidden may differ."""
    a, b = kv_shape(src_config), kv_shape(tgt_config)
    problems = []
    if a.n_kv != b.n_kv:
        problems.append(f"kv heads {a.n_kv} != {b.n_kv}")
    if a.d_h != b.d_h:
        problems.append(f"head dim {a.d_h} != {b.d_h}")
    if problems:
        raise ValueError("not matched-KV: " + "; ".join(problems))
