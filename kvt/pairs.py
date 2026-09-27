from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KVShape:
    n_layers: int
    n_kv: int
    d_h: int
    rope_theta: float


@dataclass(frozen=True)
class ModelRef:
    """Exactly which checkpoint to load (bridge spec G3).

    `model_id` is the HF repo id, or -- with `local_path` -- the identifier recorded for provenance
    (e.g. the base model a local training checkpoint descends from). `revision` pins an HF revision
    (branch, tag or commit; the resolved commit is what the dump records). `local_path` loads from
    that directory instead of the Hub. A revision and a local path together are refused: a local
    directory has no HF revision, and silently ignoring either would record the wrong provenance.
    """
    model_id: str
    revision: str | None = None
    local_path: Path | None = None

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError(f"model_id must be a non-empty string, got {self.model_id!r}")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError(f"revision must be a non-empty string or None, got {self.revision!r}")
        if self.local_path is not None and not isinstance(self.local_path, Path):
            raise TypeError(f"local_path must be a pathlib.Path or None, got {type(self.local_path).__name__}")
        if self.revision is not None and self.local_path is not None:
            raise ValueError(f"ambiguous checkpoint for {self.model_id}: both revision={self.revision!r} and "
                             f"local_path={str(self.local_path)!r} given; pass exactly one")


@dataclass(frozen=True)
class Pair:
    name: str
    source: str
    target: str
    # G3: optional per-side pins. None (the default) keeps the legacy behavior: load `source`/`target`
    # as HF ids at whatever revision the Hub/cache serves.
    source_revision: str | None = None
    source_local_path: Path | None = None
    target_revision: str | None = None
    target_local_path: Path | None = None

    def __post_init__(self):
        self.model_ref("source"), self.model_ref("target")   # validate both sides at construction

    def model_ref(self, which: str) -> ModelRef:
        if which not in ("source", "target"):
            raise ValueError(f"which must be 'source' or 'target', got {which!r}")
        return ModelRef(getattr(self, which), getattr(self, f"{which}_revision"),
                        getattr(self, f"{which}_local_path"))


PAIRS: dict[str, Pair] = {
    "qwen3-0.6b-to-1.7b": Pair("qwen3-0.6b-to-1.7b", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"),
    "qwen3-1.7b-to-4b": Pair("qwen3-1.7b-to-4b", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"),  # [STRETCH]
    "qwen3-0.6b-to-4b": Pair("qwen3-0.6b-to-4b", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"),  # [STRETCH] WP1 direct
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
