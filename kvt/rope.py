"""RoPE in the HF rotate-half layout. R(theta) is orthogonal, so strip = R^T = rotation by -theta."""
import torch


def rope_cos_sin(positions: torch.Tensor, d_h: int, theta: float, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_h, 2, dtype=torch.float32) / d_h))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]        # [T, d_h/2]
    emb = torch.cat([freqs, freqs], dim=-1)                                  # [T, d_h]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., T, d_h]; cos/sin: [T, d_h] (broadcast over leading dims)."""
    return x * cos + _rotate_half(x) * sin


def strip_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Exact inverse of apply_rope: rotate by -theta (cos even, sin odd)."""
    return x * cos - _rotate_half(x) * sin


def apply_rope_tokens_first(k: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """k: [T, n_kv, d_h] (dump layout)."""
    cos, sin = rope_cos_sin(positions, k.shape[-1], theta, dtype=k.dtype)
    return apply_rope(k, cos[:, None, :], sin[:, None, :])


def strip_rope_tokens_first(k: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    cos, sin = rope_cos_sin(positions, k.shape[-1], theta, dtype=k.dtype)
    return strip_rope(k, cos[:, None, :], sin[:, None, :])


# ---------------------------------------------------------------------------------------------
# RoPE spec (linear-ceiling E9-long, 2026-09-09): a scaled RoPE (YaRN) changes BOTH the per-dimension
# inverse frequencies AND multiplies cos/sin by an attention factor m (transformers'
# `Qwen3RotaryEmbedding.forward`: `cos = emb.cos() * self.attention_scaling`). The K a model writes
# into its cache is then m * R_yarn(pos) * k_content, so stripping with the plain-theta rotation above
# would leave a wrong rotation AND a factor m in every content-space K. The spec below is read from the
# model's own rotary embedding (never re-derived from a formula), written into every dump's meta.json,
# halt-checked against the model at dump time, and used by KVDump to strip. For a model without RoPE
# scaling the spec reduces exactly to `rope_cos_sin(positions, d_h, theta)` with m = 1.

from dataclasses import dataclass


@dataclass(frozen=True)
class RopeSpec:
    inv_freq: tuple            # d_h/2 floats, as the model's rotary embedding holds them
    attention_scaling: float   # m; 1.0 for the default RoPE
    parameters: dict           # the model config's rope_parameters (rope_type, rope_theta, factor, ...)

    def to_json(self) -> dict:
        return {"inv_freq": [float(x) for x in self.inv_freq], "attention_scaling": float(self.attention_scaling),
                "parameters": dict(self.parameters)}

    @classmethod
    def from_json(cls, d: dict) -> "RopeSpec":
        return cls(tuple(float(x) for x in d["inv_freq"]), float(d["attention_scaling"]), dict(d.get("parameters", {})))

    @classmethod
    def default(cls, d_h: int, theta: float) -> "RopeSpec":
        inv = 1.0 / (theta ** (torch.arange(0, d_h, 2, dtype=torch.float32) / d_h))
        return cls(tuple(float(x) for x in inv.tolist()), 1.0, {"rope_type": "default", "rope_theta": float(theta)})

    @classmethod
    def from_model(cls, model) -> "RopeSpec":
        """The spec the model actually applies: its rotary embedding's `inv_freq` buffer and
        `attention_scaling`, plus the config's rope_parameters for the record."""
        emb = getattr(getattr(model, "model", model), "rotary_emb", None)
        if emb is None or not hasattr(emb, "inv_freq") or not hasattr(emb, "attention_scaling"):
            raise ValueError("model exposes no rotary_emb with inv_freq/attention_scaling; cannot record its RoPE spec")
        params = getattr(model.config, "rope_parameters", None) or {"rope_theta": float(getattr(model.config, "rope_theta"))}
        return cls(tuple(float(x) for x in emb.inv_freq.detach().float().cpu().tolist()),
                   float(emb.attention_scaling), {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                                                  for k, v in dict(params).items()})

    def cos_sin(self, positions: torch.Tensor, dtype=torch.float32):
        """UNSCALED cos/sin [T, d_h] under this spec's frequencies, in the same float32 arithmetic as
        transformers' rotary forward (positions.float() times inv_freq.float(), then cos/sin)."""
        inv = torch.tensor(self.inv_freq, dtype=torch.float32)
        freqs = positions.to(torch.float32)[:, None] * inv[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope_spec_tokens_first(k: torch.Tensor, positions: torch.Tensor, spec: RopeSpec) -> torch.Tensor:
    """k: [T, n_kv, d_h] content-space -> what the model writes: m * R(pos) k."""
    cos, sin = spec.cos_sin(positions, dtype=k.dtype)
    return apply_rope(k, cos[:, None, :], sin[:, None, :]) * spec.attention_scaling


def strip_rope_spec_tokens_first(k: torch.Tensor, positions: torch.Tensor, spec: RopeSpec) -> torch.Tensor:
    """Exact inverse of apply_rope_spec_tokens_first: R(pos)^T k / m."""
    cos, sin = spec.cos_sin(positions, dtype=k.dtype)
    return strip_rope(k, cos[:, None, :], sin[:, None, :]) / spec.attention_scaling


@torch.no_grad()
def check_rope_spec_against_model(model, spec: RopeSpec, positions: torch.Tensor, atol: float) -> float:
    """HALT check: the scaled cos/sin this spec reconstructs must equal what the model's own rotary
    embedding returns at `positions`. Returns the max abs difference; raises above `atol`."""
    emb = getattr(getattr(model, "model", model), "rotary_emb")
    dev = next(model.parameters()).device
    pos = positions.to(dev).long()
    probe = torch.zeros(1, 1, dtype=torch.float32, device=dev)
    cos_hf, sin_hf = emb(probe, pos[None])                       # [1, T, d_h], already times attention_scaling
    cos_u, sin_u = spec.cos_sin(positions)
    m = spec.attention_scaling
    worst = max(float((cos_hf[0].float().cpu() - cos_u * m).abs().max()),
                float((sin_hf[0].float().cpu() - sin_u * m).abs().max()))
    if worst > atol:
        raise ValueError(f"RoPE spec does not reproduce the model's rotary embedding (max |diff| {worst:.3e} > {atol:.1e})")
    return worst
