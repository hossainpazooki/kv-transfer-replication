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
