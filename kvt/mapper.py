"""Paper Sec. 3: per-head ridge (Eq. 3-4) over concatenated top-k source layers (Eq. 5), fit in
content space (source RoPE stripped) and re-encoded with target RoPE at apply time."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from kvt.data import KVDump
from kvt.ridge import _np, fit_ridge, predict, r2_score
from kvt.rope import apply_rope, rope_cos_sin, strip_rope


def select_top_k(r2_sel: np.ndarray, k) -> np.ndarray:
    """r2_sel: [L_s, L_t]. Returns [L_t, k] source-layer indices, best first. k='all' keeps every layer."""
    L_s, L_t = r2_sel.shape
    kk = L_s if k == "all" else int(k)
    assert 1 <= kk <= L_s, (k, L_s)
    return np.stack([np.argsort(-r2_sel[:, lt], kind="stable")[:kk] for lt in range(L_t)])


def build_features(dump: KVDump, layers, kind: str, mask: np.ndarray) -> np.ndarray:
    """Concatenate per-layer [n, n_kv, d_h] slices into [n, len(layers)*n_kv*d_h], layer-major
    then head-major then d_h. Must match the column order apply_mapper.feats() builds at inference."""
    cols = [_np(dump.get(kind, int(l)))[mask].reshape(int(mask.sum()), -1) for l in layers]
    return np.concatenate(cols, axis=1).astype(np.float32)


@dataclass
class Mapper:
    k: int
    selected: np.ndarray
    n_kv: int
    d_h: int
    src_theta: float
    tgt_theta: float
    lam: float
    W_K: list = field(default_factory=list)
    b_K: list = field(default_factory=list)
    W_V: list = field(default_factory=list)
    b_V: list = field(default_factory=list)

    @staticmethod
    def formula_params(L_t: int, n_kv: int, k: int, d_h: int) -> int:
        """Appendix D: 2 * L_t * n_kv_t * (k * n_kv_s * d_h_s) * d_h_t, weights only (no bias)."""
        return 2 * L_t * n_kv * (k * n_kv * d_h) * d_h

    def n_weight_params(self) -> int:
        return int(sum(w.size for w in self.W_K) + sum(w.size for w in self.W_V))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for l in range(len(self.W_K)):
            tensors[f"K.W.{l}"], tensors[f"K.b.{l}"] = self.W_K[l], self.b_K[l]
            tensors[f"V.W.{l}"], tensors[f"V.b.{l}"] = self.W_V[l], self.b_V[l]
        save_file(tensors, str(path.with_suffix(".safetensors")))
        path.with_suffix(".json").write_text(json.dumps({
            "k": self.k, "selected": self.selected.tolist(), "n_kv": self.n_kv, "d_h": self.d_h,
            "src_theta": self.src_theta, "tgt_theta": self.tgt_theta, "lam": self.lam,
            "n_weight_params": self.n_weight_params()}, indent=2))

    @classmethod
    def load(cls, path) -> "Mapper":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        t = load_file(str(path.with_suffix(".safetensors")))
        L_t = len(meta["selected"])
        return cls(k=meta["k"], selected=np.asarray(meta["selected"]), n_kv=meta["n_kv"], d_h=meta["d_h"],
                    src_theta=meta["src_theta"], tgt_theta=meta["tgt_theta"], lam=meta["lam"],
                    W_K=[t[f"K.W.{l}"] for l in range(L_t)], b_K=[t[f"K.b.{l}"] for l in range(L_t)],
                    W_V=[t[f"V.W.{l}"] for l in range(L_t)], b_V=[t[f"V.b.{l}"] for l in range(L_t)])


def fit_mapper(src: KVDump, tgt: KVDump, selected: np.ndarray, lam: float, train_mask: np.ndarray) -> Mapper:
    m = Mapper(k=int(selected.shape[1]), selected=selected, n_kv=tgt.n_kv, d_h=tgt.d_h,
               src_theta=src.rope_theta, tgt_theta=tgt.rope_theta, lam=lam)
    n = int(train_mask.sum())
    for lt in range(tgt.n_layers):
        X_K = build_features(src, selected[lt], "K_stripped", train_mask)
        Y_K = _np(tgt.get("K_stripped", lt))[train_mask].reshape(n, -1)
        W, b = fit_ridge(X_K, Y_K, lam)
        m.W_K.append(W); m.b_K.append(b)
        X_V = build_features(src, selected[lt], "V", train_mask)
        Y_V = _np(tgt.get("V", lt))[train_mask].reshape(n, -1)
        W, b = fit_ridge(X_V, Y_V, lam)
        m.W_V.append(W); m.b_V.append(b)
    return m


def mapper_r2(m: Mapper, src: KVDump, tgt: KVDump, mask: np.ndarray) -> dict:
    """Head-averaged R^2 per target layer; per-head = columns h*d_h:(h+1)*d_h (ridge is separable)."""
    out = {"K": np.zeros(len(m.W_K)), "V": np.zeros(len(m.W_V))}
    n = int(mask.sum())
    for lt in range(len(m.W_K)):
        for kind, W, b, key in (("K_stripped", m.W_K[lt], m.b_K[lt], "K"), ("V", m.W_V[lt], m.b_V[lt], "V")):
            X = build_features(src, m.selected[lt], kind, mask)
            Y = _np(tgt.get(kind, lt))[mask].reshape(n, -1)
            Yhat = predict(X, W, b)
            out[key][lt] = float(np.mean([r2_score(Y[:, h * m.d_h:(h + 1) * m.d_h], Yhat[:, h * m.d_h:(h + 1) * m.d_h])
                                          for h in range(m.n_kv)]))
    return out


@torch.no_grad()
def apply_mapper(m: Mapper, src_kvs, positions: torch.Tensor):
    """src_kvs: list over source layers of (K_rope[B,n_kv,T,d_h], V). Returns list over target layers."""
    B, n_kv, T, d_h = src_kvs[0][0].shape
    dev = src_kvs[0][0].device
    cos_s, sin_s = rope_cos_sin(positions, d_h, m.src_theta)
    cos_t, sin_t = rope_cos_sin(positions, d_h, m.tgt_theta)
    K_strip = [strip_rope(k.float(), cos_s.to(dev), sin_s.to(dev)) for k, _ in src_kvs]     # [B,n_kv,T,d_h]
    V_src = [v.float() for _, v in src_kvs]

    def feats(tensors, layers):
        return torch.cat([tensors[int(l)].permute(0, 2, 1, 3).reshape(B, T, n_kv * d_h) for l in layers], dim=-1)

    out = []
    for lt in range(len(m.W_K)):
        W_K, b_K = torch.from_numpy(m.W_K[lt]).to(dev), torch.from_numpy(m.b_K[lt]).to(dev)
        W_V, b_V = torch.from_numpy(m.W_V[lt]).to(dev), torch.from_numpy(m.b_V[lt]).to(dev)
        K_hat = (feats(K_strip, m.selected[lt]) @ W_K + b_K).reshape(B, T, n_kv, d_h).permute(0, 2, 1, 3)
        K_hat = apply_rope(K_hat, cos_t.to(dev), sin_t.to(dev))
        V_hat = (feats(V_src, m.selected[lt]) @ W_V + b_V).reshape(B, T, n_kv, d_h).permute(0, 2, 1, 3)
        out.append((K_hat.contiguous(), V_hat.contiguous()))
    return out
