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
    space: str = "content"
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
            "space": self.space,
            "n_weight_params": self.n_weight_params()}, indent=2))

    @classmethod
    def load(cls, path) -> "Mapper":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        t = load_file(str(path.with_suffix(".safetensors")))
        L_t = len(meta["selected"])
        return cls(k=meta["k"], selected=np.asarray(meta["selected"]), n_kv=meta["n_kv"], d_h=meta["d_h"],
                    src_theta=meta["src_theta"], tgt_theta=meta["tgt_theta"], lam=meta["lam"],
                    space=meta.get("space", "content"),
                    W_K=[t[f"K.W.{l}"] for l in range(L_t)], b_K=[t[f"K.b.{l}"] for l in range(L_t)],
                    W_V=[t[f"V.W.{l}"] for l in range(L_t)], b_V=[t[f"V.b.{l}"] for l in range(L_t)])


def _k_kind(space: str) -> str:
    """Which stored cache kind a mapper of this space consumes and predicts."""
    if space == "content":
        return "K_stripped"
    if space == "rope":
        return "K_rope"
    raise ValueError(f"unknown mapper space {space!r}; expected 'content' or 'rope'")


# One target layer's working set at k=8 is 16 source tensors + 2 target tensors. 24 holds that
# with headroom while bounding peak cache at 24 * 440 MB = 10.6 GB at 420 sequences, leaving
# room for the 3.4 GB design matrix and the 537 MB float64 Gram. Without a bound the cache
# reaches 49 GB at 420 sequences and the fit dies on a 32 GB box.
CACHE_LIMIT = 24


def fit_mapper(src: KVDump, tgt: KVDump, selected: np.ndarray, lam: float,
               train_mask: np.ndarray, space: str = "content") -> Mapper:
    kind = _k_kind(space)
    src.set_cache_limit(CACHE_LIMIT)
    tgt.set_cache_limit(CACHE_LIMIT)
    m = Mapper(k=int(selected.shape[1]), selected=selected, n_kv=tgt.n_kv, d_h=tgt.d_h,
               src_theta=src.rope_theta, tgt_theta=tgt.rope_theta, lam=lam, space=space)
    n = int(train_mask.sum())
    for lt in range(tgt.n_layers):
        X_K = build_features(src, selected[lt], kind, train_mask)
        Y_K = _np(tgt.get(kind, lt))[train_mask].reshape(n, -1)
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
    src.set_cache_limit(CACHE_LIMIT)
    tgt.set_cache_limit(CACHE_LIMIT)
    n = int(mask.sum())
    k_kind = _k_kind(getattr(m, "space", "content"))
    for lt in range(len(m.W_K)):
        for kind, W, b, key in ((k_kind, m.W_K[lt], m.b_K[lt], "K"),
                                ("V", m.W_V[lt], m.b_V[lt], "V")):
            X = build_features(src, m.selected[lt], kind, mask)
            Y = _np(tgt.get(kind, lt))[mask].reshape(n, -1)
            Yhat = predict(X, W, b)
            out[key][lt] = float(np.mean([r2_score(Y[:, h * m.d_h:(h + 1) * m.d_h], Yhat[:, h * m.d_h:(h + 1) * m.d_h])
                                          for h in range(m.n_kv)]))
    return out


@torch.no_grad()
def apply_mapper(m: Mapper, src_kvs, positions: torch.Tensor):
    """src_kvs: list over source layers of (K_rope[B,n_kv,T,d_h], V). Returns list over target layers.

    space='content': strip source RoPE, map, re-apply target RoPE (the paper's design).
    space='rope':    map the rotated keys directly. The two differ ONLY here, which is what
                     makes the WP3 comparison a controlled one.
    """
    space = getattr(m, "space", "content")
    _k_kind(space)                                    # validate early
    B, n_kv, T, d_h = src_kvs[0][0].shape
    dev = src_kvs[0][0].device
    cos_s, sin_s = rope_cos_sin(positions, d_h, m.src_theta)
    cos_t, sin_t = rope_cos_sin(positions, d_h, m.tgt_theta)
    if space == "content":
        K_in = [strip_rope(k.float(), cos_s.to(dev), sin_s.to(dev)) for k, _ in src_kvs]
    else:
        K_in = [k.float() for k, _ in src_kvs]
    V_src = [v.float() for _, v in src_kvs]

    def feats(tensors, layers):
        return torch.cat([tensors[int(l)].permute(0, 2, 1, 3).reshape(B, T, n_kv * d_h) for l in layers], dim=-1)

    out = []
    for lt in range(len(m.W_K)):
        W_K, b_K = torch.from_numpy(m.W_K[lt]).to(dev), torch.from_numpy(m.b_K[lt]).to(dev)
        W_V, b_V = torch.from_numpy(m.W_V[lt]).to(dev), torch.from_numpy(m.b_V[lt]).to(dev)
        K_hat = (feats(K_in, m.selected[lt]) @ W_K + b_K).reshape(B, T, n_kv, d_h).permute(0, 2, 1, 3)
        if space == "content":
            K_hat = apply_rope(K_hat, cos_t.to(dev), sin_t.to(dev))
        V_hat = (feats(V_src, m.selected[lt]) @ W_V + b_V).reshape(B, T, n_kv, d_h).permute(0, 2, 1, 3)
        out.append((K_hat.contiguous(), V_hat.contiguous()))
    return out


def compose(a: Mapper, b: Mapper) -> Mapper:
    """Closed-form composition: returns C with C(x) == b(a(x)) up to float error.

    `a` maps source S -> middle M, `b` maps M -> target T, and C maps S -> T directly.
    See docs/superpowers/plans/2026-08-24-cache-economics-followon.md Task 2 for the
    derivation. The point of having BOTH this and the two-call path is that they are
    independent computations of the same quantity, so disagreement is a bug signal
    (gate H-C3) rather than something to be discovered downstream as a bad R^2.
    """
    if a.n_kv != b.n_kv or a.d_h != b.d_h:
        raise ValueError(
            f"shape mismatch: a has n_kv={a.n_kv} d_h={a.d_h}, b has n_kv={b.n_kv} d_h={b.d_h}")
    if getattr(a, "space", "content") != getattr(b, "space", "content"):
        raise ValueError(
            f"cannot compose mappers fitted in different spaces: "
            f"{getattr(a, 'space', 'content')!r} then {getattr(b, 'space', 'content')!r}")
    if abs(float(a.tgt_theta) - float(b.src_theta)) > 1e-9:
        raise ValueError(
            f"a's target theta ({a.tgt_theta}) must equal b's source theta ({b.src_theta}); "
            "otherwise the RoPE re-apply after a and the strip before b do not cancel")

    D = a.n_kv * a.d_h
    L_M = len(a.W_K)
    L_T, k_b = b.selected.shape
    sel_rows, W_K, b_K, W_V, b_V = [], [], [], [], []

    for lt in range(L_T):
        rows, wk, wv = [], [], []
        bk = np.asarray(b.b_K[lt], dtype=np.float64)
        bv = np.asarray(b.b_V[lt], dtype=np.float64)
        for j in range(k_b):
            s = int(b.selected[lt, j])
            if not (0 <= s < L_M):
                raise ValueError(
                    f"b selects middle layer {s} for target layer {lt}, but a only produces "
                    f"{L_M} layers")
            BK_j = np.asarray(b.W_K[lt][j * D:(j + 1) * D, :], dtype=np.float64)
            BV_j = np.asarray(b.W_V[lt][j * D:(j + 1) * D, :], dtype=np.float64)
            rows.extend(int(x) for x in a.selected[s])
            wk.append(np.asarray(a.W_K[s], dtype=np.float64) @ BK_j)
            wv.append(np.asarray(a.W_V[s], dtype=np.float64) @ BV_j)
            bk = bk + np.asarray(a.b_K[s], dtype=np.float64) @ BK_j
            bv = bv + np.asarray(a.b_V[s], dtype=np.float64) @ BV_j
        sel_rows.append(rows)
        W_K.append(np.vstack(wk).astype(np.float32)); b_K.append(bk.astype(np.float32))
        W_V.append(np.vstack(wv).astype(np.float32)); b_V.append(bv.astype(np.float32))

    return Mapper(k=int(a.selected.shape[1]) * int(b.selected.shape[1]),
                  selected=np.asarray(sel_rows, dtype=np.int64),
                  n_kv=a.n_kv, d_h=a.d_h, src_theta=a.src_theta, tgt_theta=b.tgt_theta,
                  lam=max(float(a.lam), float(b.lam)),
                  space=getattr(a, "space", "content"),
                  W_K=W_K, b_K=b_K, W_V=W_V, b_V=b_V)
