"""Closed-form ridge (paper Eq. 4): W* = (X^T X + lam I)^-1 X^T Y on centered X, Y; b = ybar - xbar W."""
import numpy as np
import torch

from kvt.data import KVDump


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def fit_ridge(X, Y, lam: float, chunk: int = 16384):
    X, Y = _np(X), _np(Y)
    n, p = X.shape
    q = Y.shape[1]
    xm = X.mean(0, dtype=np.float64); ym = Y.mean(0, dtype=np.float64)
    G = np.zeros((p, p), dtype=np.float64); XtY = np.zeros((p, q), dtype=np.float64)
    for s in range(0, n, chunk):
        Xc = X[s : s + chunk].astype(np.float64) - xm
        Yc = Y[s : s + chunk].astype(np.float64) - ym
        G += Xc.T @ Xc
        XtY += Xc.T @ Yc
    G[np.diag_indices(p)] += lam
    try:
        W = np.linalg.solve(G, XtY)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(G, XtY, rcond=None)[0]
    b = ym - xm @ W
    return W.astype(np.float32), b.astype(np.float32)


def predict(X, W, b) -> np.ndarray:
    return _np(X).astype(np.float32) @ W + b


def r2_score(Y, Yhat) -> float:
    """Definition A5: pooled over rows and columns; ybar is the mean of the set being scored."""
    Y, Yhat = _np(Y).astype(np.float64), _np(Yhat).astype(np.float64)
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def probe_r2(src: KVDump, tgt: KVDump, kind: str, holdout_frac: float, lam: float = 0.0) -> dict:
    """Paper Sec. 2.2 probe: per (l_s, l_t, h) single-source OLS, head-matched, head-averaged R^2."""
    assert src.n_kv == tgt.n_kv and src.d_h == tgt.d_h, "probe requires matched-KV dumps"
    tr, ho = src.split(holdout_frac)
    assert (tr == tgt.split(holdout_frac)[0]).all(), "dumps were made on different token sets"
    out = {"train": np.zeros((src.n_layers, tgt.n_layers)), "heldout": np.zeros((src.n_layers, tgt.n_layers))}
    for ls in range(src.n_layers):
        Xall = _np(src.get(kind, ls))                                    # [n, n_kv, d_h]
        for lt in range(tgt.n_layers):
            Yall = _np(tgt.get(kind, lt))
            r_tr, r_ho = [], []
            for h in range(src.n_kv):
                W, b = fit_ridge(Xall[tr, h], Yall[tr, h], lam)
                r_tr.append(r2_score(Yall[tr, h], predict(Xall[tr, h], W, b)))
                r_ho.append(r2_score(Yall[ho, h], predict(Xall[ho, h], W, b)))
            out["train"][ls, lt] = float(np.mean(r_tr))
            out["heldout"][ls, lt] = float(np.mean(r_ho))
    return out
