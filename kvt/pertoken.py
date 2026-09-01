"""Per-token squared deviations that sum to the per-head SSE the scorers record.

Added for linear-ceiling's entry 0023 (E9 per-token amendment). The scorers keep writing
per-layer, per-head SSE/SST and R^2 (definition A5 per head, head-averaged, layer-averaged);
this module makes the SSE the float64 sum over tokens of a per-token, per-head square that is
ALSO written out, so a reader can recover any per-token distribution and check that the
sums reproduce the recorded moments. Nothing here changes a recorded R^2.
"""
import numpy as np


def per_token_squares(Y, Yhat, n_kv: int, d_h: int):
    """Y, Yhat: [n, n_kv*d_h] (head-major columns). Returns float64 arrays
    sq[n, n_kv] = ||y_h - yhat_h||^2 and ref[n, n_kv] = ||y_h||^2 (the reference's own norm)."""
    Y64 = np.asarray(Y, dtype=np.float64).reshape(len(Y), n_kv, d_h)
    H64 = np.asarray(Yhat, dtype=np.float64).reshape(len(Yhat), n_kv, d_h)
    return ((Y64 - H64) ** 2).sum(2), (Y64 ** 2).sum(2)


def moments(Y, Yhat, n_kv: int, d_h: int):
    """Per-head SSE (as the float64 sum of per-token squares), SST, and head-mean R^2 (A5 per
    head), plus the per-token arrays. SSE[h] == sq[:, h].sum() by construction."""
    sq, ref = per_token_squares(Y, Yhat, n_kv, d_h)
    Y64 = np.asarray(Y, dtype=np.float64).reshape(len(Y), n_kv, d_h)
    sse = sq.sum(0)
    sst = ((Y64 - Y64.mean(0, keepdims=True)) ** 2).sum((0, 2))
    r2 = 1.0 - sse / sst
    rec = {"sse": [float(x) for x in sse], "sst": [float(x) for x in sst], "r2_head_mean": float(np.mean(r2))}
    return rec, sq, ref


def pooled_r2(Y, Yhat) -> float:
    """A5 pooled over ALL columns of the layer at once (heads not averaged) -- the diagnostic
    linear-ceiling's 0023 records beside the head-averaged figure; it is NOT what mapper_r2 or
    score_positions report."""
    Y64, H64 = np.asarray(Y, dtype=np.float64), np.asarray(Yhat, dtype=np.float64)
    return float(1.0 - ((Y64 - H64) ** 2).sum() / ((Y64 - Y64.mean(0)) ** 2).sum())
