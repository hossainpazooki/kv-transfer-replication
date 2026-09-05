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


def per_sequence_moments(sq, Y, seq_idx, n_kv: int, d_h: int):
    """Per-sequence decomposition of the per-head moments (linear-ceiling E8 amendment).

    `sq` [n, n_kv] are the per-token squares of the masked rows, `Y` [n, n_kv*d_h] their reference
    values and `seq_idx` [n] the sequence each row came from. Returns (seq_ids [S], sse_seq [S, n_kv],
    sst_seq [S, n_kv]) in float64, where SST is taken around the GLOBAL mean of the masked rows -- so
    sse_seq.sum(0) == sq.sum(0) and sst_seq.sum(0) == the pooled SST exactly, and the pooled per-head
    R^2 is 1 - sse_seq.sum(0) / sst_seq.sum(0). A per-sequence R^2 = 1 - sse_s / sst_s is a share of
    the same decomposition, not a re-centered fit."""
    sq = np.asarray(sq, dtype=np.float64)
    Y64 = np.asarray(Y, dtype=np.float64).reshape(len(Y), n_kv, d_h)
    seq_idx = np.asarray(seq_idx)
    if len(sq) != len(Y64) or len(seq_idx) != len(Y64):
        raise ValueError(f"per_sequence_moments: {len(sq)} squares, {len(Y64)} rows, {len(seq_idx)} indices")
    cen = ((Y64 - Y64.mean(0, keepdims=True)) ** 2).sum(2)          # [n, n_kv], around the global mean
    seq_ids, inv = np.unique(seq_idx, return_inverse=True)
    sse_seq = np.zeros((len(seq_ids), n_kv)); sst_seq = np.zeros((len(seq_ids), n_kv))
    np.add.at(sse_seq, inv, sq)
    np.add.at(sst_seq, inv, cen)
    return seq_ids.astype(np.int64), sse_seq, sst_seq
