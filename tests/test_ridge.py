import numpy as np
import torch

from kvt.data import dump_kv, KVDump
from kvt.ridge import fit_ridge, predict, probe_r2, r2_score


def test_ridge_recovers_exact_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 8)).astype(np.float32)
    W_true = rng.normal(size=(8, 3)); b_true = rng.normal(size=3)
    Y = (X @ W_true + b_true).astype(np.float32)
    W, b = fit_ridge(X, Y, lam=0.0)
    assert np.allclose(W, W_true, atol=1e-4) and np.allclose(b, b_true, atol=1e-4)
    assert r2_score(Y, predict(X, W, b)) > 0.99999


def test_ridge_lambda_shrinks():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4)).astype(np.float32); Y = X.astype(np.float32).copy()
    W0, _ = fit_ridge(X, Y, lam=0.0); W1, _ = fit_ridge(X, Y, lam=1000.0)
    assert np.linalg.norm(W1) < np.linalg.norm(W0)


def test_noise_train_r2_equals_overfit_floor_and_heldout_is_nonpositive():
    """H2: for independent noise, in-sample R^2 ~ p/n; held-out R^2 <= ~0."""
    rng = np.random.default_rng(2)
    n, p = 2000, 100
    X = rng.normal(size=(n, p)).astype(np.float32); Y = rng.normal(size=(n, 5)).astype(np.float32)
    Xh = rng.normal(size=(n, p)).astype(np.float32); Yh = rng.normal(size=(n, 5)).astype(np.float32)
    W, b = fit_ridge(X, Y, lam=0.0)
    train = r2_score(Y, predict(X, W, b)); held = r2_score(Yh, predict(Xh, W, b))
    assert abs(train - p / n) < 0.02, train
    assert held < 0.01, held


@torch.no_grad()
def test_probe_self_is_identity_on_diagonal(tmp_path, tiny_tgt, tiny_tokens):
    seqs = tiny_tokens(n_seqs=10, seq_len=64)
    dump_kv(tiny_tgt, seqs, stride=2, out_dir=tmp_path)
    d = KVDump.load(tmp_path)
    r = probe_r2(d, d, "K_stripped", holdout_frac=0.2)
    assert r["train"].shape == (3, 3)
    assert np.allclose(np.diag(r["train"]), 1.0, atol=1e-4)
    assert np.allclose(np.diag(r["heldout"]), 1.0, atol=1e-3)
    assert r["heldout"].shape == (3, 3)
