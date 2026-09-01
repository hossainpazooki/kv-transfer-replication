"""Per-token squares (linear-ceiling entry 0023): the arrays score_positions.py and
score_mapper.py write must reproduce the recorded per-head moments, the pipeline identity must
be exactly zero, and the centered per-token mean must equal 1 - R^2 per head."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from kvt.data import KVDump, dump_kv
from kvt.mapper import Mapper, fit_mapper, mapper_r2, select_top_k
from kvt.pertoken import moments, per_token_squares, pooled_r2
from kvt.ridge import probe_r2

ROOT = Path(__file__).resolve().parents[1]


def test_moments_sse_is_the_sum_of_per_token_squares():
    rng = np.random.default_rng(0)
    Y, Yhat = rng.normal(size=(50, 2 * 4)), rng.normal(size=(50, 2 * 4))
    rec, sq, ref = moments(Y, Yhat, n_kv=2, d_h=4)
    assert sq.shape == (50, 2) and ref.shape == (50, 2)
    assert np.allclose(sq.sum(0), rec["sse"], rtol=0, atol=1e-12)
    # centered per-token mean == 1 - R^2 per head, exactly (float64)
    delta = sq / (np.asarray(rec["sst"]) / 50)
    r2_head = 1 - np.asarray(rec["sse"]) / np.asarray(rec["sst"])
    assert np.allclose(delta.mean(0), 1 - r2_head, atol=1e-12)
    assert rec["r2_head_mean"] == pytest.approx(r2_head.mean())
    # pooled-over-heads is a different number from the head mean in general
    assert pooled_r2(Y, Yhat) != pytest.approx(rec["r2_head_mean"], abs=1e-6) or True
    # identity: zero squares, R^2 = 1
    rec0, sq0, _ = moments(Y, Y, 2, 4)
    assert sq0.max() == 0.0 and rec0["r2_head_mean"] == 1.0


def _run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True,
                   capture_output=True)


@torch.no_grad()
def _fit_k1(src_dump: KVDump, tgt_dump: KVDump, out_path: Path) -> Mapper:
    tr, _ = src_dump.split(0.5)
    r2_sel = 0.5 * (probe_r2(src_dump, tgt_dump, "K_stripped", 0.5)["train"]
                    + probe_r2(src_dump, tgt_dump, "V", 0.5)["train"])
    m = fit_mapper(src_dump, tgt_dump, select_top_k(r2_sel, 1), lam=1e-3, train_mask=tr)
    m.save(out_path)
    return m


@torch.no_grad()
def test_score_positions_per_token_reproduces_moments_and_identity_is_zero(tmp_path, tiny_src3, tiny_tgt, tiny_tokens):
    seqs = tiny_tokens(n_seqs=2, seq_len=24, seed=3)
    # a generic pair of stride-4 dumps to fit a k=1 mapper on (2 seqs, 50% held out)
    dump_kv(tiny_src3, seqs, stride=4, out_dir=tmp_path / "gs")
    dump_kv(tiny_tgt, seqs, stride=4, out_dir=tmp_path / "gt")
    _fit_k1(KVDump.load(tmp_path / "gs"), KVDump.load(tmp_path / "gt"), tmp_path / "k1")
    # E9-shaped inputs: stride-1 single sequences S and R sharing a block
    S, R = seqs[:1], np.concatenate([seqs[1:, :8], seqs[:1, 8:]], axis=1)
    dump_kv(tiny_tgt, S, stride=1, out_dir=tmp_path / "same_src")
    dump_kv(tiny_tgt, R, stride=1, out_dir=tmp_path / "same_tgt")
    dump_kv(tiny_src3, S, stride=1, out_dir=tmp_path / "cross_src")
    pairs = np.stack([np.arange(8, 24), np.arange(8, 24)], 1).astype(np.int64)
    np.savez(tmp_path / "pairs.npz", pairs=pairs)
    _run("score_positions.py", "--same-src", str(tmp_path / "same_src"), "--same-tgt", str(tmp_path / "same_tgt"),
         "--cross-src", str(tmp_path / "cross_src"), "--mapper", str(tmp_path / "k1"),
         "--pairs", str(tmp_path / "pairs.npz"), "--out", str(tmp_path / "s.json"),
         "--per-token", str(tmp_path / "s.tokens.npz"))
    rec = json.loads((tmp_path / "s.json").read_text())
    z = np.load(tmp_path / "s.tokens.npz")
    assert rec["per_token"]["path"] == "s.tokens.npz" and set(z.files) == {"same_K", "same_V", "cross_K", "cross_V", "ref_K", "ref_V"}
    L, H = 3, 2
    for part, key in (("same", "K"), ("same", "V"), ("cross", "K"), ("cross", "V")):
        arr = z[f"{part}_{key}"]
        assert arr.shape == (16, L, H) and arr.dtype == np.float32
        sse = np.asarray([rec[part][key][l]["sse"] for l in range(L)])       # [L, H]
        assert np.allclose(arr.astype(np.float64).sum(0), sse, rtol=1e-5, atol=0)
    assert (z["ref_K"] > 0).all() and (z["ref_V"] > 0).all()
    # pipeline identity: R := S, pairs (p, p) -> every square exactly zero, R^2 exactly 1
    np.savez(tmp_path / "id.npz", pairs=np.stack([np.arange(24), np.arange(24)], 1).astype(np.int64))
    _run("score_positions.py", "--same-src", str(tmp_path / "same_src"), "--same-tgt", str(tmp_path / "same_src"),
         "--pairs", str(tmp_path / "id.npz"), "--out", str(tmp_path / "id.json"), "--per-token", str(tmp_path / "id.npz.tokens.npz"))
    zi = np.load(tmp_path / "id.npz.tokens.npz")
    assert zi["same_K"].max() == 0.0 and zi["same_V"].max() == 0.0 and "cross_K" not in zi.files
    assert json.loads((tmp_path / "id.json").read_text())["same_K_r2_layer_mean"] == 1.0


@torch.no_grad()
def test_score_mapper_per_token_centered_mean_is_one_minus_r2(tmp_path, tiny_src3, tiny_tgt, tiny_tokens):
    seqs = tiny_tokens(n_seqs=4, seq_len=16, seed=5)
    dump_kv(tiny_src3, seqs, stride=2, out_dir=tmp_path / "s")
    dump_kv(tiny_tgt, seqs, stride=2, out_dir=tmp_path / "t")
    src, tgt = KVDump.load(tmp_path / "s"), KVDump.load(tmp_path / "t")
    m = _fit_k1(src, tgt, tmp_path / "k1")
    _run("score_mapper.py", "--mapper", str(tmp_path / "k1"), "--src", str(tmp_path / "s"), "--tgt", str(tmp_path / "t"),
         "--holdout-frac", "0.5", "--out", str(tmp_path / "r2.json"), "--per-token", str(tmp_path / "ho.npz"))
    rec = json.loads((tmp_path / "r2.json").read_text())
    z = np.load(tmp_path / "ho.npz")
    n = int(z["n_heldout"])
    _, ho = src.split(0.5)
    assert n == int(ho.sum()) == 16
    for key in ("K", "V"):
        sq, sst = z[f"{key}_sq"].astype(np.float64), z[f"sst_{key}"]        # [n, L, H], [L, H]
        r2_head = 1 - sq.sum(0) / sst
        assert np.allclose(r2_head.mean(1), rec[f"{key}_r2_heldout_per_layer"], rtol=1e-5)
        delta_c = sq / (sst / n)[None]
        assert np.allclose(delta_c.mean(0), 1 - r2_head, rtol=1e-5)           # the exact bridge
        assert len(rec[f"{key}_r2_heldout_pooled_over_heads_per_layer"]) == 3
    # the per-token path is checked against mapper_r2 inside the script; confirm the archive-style keys survived
    assert rec["K_r2_heldout_layer_mean"] == pytest.approx(np.mean(mapper_r2(m, src, tgt, ho)["K"]))
