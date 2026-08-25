import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kvt.data import KVDump
from scripts.fit_mapper import check_tag_required, resolve_dump_root, resolve_masks, resolve_out_dirs


def _dump(n_seqs: int = 10, per_seq: int = 4) -> KVDump:
    meta = {"n_layers": 2, "n_kv": 2, "d_h": 16, "rope_theta": 10000.0,
            "n_seqs": n_seqs, "stride": 1}
    positions = np.tile(np.arange(per_seq), n_seqs).astype(np.int64)
    seq_idx = np.repeat(np.arange(n_seqs), per_seq).astype(np.int64)
    return KVDump(root=None, meta=meta, positions=positions, seq_idx=seq_idx)


def test_fit_mapper_exposes_the_new_flags():
    out = subprocess.run([sys.executable, "scripts/fit_mapper.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for flag in ("--space", "--n-train", "--holdout", "--tag", "--dump-root"):
        assert flag in out.stdout, f"{flag} missing from fit_mapper --help"


def test_compose_mapper_help():
    out = subprocess.run([sys.executable, "scripts/compose_mapper.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for flag in ("--a", "--b", "--out"):
        assert flag in out.stdout


def test_eval_perplexity_help():
    out = subprocess.run([sys.executable, "scripts/eval_perplexity.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for flag in ("--prefix-lens", "--n-windows", "--pair"):
        assert flag in out.stdout


def test_summarize_curve_refuses_an_empty_directory(tmp_path):
    from scripts.summarize_curve import load_curve_points
    with pytest.raises(ValueError, match="no curve points"):
        load_curve_points(tmp_path)


def test_summarize_curve_refuses_inconsistent_holdout(tmp_path):
    """The WP2 invariant made machine-checkable: every point must share one held-out range,
    or the curve is comparing different test sets and is not a curve in n."""
    from scripts.summarize_curve import load_curve_points
    (tmp_path / "n50_k1.json").write_text(json.dumps(
        {"n_train": 50, "k": 1, "holdout": [400, 420], "K_heldout": 0.6, "V_heldout": 0.5,
         "K_train": 0.7, "V_train": 0.6, "p_over_n": 0.1}))
    (tmp_path / "n100_k1.json").write_text(json.dumps(
        {"n_train": 100, "k": 1, "holdout": [380, 400], "K_heldout": 0.6, "V_heldout": 0.5,
         "K_train": 0.7, "V_train": 0.6, "p_over_n": 0.05}))
    with pytest.raises(ValueError, match="different held-out"):
        load_curve_points(tmp_path)


def test_summarize_curve_accepts_a_consistent_set(tmp_path):
    from scripts.summarize_curve import load_curve_points
    for n in (50, 100):
        (tmp_path / f"n{n}_k1.json").write_text(json.dumps(
            {"n_train": n, "k": 1, "holdout": [400, 420], "K_heldout": 0.6, "V_heldout": 0.5,
             "K_train": 0.7, "V_train": 0.6, "p_over_n": 0.1}))
    pts = load_curve_points(tmp_path)
    assert len(pts) == 2
    assert [p["n_train"] for p in pts] == [50, 100]


def test_summarize_curve_rejects_a_single_point(tmp_path):
    """A one-point directory trivially satisfies the holdout-consistency check (len({..}) == 1
    is never != 1), so it needs its own refusal or a stray/premature JSON silently produces a
    one-row summary."""
    from scripts.summarize_curve import load_curve_points
    (tmp_path / "n50_k1.json").write_text(json.dumps(
        {"n_train": 50, "k": 1, "holdout": [400, 420], "K_heldout": 0.6, "V_heldout": 0.5,
         "K_train": 0.7, "V_train": 0.6, "p_over_n": 0.1}))
    with pytest.raises(ValueError, match="at least 2"):
        load_curve_points(tmp_path)


def _curve_point(**overrides):
    from scripts.fit_mapper import curve_point
    kwargs = dict(n_train=50, k=1, holdout=(400, 420), p_over_n=0.1,
                  K_train=0.7, K_heldout=0.6, V_train=0.6, V_heldout=0.5,
                  n_train_tokens=200, n_heldout_tokens=80,
                  space="content", dump_root="data/kv/p")
    kwargs.update(overrides)
    return curve_point(**kwargs)


def test_curve_point_round_trips_through_load_curve_points(tmp_path):
    """curve_point's output is exactly what load_curve_points expects to read back -- the
    contract FIX 2 closes: the curve file schema and the reader's schema cannot drift apart."""
    from scripts.summarize_curve import load_curve_points

    pt1 = _curve_point(n_train=50)
    pt2 = _curve_point(n_train=100, p_over_n=0.05, n_train_tokens=400)
    assert set(pt1) == {"n_train", "k", "holdout", "p_over_n", "K_train", "K_heldout",
                        "V_train", "V_heldout", "n_train_tokens", "n_heldout_tokens",
                        "space", "dump_root"}

    (tmp_path / "n50_k1.json").write_text(json.dumps(pt1))
    (tmp_path / "n100_k1.json").write_text(json.dumps(pt2))
    pts = load_curve_points(tmp_path)
    assert [p["n_train"] for p in pts] == [50, 100]


def test_load_curve_points_refuses_mixed_space(tmp_path):
    """A curve mixing content-space and rope-space points is not a curve in n."""
    from scripts.summarize_curve import load_curve_points

    (tmp_path / "n50_k1.json").write_text(json.dumps(_curve_point(n_train=50, space="content")))
    (tmp_path / "n100_k1.json").write_text(json.dumps(_curve_point(n_train=100, space="rope")))
    with pytest.raises(ValueError, match="feature spaces"):
        load_curve_points(tmp_path)


def test_load_curve_points_refuses_mixed_dump_root(tmp_path):
    """A curve whose points were fitted against two different dumps is not comparable."""
    from scripts.summarize_curve import load_curve_points

    (tmp_path / "n50_k1.json").write_text(json.dumps(_curve_point(n_train=50, dump_root="data/kv/p")))
    (tmp_path / "n100_k1.json").write_text(
        json.dumps(_curve_point(n_train=100, dump_root="data/kv/p-n420")))
    with pytest.raises(ValueError, match="dump root"):
        load_curve_points(tmp_path)


def test_resolve_dump_root_defaults_to_pair_and_honors_override():
    """--dump-root must demonstrably change the path used, not just be accepted by argparse."""
    assert resolve_dump_root(None, "p") == Path("data/kv") / "p"
    assert resolve_dump_root("x/y", "p") == Path("x/y")


def test_resolve_out_dirs_tag_lands_in_both_mapper_and_results_paths():
    """--tag must reach BOTH output directories, or two differently-tagged runs could
    silently overwrite each other's artifact or report."""
    m, r = resolve_out_dirs("p", None)
    assert m == Path("mappers") / "p"
    assert r == Path("results/mapper") / "p"
    m_tag, r_tag = resolve_out_dirs("p", "n200")
    assert m_tag == Path("mappers") / "p" / "n200"
    assert r_tag == Path("results/mapper") / "p" / "n200"


def test_resolve_masks_with_n_train_gives_prefix_and_fixed_holdout():
    """--n-train/--holdout must produce a different split than the default holdout_frac
    split -- otherwise the flags are accepted but not wired into the actual masks."""
    d = _dump(n_seqs=10, per_seq=4)
    tr, ho = resolve_masks(d, n_train=4, holdout=(8, 10), holdout_frac=0.2)
    assert set(d.seq_idx[tr].tolist()) == {0, 1, 2, 3}
    assert set(d.seq_idx[ho].tolist()) == {8, 9}
    assert not (tr & ho).any()

    tr_default, ho_default = resolve_masks(d, n_train=None, holdout=None, holdout_frac=0.2)
    assert not np.array_equal(tr, tr_default)


def test_resolve_masks_requires_n_train_and_holdout_together():
    d = _dump(n_seqs=10, per_seq=4)
    with pytest.raises(ValueError, match="must be given together"):
        resolve_masks(d, n_train=4, holdout=None, holdout_frac=0.2)
    with pytest.raises(ValueError, match="must be given together"):
        resolve_masks(d, n_train=None, holdout=(8, 10), holdout_frac=0.2)


def test_resolve_masks_rejects_overlapping_prefix_and_holdout():
    d = _dump(n_seqs=10, per_seq=4)
    with pytest.raises(ValueError, match="overlaps"):
        resolve_masks(d, n_train=9, holdout=(8, 10), holdout_frac=0.2)


def test_resolve_masks_rejects_zero_n_train():
    """--n-train 0 would select zero training rows; kvt.ridge.r2_score on an empty array
    returns NaN rather than raising, so this must be refused explicitly."""
    d = _dump(n_seqs=10, per_seq=4)
    with pytest.raises(ValueError, match="must be >= 1"):
        resolve_masks(d, n_train=0, holdout=(8, 10), holdout_frac=0.2)


def test_resolve_masks_rejects_empty_holdout_range():
    """--holdout 400 400 (hi == lo) selects zero held-out rows -- same NaN failure mode."""
    d = _dump(n_seqs=10, per_seq=4)
    with pytest.raises(ValueError, match="empty"):
        resolve_masks(d, n_train=4, holdout=(8, 8), holdout_frac=0.2)


def test_resolve_masks_rejects_a_holdout_range_outside_the_actual_dump():
    """meta declares n_seqs=20 but the actual seq_idx only contains sequences 0..9, as if the
    dump were truncated after generation. [15, 20) is well-formed against KVDump.seq_range_mask's
    own bounds check (0 <= lo <= hi <= n_seqs), but selects zero rows in the real data -- exactly
    the case the final non-empty check exists to catch."""
    meta = {"n_layers": 2, "n_kv": 2, "d_h": 16, "rope_theta": 10000.0, "n_seqs": 20, "stride": 1}
    per_seq, n_seqs_actual = 4, 10
    positions = np.tile(np.arange(per_seq), n_seqs_actual).astype(np.int64)
    seq_idx = np.repeat(np.arange(n_seqs_actual), per_seq).astype(np.int64)
    d = KVDump(root=None, meta=meta, positions=positions, seq_idx=seq_idx)
    with pytest.raises(ValueError, match="zero rows"):
        resolve_masks(d, n_train=5, holdout=(15, 20), holdout_frac=0.2)


def test_check_tag_required_allows_fully_default_run():
    check_tag_required(tag=None, dump_root=None, n_train=None, space="content")


def test_check_tag_required_allows_any_nondefault_run_with_a_tag():
    check_tag_required(tag="n200", dump_root="x/y", n_train=200, space="rope")


def test_check_tag_required_refuses_nondefault_dump_root_without_tag():
    with pytest.raises(ValueError, match="--dump-root"):
        check_tag_required(tag=None, dump_root="x/y", n_train=None, space="content")


def test_check_tag_required_refuses_nondefault_n_train_without_tag():
    with pytest.raises(ValueError, match="--n-train"):
        check_tag_required(tag=None, dump_root=None, n_train=200, space="content")


def test_check_tag_required_refuses_nondefault_space_without_tag():
    with pytest.raises(ValueError, match="--space"):
        check_tag_required(tag=None, dump_root=None, n_train=None, space="rope")


def test_check_tag_required_refuses_combination_without_tag_naming_all():
    with pytest.raises(ValueError) as excinfo:
        check_tag_required(tag=None, dump_root="x/y", n_train=200, space="rope")
    msg = str(excinfo.value)
    assert "--dump-root" in msg and "--n-train" in msg and "--space" in msg


def _write_jsonl(path: Path, recs: list[dict]):
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def _perplexity_records(n_windows: int, sum_nll_per_window: float, n_tokens_per_window: int):
    return [{"window": w, "prefix_len": 512, "sum_nll": sum_nll_per_window,
             "n_tokens": n_tokens_per_window} for w in range(n_windows)]


def test_summarize_perplexity_gate_refuses_when_native_injected_diverges(tmp_path):
    from scripts.summarize_perplexity import enforce_native_injected_gate, load_and_validate_records

    _write_jsonl(tmp_path / "native.jsonl", _perplexity_records(2, sum_nll_per_window=15.0, n_tokens_per_window=100))
    # Way outside rtol=1e-4: injected total is double the native total.
    _write_jsonl(tmp_path / "native-injected.jsonl", _perplexity_records(2, sum_nll_per_window=30.0, n_tokens_per_window=100))

    records = load_and_validate_records(tmp_path)
    with pytest.raises(ValueError, match="native-injected != native"):
        enforce_native_injected_gate(records)


def test_summarize_perplexity_refuses_when_native_injected_is_absent(tmp_path):
    from scripts.summarize_perplexity import load_and_validate_records

    _write_jsonl(tmp_path / "native.jsonl", _perplexity_records(2, sum_nll_per_window=15.0, n_tokens_per_window=100))
    with pytest.raises(ValueError, match="native-injected"):
        load_and_validate_records(tmp_path)


def test_summarize_perplexity_summarizes_a_well_formed_set(tmp_path):
    from scripts.summarize_perplexity import summarize_prefix_dir

    # native and native-injected match exactly (float-noise gate trivially satisfied).
    _write_jsonl(tmp_path / "native.jsonl", _perplexity_records(2, sum_nll_per_window=15.0, n_tokens_per_window=100))
    _write_jsonl(tmp_path / "native-injected.jsonl", _perplexity_records(2, sum_nll_per_window=15.0, n_tokens_per_window=100))
    # A degraded condition: higher NLL -> higher perplexity -> positive % degradation.
    _write_jsonl(tmp_path / "content-k1.jsonl", _perplexity_records(2, sum_nll_per_window=25.0, n_tokens_per_window=100))

    rows = summarize_prefix_dir(tmp_path)
    assert set(rows) == {"native", "native-injected", "content-k1"}

    import math
    expected_native_ppl = math.exp((15.0 * 2) / (100 * 2))
    assert rows["native"]["perplexity"] == pytest.approx(expected_native_ppl)
    assert rows["native"]["pct_degradation_vs_native"] == pytest.approx(0.0)
    assert rows["native-injected"]["perplexity"] == pytest.approx(expected_native_ppl)
    assert rows["content-k1"]["perplexity"] > rows["native"]["perplexity"]
    assert rows["content-k1"]["pct_degradation_vs_native"] > 0


def test_summarize_perplexity_refuses_mismatched_window_sets(tmp_path):
    from scripts.summarize_perplexity import load_and_validate_records

    _write_jsonl(tmp_path / "native.jsonl", _perplexity_records(3, sum_nll_per_window=15.0, n_tokens_per_window=100))
    _write_jsonl(tmp_path / "native-injected.jsonl", _perplexity_records(3, sum_nll_per_window=15.0, n_tokens_per_window=100))
    other = _perplexity_records(3, sum_nll_per_window=15.0, n_tokens_per_window=100)
    other[0]["window"] = 99  # one condition scored a different window set than native
    _write_jsonl(tmp_path / "content-k1.jsonl", other)

    with pytest.raises(ValueError, match="different window set"):
        load_and_validate_records(tmp_path)


# --- eval_hellaswag.py parse_extra ---

def test_eval_hellaswag_exposes_extra_flags():
    out = subprocess.run([sys.executable, "scripts/eval_hellaswag.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for flag in ("--extra", "--middle-model"):
        assert flag in out.stdout, f"{flag} missing from eval_hellaswag --help"


def test_parse_extra_default_model_is_source():
    from scripts.eval_hellaswag import parse_extra
    out = parse_extra(["composed-BA-k1=mappers/p/composed-k1"])
    assert out == [("composed-BA-k1", "mappers/p/composed-k1", "source")]


def test_parse_extra_explicit_middle_model():
    from scripts.eval_hellaswag import parse_extra
    out = parse_extra(["composed-BA-k1=mappers/p/composed-k1@middle"])
    assert out == [("composed-BA-k1", "mappers/p/composed-k1", "middle")]


def test_parse_extra_multiple_specs():
    from scripts.eval_hellaswag import parse_extra
    out = parse_extra(["a=path/a@source", "b=path/b@middle"])
    assert out == [("a", "path/a", "source"), ("b", "path/b", "middle")]


def test_parse_extra_refuses_spec_with_no_equals():
    from scripts.eval_hellaswag import parse_extra
    with pytest.raises(ValueError, match="expected NAME=MAPPER_PATH"):
        parse_extra(["composed-BA-k1"])


def test_parse_extra_refuses_empty_name():
    from scripts.eval_hellaswag import parse_extra
    with pytest.raises(ValueError, match="NAME must not be empty"):
        parse_extra(["=mappers/p/k1"])


def test_parse_extra_refuses_unknown_model_tag():
    from scripts.eval_hellaswag import parse_extra
    with pytest.raises(ValueError, match="unknown @MODEL"):
        parse_extra(["composed-BA-k1=mappers/p/composed-k1@target"])


def test_parse_extra_refuses_duplicate_name():
    from scripts.eval_hellaswag import parse_extra
    with pytest.raises(ValueError, match="duplicate"):
        parse_extra(["a=path/a", "a=path/b@middle"])


def test_parse_extra_refuses_empty_mapper_path():
    from scripts.eval_hellaswag import parse_extra
    with pytest.raises(ValueError, match="MAPPER_PATH must not be empty"):
        parse_extra(["a=@middle"])


def test_main_refuses_at_middle_without_middle_model_flag(monkeypatch):
    """The @middle-without---middle-model check lives in main(), not parse_extra."""
    import sys as _sys
    from scripts import eval_hellaswag
    monkeypatch.setattr(_sys, "argv", [
        "eval_hellaswag.py", "--pair", "qwen3-0.6b-to-1.7b",
        "--extra", "composed-BA-k1=mappers/p/composed-k1@middle",
    ])
    with pytest.raises(SystemExit, match="--middle-model"):
        eval_hellaswag.main()


# --- summarize_hellaswag.py --compare / McNemar comparisons ---

def _hs_rec(idx, gold, lp):
    return {"idx": idx, "gold": gold, "logprobs": lp, "nbytes": [4, 4, 4, 4]}


def test_build_comparisons_computes_mcnemar_over_acc_norm():
    from scripts.summarize_hellaswag import build_comparisons
    native = [_hs_rec(i, 0, [-0.1, -2, -3, -4]) for i in range(10)]
    a = [_hs_rec(i, 0, [-0.1, -2, -3, -4]) if i < 6 else _hs_rec(i, 0, [-4, -0.1, -2, -3])
         for i in range(10)]
    b = [_hs_rec(i, 0, [-0.1, -2, -3, -4]) if i < 8 else _hs_rec(i, 0, [-4, -0.1, -2, -3])
         for i in range(10)]
    records = {"native": native, "composed-BA-k1": a, "mapped-C-k1": b}
    comps = build_comparisons(records, [("composed-BA-k1", "mapped-C-k1")])
    assert len(comps) == 1
    c = comps[0]
    assert c["a"] == "composed-BA-k1" and c["b"] == "mapped-C-k1"
    assert c["acc_a"] == pytest.approx(0.6) and c["acc_b"] == pytest.approx(0.8)
    assert c["b_count"] == 0 and c["c_count"] == 2 and c["n_discordant"] == 2
    assert c["p"] == pytest.approx(0.5)


def test_build_comparisons_refuses_a_missing_condition():
    from scripts.summarize_hellaswag import build_comparisons
    records = {"native": [_hs_rec(0, 0, [-0.1, -2, -3, -4])],
               "composed-BA-k1": [_hs_rec(0, 0, [-0.1, -2, -3, -4])]}
    with pytest.raises(ValueError) as exc:
        build_comparisons(records, [("composed-BA-k1", "does-not-exist")])
    msg = str(exc.value)
    assert "does-not-exist" in msg
    # helpful message lists what IS present
    assert "composed-BA-k1" in msg and "native" in msg


def test_render_comparisons_table_includes_both_conditions_and_acc_norm_header():
    from scripts.summarize_hellaswag import render_comparisons_table
    comps = [{"a": "composed-BA-k1", "b": "mapped-C-k1", "acc_a": 0.6, "acc_b": 0.8,
              "b_count": 0, "c_count": 2, "n_discordant": 2, "p": 0.5}]
    table = render_comparisons_table(comps)
    assert "composed-BA-k1" in table and "mapped-C-k1" in table
    assert "acc_norm" in table  # header must say so, not just be raw accuracy


def test_summarize_hellaswag_help_exposes_compare_flag():
    out = subprocess.run([sys.executable, "scripts/summarize_hellaswag.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "--compare" in out.stdout


def test_summarize_hellaswag_no_compare_output_is_byte_identical(tmp_path, monkeypatch):
    """FIX 2's hard requirement: not passing --compare must reproduce the exact prior
    output (summary.md content and summary.json condition rows), never a changed schema."""
    import sys as _sys
    from scripts import summarize_hellaswag

    pair = "qwen3-0.6b-to-1.7b"
    root = tmp_path / "results" / "hellaswag" / pair
    root.mkdir(parents=True)
    native = [_hs_rec(i, i % 4, [-0.1, -2, -3, -4]) for i in range(5)]
    mapped = [_hs_rec(i, i % 4, [-0.2, -1.9, -3, -4]) for i in range(5)]
    _write_jsonl(root / "native.jsonl", native)
    _write_jsonl(root / "mapped-k1.jsonl", mapped)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_sys, "argv", ["summarize_hellaswag.py", "--pair", pair])
    summarize_hellaswag.main()

    md = (root / "summary.md").read_text()
    js = json.loads((root / "summary.json").read_text())
    assert "comparisons" not in js
    assert "Paired comparisons" not in md
    assert set(js) == {"native", "mapped-k1"}
