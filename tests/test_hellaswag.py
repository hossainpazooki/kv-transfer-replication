import json

import numpy as np
import pytest
import torch

from kvt.hellaswag import (byte_lengths, encode, preprocess, providers, score_native,
                           score_with_cache_provider, summarize_records)
from scripts.summarize_hellaswag import load_and_validate_records


class _Tok:
    """Byte-level stand-in tokenizer: one token per byte, vocab 256."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode("utf-8"))}


def test_preprocess_matches_lm_eval_rules():
    assert preprocess("  A [title] b  c [header] d ") == "A. b c d"


def test_encode_slices_continuation_by_context_length():
    whole, n_ctx = encode(_Tok(), "abc", " de")
    assert whole == list(b"abc de") and n_ctx == 3


@torch.no_grad()
def test_native_injected_equals_native_and_identity_runs(tiny_src3, tiny_tgt):
    tok = _Tok()
    ex = {"query": "the cat sat on the", "choices": [" mat", " hat", " rug", " sun"], "gold": 0}
    lp_native = score_native(tiny_tgt, tok, ex)
    prov = providers(tiny_src3, tiny_tgt, mappers={})           # same layer count -> identity control is legal
    lp_inj = score_with_cache_provider(tiny_tgt, tok, ex, prov["native-injected"])
    assert np.allclose(lp_native, lp_inj, atol=1e-3)
    lp_id = score_with_cache_provider(tiny_tgt, tok, ex, prov["identity"])
    assert len(lp_id) == 4 and all(np.isfinite(lp_id))
    assert byte_lengths(ex["choices"]) == [4, 4, 4, 4]


def test_summarize_computes_acc_norm_and_retention():
    recs = {
        "native": [{"gold": 0, "logprobs": [-1.0, -2.0, -3.0, -4.0], "nbytes": [4, 4, 4, 4]},
                   {"gold": 1, "logprobs": [-1.0, -0.5, -3.0, -4.0], "nbytes": [4, 4, 4, 4]}],
        "mapped-k1": [{"gold": 0, "logprobs": [-1.0, -2.0, -3.0, -4.0], "nbytes": [4, 4, 4, 4]},
                      {"gold": 1, "logprobs": [-0.1, -0.5, -3.0, -4.0], "nbytes": [4, 4, 4, 4]}],
    }
    s = summarize_records(recs, chance=0.25)
    assert s["native"]["acc_norm"] == 1.0 and s["mapped-k1"]["acc_norm"] == 0.5
    assert s["mapped-k1"]["retention_raw"] == 0.5
    assert abs(s["mapped-k1"]["retention_floor_norm"] - (0.5 - 0.25) / (1.0 - 0.25)) < 1e-9
    assert "ci95" in s["native"]


def test_identity_provider_raises_valueerror_before_forward_on_layer_mismatch(tiny_src, tiny_tgt):
    """tiny_src has 2 layers, tiny_tgt has 3 -> identity control is illegal (A4) and must fail loud,
    not with a bare assert, and before wasting a forward pass on the mismatched source."""
    prov = providers(tiny_src, tiny_tgt, mappers={})
    prefix = torch.zeros((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match=r"2.*3|3.*2"):
        prov["identity"](prefix)


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _rec(idx, gold=0):
    return {"idx": idx, "gold": gold, "logprobs": [-1.0, -2.0, -3.0, -4.0], "nbytes": [4, 4, 4, 4]}


def test_load_and_validate_records_raises_on_no_jsonl_files(tmp_path):
    empty_dir = tmp_path / "empty_pair"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match=str(empty_dir).replace("\\", "\\\\")):
        load_and_validate_records(empty_dir)


def test_load_and_validate_records_raises_when_native_missing(tmp_path):
    _write_jsonl(tmp_path / "mapped-k1.jsonl", [_rec(0), _rec(1)])
    with pytest.raises(ValueError) as exc:
        load_and_validate_records(tmp_path)
    msg = str(exc.value)
    assert "native.jsonl is missing" in msg
    assert "mapped-k1" in msg


def test_load_and_validate_records_raises_on_idx_mismatch(tmp_path):
    _write_jsonl(tmp_path / "native.jsonl", [_rec(0), _rec(1), _rec(2)])
    # mapped-k1 scored a different set of examples than native (2 dropped, 5 extra)
    _write_jsonl(tmp_path / "mapped-k1.jsonl", [_rec(0), _rec(1), _rec(5)])
    with pytest.raises(ValueError, match="mapped-k1"):
        load_and_validate_records(tmp_path)


def test_load_and_validate_records_raises_on_expect_n_mismatch(tmp_path):
    _write_jsonl(tmp_path / "native.jsonl", [_rec(0), _rec(1)])
    _write_jsonl(tmp_path / "mapped-k1.jsonl", [_rec(0), _rec(1)])
    with pytest.raises(ValueError) as exc:
        load_and_validate_records(tmp_path, expect_n=3)
    msg = str(exc.value)
    assert "--expect-n=3" in msg
    assert "native" in msg and "mapped-k1" in msg


def test_load_and_validate_records_raises_on_count_mismatch_without_expect_n(tmp_path):
    """The original motivating case: one condition truncated mid-run, no --expect-n given."""
    _write_jsonl(tmp_path / "native.jsonl", [_rec(0), _rec(1), _rec(2)])
    _write_jsonl(tmp_path / "mapped-k1.jsonl", [_rec(0), _rec(1)])
    with pytest.raises(ValueError) as exc:
        load_and_validate_records(tmp_path)
    msg = str(exc.value)
    assert "native has 3 records" in msg
    assert "mapped-k1" in msg


def test_load_and_validate_records_raises_on_empty_native_and_mapped(tmp_path):
    """Reproduces the reported hole: native.jsonl and mapped-k1.jsonl both exist but are
    empty. Must raise, never fall through to a full-NaN summary."""
    (tmp_path / "native.jsonl").write_text("")
    (tmp_path / "mapped-k1.jsonl").write_text("")
    with pytest.raises(ValueError) as exc:
        load_and_validate_records(tmp_path)
    msg = str(exc.value)
    assert "zero scored examples" in msg
    assert "native" in msg and "mapped-k1" in msg


def test_load_and_validate_records_raises_on_zero_record_non_native_condition(tmp_path):
    _write_jsonl(tmp_path / "native.jsonl", [_rec(0), _rec(1)])
    (tmp_path / "mapped-k1.jsonl").write_text("")
    with pytest.raises(ValueError) as exc:
        load_and_validate_records(tmp_path)
    msg = str(exc.value)
    assert "zero scored examples" in msg
    assert "mapped-k1" in msg


def test_load_and_validate_records_raises_on_duplicate_idx_within_condition(tmp_path):
    _write_jsonl(tmp_path / "native.jsonl", [_rec(0), _rec(1), _rec(2)])
    _write_jsonl(tmp_path / "mapped-k1.jsonl", [_rec(0), _rec(0), _rec(2)])  # idx 0 duplicated
    with pytest.raises(ValueError) as exc:
        load_and_validate_records(tmp_path)
    msg = str(exc.value)
    assert "mapped-k1" in msg
    assert "duplicate" in msg
    assert "0" in msg


def test_load_and_validate_records_raises_on_malformed_json_names_file_and_line(tmp_path):
    p = tmp_path / "native.jsonl"
    p.write_text(json.dumps(_rec(0)) + "\n" + "not valid json\n")
    _write_jsonl(tmp_path / "mapped-k1.jsonl", [_rec(0)])
    with pytest.raises(ValueError) as exc:
        load_and_validate_records(tmp_path)
    msg = str(exc.value)
    assert "native.jsonl" in msg
    assert "line 2" in msg


def test_load_and_validate_records_happy_path(tmp_path):
    _write_jsonl(tmp_path / "native.jsonl", [_rec(0), _rec(1), _rec(2)])
    _write_jsonl(tmp_path / "mapped-k1.jsonl", [_rec(0), _rec(1), _rec(2)])
    records = load_and_validate_records(tmp_path, expect_n=3)
    assert set(records) == {"native", "mapped-k1"}
    assert len(records["native"]) == 3
    s = summarize_records(records, chance=0.25)
    assert s["native"]["n"] == 3 and s["mapped-k1"]["n"] == 3
    assert "retention_raw" in s["mapped-k1"]
