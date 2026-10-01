"""G3 checkpoint interface: pinned revisions, local checkpoints, shard manifests, dump provenance.
All offline: tiny models saved with save_pretrained, and the Hub replaced by monkeypatches."""
import json
from pathlib import Path

import pytest
import torch

import kvt.checkpoint as ckpt
import kvt.models as models
from kvt.checkpoint import (MANIFEST_FILE, CheckpointMismatch, build_manifest, manifest_text, record_checkpoint,
                            verify_dump_checkpoint, verify_manifest)
from kvt.data import KVDump, dump_kv
from kvt.pairs import PAIRS, ModelRef, Pair


@pytest.fixture
def local_ckpt(tmp_path, tiny_tgt):
    d = tmp_path / "ckpt"
    tiny_tgt.save_pretrained(d)
    return d


def _flip_last_byte(path: Path) -> None:
    """Mutate a shard the way a trainer overwriting it would, without truncating: safetensors mmaps
    the loaded file, and Windows refuses to truncate a mapped file (OSError 22) while an in-place
    write goes through on every platform."""
    with open(path, "r+b") as f:
        f.seek(-1, 2)
        last = f.read(1)[0]
        f.seek(-1, 2)
        f.write(bytes([last ^ 1]))


def _fake_shards(d: Path, names=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:                                   # bytes depend on the name, not the write order
        i = int(n[6:11]) - 1
        (d / n).write_bytes(bytes([i]) * (1000 + i))
    (d / "config.json").write_text('{"a": 1}')
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{n}": n for n in sorted(names)}}))
    return d


# ---- legacy pairs -------------------------------------------------------------------------------

def test_legacy_pairs_are_unchanged():
    p = Pair("n", "org/a", "org/b")
    assert (p.name, p.source, p.target) == ("n", "org/a", "org/b")
    assert p == Pair("n", "org/a", "org/b")
    assert p.model_ref("source") == ModelRef("org/a") and p.model_ref("target") == ModelRef("org/b")
    for pair in PAIRS.values():
        for w in ("source", "target"):
            r = pair.model_ref(w)
            assert r.model_id == getattr(pair, w) and r.revision is None and r.local_path is None


def test_legacy_string_load_calls_from_pretrained_exactly_as_before(monkeypatch):
    calls = []

    class Fake:
        @staticmethod
        def from_pretrained(path, **kw):
            calls.append((path, kw))
            return torch.nn.Linear(1, 1)
    monkeypatch.setattr(models, "AutoModelForCausalLM", Fake)
    models.load_model("Qwen/Qwen3-0.6B")
    assert calls == [("Qwen/Qwen3-0.6B", {"dtype": torch.float32, "attn_implementation": models.ATTN_IMPLEMENTATION})]
    assert models.pretrained_args("Qwen/Qwen3-0.6B") == ("Qwen/Qwen3-0.6B", {})
    assert models.pretrained_args(ModelRef("Qwen/Qwen3-0.6B")) == ("Qwen/Qwen3-0.6B", {})


# ---- revision -----------------------------------------------------------------------------------

def test_revision_is_passed_to_model_tokenizer_and_config(monkeypatch):
    seen = {}

    def fake(kind):
        class F:
            @staticmethod
            def from_pretrained(path, **kw):
                seen[kind] = (path, kw.get("revision"), kw.get("local_files_only"))
                if kind == "config":
                    return types_ns()
                return torch.nn.Linear(1, 1)
        return F

    def types_ns():
        from types import SimpleNamespace
        return SimpleNamespace(rope_parameters={"rope_theta": 1e6}, max_position_embeddings=0)

    import transformers
    monkeypatch.setattr(models, "AutoModelForCausalLM", fake("model"))
    monkeypatch.setattr(models, "AutoTokenizer", fake("tok"))
    monkeypatch.setattr(transformers, "AutoConfig", fake("config"))
    ref = ModelRef("org/m", revision="abc123")
    models.load_model(ref, rope_scaling={"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 8})
    models.load_tokenizer(ref)
    assert seen == {"model": ("org/m", "abc123", None), "tok": ("org/m", "abc123", None),
                    "config": ("org/m", "abc123", None)}


def test_pair_carries_per_side_revision():
    p = Pair("n", "org/a", "org/b", target_revision="step-100")
    assert p.model_ref("target") == ModelRef("org/b", revision="step-100")
    assert p.model_ref("source") == ModelRef("org/a")


# ---- local path ---------------------------------------------------------------------------------

def test_local_path_loads_that_checkpoint(local_ckpt, tiny_tgt):
    m = models.load_model(ModelRef("base/model", local_path=local_ckpt))
    for (k, a), (_, b) in zip(tiny_tgt.state_dict().items(), m.state_dict().items()):
        assert torch.equal(a, b.cpu()), k
    assert models.pretrained_args(ModelRef("base/model", local_path=local_ckpt)) == (
        str(local_ckpt), {"local_files_only": True})


def test_local_path_that_is_missing_or_incomplete_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        models.load_model(ModelRef("base", local_path=tmp_path / "nope"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no config.json"):
        models.load_model(ModelRef("base", local_path=tmp_path / "empty"))


# ---- ambiguous / invalid ------------------------------------------------------------------------

@pytest.mark.parametrize("kw, err, match", [
    (dict(model_id="m", revision="r", local_path=Path("x")), ValueError, "ambiguous"),
    (dict(model_id="", revision=None), ValueError, "model_id"),
    (dict(model_id="m", revision=""), ValueError, "revision"),
    (dict(model_id="m", revision="  "), ValueError, "revision"),
    (dict(model_id="m", local_path="some/str/path"), TypeError, "pathlib.Path"),
])
def test_model_ref_rejects_ambiguous_or_invalid(kw, err, match):
    with pytest.raises(err, match=match):
        ModelRef(**kw)


def test_pair_rejects_ambiguous_side_and_unknown_side():
    with pytest.raises(ValueError, match="ambiguous"):
        Pair("n", "a", "b", source_revision="r", source_local_path=Path("x"))
    with pytest.raises(ValueError, match="which"):
        Pair("n", "a", "b").model_ref("middle")


def test_loader_rejects_non_ref_objects():
    with pytest.raises(TypeError):
        models.pretrained_args(Path("x"))


# ---- manifest -----------------------------------------------------------------------------------

def test_manifest_is_deterministic_and_content_addressed(tmp_path):
    a = _fake_shards(tmp_path / "a")
    b = _fake_shards(tmp_path / "b", names=("model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"))
    ma, mb = build_manifest(a), build_manifest(b)
    assert ma == build_manifest(a) and manifest_text(ma) == manifest_text(build_manifest(a))
    assert ma == mb, "same bytes in another directory, written in another order, must give the same manifest"
    assert [f["path"] for f in ma["files"]] == sorted(f["path"] for f in ma["files"])
    assert {f["path"] for f in ma["files"]} == {"config.json", "model.safetensors.index.json",
                                                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
    import hashlib
    shard = a / "model-00001-of-00002.safetensors"
    entry = next(f for f in ma["files"] if f["path"] == shard.name)
    assert entry["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest() and entry["size"] == 1000
    (b / "model-00002-of-00002.safetensors").write_bytes(b"\x01" * 1000 + b"\x02")
    assert build_manifest(b)["digest"] != ma["digest"]


def test_manifest_refuses_no_safetensors_and_missing_indexed_shard(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(CheckpointMismatch, match="no \\*.safetensors"):
        build_manifest(tmp_path / "bin")
    d = _fake_shards(tmp_path / "partial")
    (d / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(CheckpointMismatch, match="names shards that are not in"):
        build_manifest(d)


def test_verify_manifest_names_every_difference(tmp_path):
    d = _fake_shards(tmp_path / "c")
    m = build_manifest(d)
    verify_manifest(m, d)
    (d / "model-00001-of-00002.safetensors").write_bytes(b"\x09" * 1000)       # same size, different bytes
    with pytest.raises(CheckpointMismatch, match="changed model-00001"):
        verify_manifest(m, d)
    d2 = _fake_shards(tmp_path / "d")
    (d2 / "extra.safetensors").write_bytes(b"e")
    with pytest.raises(CheckpointMismatch, match="unexpected extra.safetensors"):
        verify_manifest(m, d2)
    d3 = _fake_shards(tmp_path / "e")
    (d3 / "config.json").unlink()
    with pytest.raises(CheckpointMismatch, match="missing config.json"):
        verify_manifest(m, d3)


def test_hand_edited_manifest_is_refused(tmp_path):
    d = _fake_shards(tmp_path / "c")
    m = build_manifest(d)
    m["files"][0]["sha256"] = "0" * 64
    with pytest.raises(CheckpointMismatch, match="digest does not match"):
        verify_manifest(m, d)




def test_hf_ref_resolves_to_the_snapshot_commit_and_pins_it(tmp_path, monkeypatch):
    commit = "a" * 40
    snap = _fake_shards(tmp_path / "hub" / "snapshots" / commit)
    calls = []

    def fake_snapshot_download(repo_id, revision=None, allow_patterns=None, local_files_only=False):
        calls.append((repo_id, revision, tuple(allow_patterns), local_files_only))
        return str(snap)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    rec = record_checkpoint(ModelRef("org/m", revision="main"))
    assert calls == [("org/m", "main", tuple(ckpt.HF_PATTERNS), False)]
    assert rec.resolved_commit == commit and rec.checkpoint_dir == snap
    assert rec.load_ref() == ModelRef("org/m", revision=commit)
    assert rec.provenance() == {"kind": "hf", "model_id": "org/m", "revision": "main", "resolved_commit": commit,
                                "local_checkpoint": None, "manifest_file": MANIFEST_FILE,
                                "manifest_digest": rec.manifest["digest"]}
    # Locating it again for verification reads the cache for the COMMIT and never downloads.
    assert ckpt.locate_checkpoint(rec.provenance()) == snap
    assert calls[-1] == ("org/m", commit, tuple(ckpt.HF_PATTERNS), True)


def test_provenance_never_records_a_filesystem_path(tmp_path, local_ckpt):
    for ref in (ModelRef("base/model", local_path=local_ckpt), ModelRef(str(local_ckpt))):
        prov = record_checkpoint(ref).provenance()
        assert prov["kind"] == "local" and prov["local_checkpoint"] == "ckpt"
        assert str(tmp_path) not in json.dumps(prov) and "/" not in prov["local_checkpoint"]
    assert record_checkpoint(ModelRef(str(local_ckpt))).provenance()["model_id"] == "ckpt"


# ---- dump provenance + automatic verification on load -------------------------------------------

@torch.no_grad()
def _dump_with_record(tmp_path, local_ckpt, tiny_tokens, name="with"):
    rec = record_checkpoint(ModelRef("base/model", local_path=local_ckpt))
    model = models.load_model(rec.load_ref())
    dump_kv(model, tiny_tokens(n_seqs=2, seq_len=8), stride=2, out_dir=tmp_path / name, checkpoint=rec)
    return rec, model


@torch.no_grad()
def test_dump_records_provenance_and_keeps_existing_metadata(tmp_path, local_ckpt, tiny_tokens):
    rec, model = _dump_with_record(tmp_path, local_ckpt, tiny_tokens)
    dump_kv(model, tiny_tokens(n_seqs=2, seq_len=8), stride=2, out_dir=tmp_path / "without")
    text = (tmp_path / "with" / "meta.json").read_text()
    with_meta, without_meta = json.loads(text), json.loads((tmp_path / "without" / "meta.json").read_text())
    assert "checkpoint" not in without_meta and not (tmp_path / "without" / MANIFEST_FILE).exists()
    assert without_meta["model"] == str(local_ckpt)          # legacy behavior: _name_or_path, untouched
    assert with_meta["model"] == "base/model"                  # with provenance: the id, never the path
    assert ({k: v for k, v in with_meta.items() if k not in ("checkpoint", "model")}
            == {k: v for k, v in without_meta.items() if k != "model"}), "existing metadata intact"
    assert with_meta["checkpoint"] == {
        "kind": "local", "model_id": "base/model", "revision": None, "resolved_commit": None,
        "local_checkpoint": "ckpt", "manifest_file": MANIFEST_FILE, "manifest_digest": rec.manifest["digest"]}
    assert str(tmp_path) not in text
    assert (tmp_path / "with" / MANIFEST_FILE).read_text() == manifest_text(rec.manifest)
    assert any(f["path"].endswith(".safetensors") for f in rec.manifest["files"])


@torch.no_grad()
def test_legacy_dump_loads_without_any_checkpoint(tmp_path, tiny_tgt, tiny_tokens, monkeypatch):
    monkeypatch.delenv(ckpt.CHECKPOINT_ROOT_ENV, raising=False)
    dump_kv(tiny_tgt, tiny_tokens(n_seqs=2, seq_len=8), stride=2, out_dir=tmp_path)
    assert KVDump.load(tmp_path).n_seqs == 2
    with pytest.raises(CheckpointMismatch, match="no checkpoint provenance"):
        verify_dump_checkpoint(tmp_path)                       # explicit verification still refuses it


@torch.no_grad()
def test_kvdump_load_verifies_provenance_automatically(tmp_path, local_ckpt, tiny_tokens, monkeypatch):
    rec, _ = _dump_with_record(tmp_path, local_ckpt, tiny_tokens)
    dump = tmp_path / "with"
    monkeypatch.delenv(ckpt.CHECKPOINT_ROOT_ENV, raising=False)
    with pytest.raises(ckpt.CheckpointUnavailable, match=ckpt.CHECKPOINT_ROOT_ENV):
        KVDump.load(dump)                                      # cannot find it -> refused, not skipped
    assert KVDump.load(dump, checkpoint_dir=local_ckpt).n_seqs == 2
    monkeypatch.setenv(ckpt.CHECKPOINT_ROOT_ENV, str(local_ckpt.parent))
    assert KVDump.load(dump).n_seqs == 2
    assert verify_dump_checkpoint(dump) == rec.manifest

    copy = tmp_path / "elsewhere" / "ckpt"                     # same bytes, another machine's layout
    copy.mkdir(parents=True)
    for f in local_ckpt.iterdir():
        (copy / f.name).write_bytes(f.read_bytes())
    monkeypatch.setenv(ckpt.CHECKPOINT_ROOT_ENV, str(copy.parent))
    assert KVDump.load(dump).n_seqs == 2

    _flip_last_byte(next(copy.glob("*.safetensors")))
    with pytest.raises(CheckpointMismatch, match="changed .*safetensors"):
        KVDump.load(dump)
    with pytest.raises(CheckpointMismatch, match="changed"):
        verify_dump_checkpoint(dump)


@torch.no_grad()
def test_kvdump_load_refuses_a_swapped_or_missing_manifest(tmp_path, local_ckpt, tiny_tokens):
    _dump_with_record(tmp_path, local_ckpt, tiny_tokens, name="d")
    d = tmp_path / "d"
    (d / MANIFEST_FILE).write_text(manifest_text(build_manifest(_fake_shards(tmp_path / "other"))))
    with pytest.raises(CheckpointMismatch, match="meta.json's"):
        KVDump.load(d, checkpoint_dir=local_ckpt)
    (d / MANIFEST_FILE).unlink()
    with pytest.raises(CheckpointMismatch, match="missing"):
        KVDump.load(d, checkpoint_dir=local_ckpt)


# ---- scripts/dump_kv.py end to end, Hub mocked ---------------------------------------------------

class _ScriptHarness:
    """Runs scripts.dump_kv.main in-process against a fake Hub of two tiny saved models, logging
    the order of hashing, loading and verification."""

    def __init__(self, tmp_path, monkeypatch, tiny_src3, tiny_tgt, tiny_tokens):
        import huggingface_hub
        import transformers
        import scripts.dump_kv as script
        self.tmp, self.script, self.events = tmp_path, script, []
        self.commit = "c" * 40
        self.hub = {"org/src": tmp_path / "hub" / "src" / ("b" * 40), "org/tgt": tmp_path / "hub" / "tgt" / self.commit}
        tiny_src3.save_pretrained(self.hub["org/src"])
        tiny_tgt.save_pretrained(self.hub["org/tgt"])
        self.runs = tmp_path / "runs"
        tiny_src3.save_pretrained(self.runs / "step-100")
        self.tokens = tmp_path / "tok.npy"
        import numpy as np
        np.save(self.tokens, tiny_tokens(n_seqs=2, seq_len=8))
        monkeypatch.setitem(PAIRS, "tiny", Pair("tiny", "org/src", "org/tgt"))
        monkeypatch.delenv(ckpt.CHECKPOINT_ROOT_ENV, raising=False)

        def snapshot_download(repo_id, revision=None, allow_patterns=None, local_files_only=False):
            d = self.hub[repo_id]
            if revision not in (None, "main", d.name):
                raise FileNotFoundError(f"no revision {revision}")
            self.events.append(("snapshot", repo_id, revision, local_files_only))
            return str(d)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

        for cls in (transformers.AutoConfig, transformers.AutoModelForCausalLM):
            real = cls.from_pretrained

            def fp(path, *a, revision=None, local_files_only=None, _real=real, _cls=cls.__name__, **kw):
                if Path(path).is_dir():
                    self.events.append((_cls, "dir", Path(path).name, local_files_only))
                else:                                          # an HF id: must resolve to the fake Hub
                    d = self.hub[path]
                    assert revision in (None, d.name), f"{path} loaded at unhashed revision {revision}"
                    self.events.append((_cls, path, revision))
                    path = str(d)
                return _real(path, *a, **kw)
            monkeypatch.setattr(cls, "from_pretrained", staticmethod(fp))

        def logged(name, fn):
            def w(*a, **kw):
                self.events.append((name, *a))
                return fn(*a, **kw)
            return w
        monkeypatch.setattr(script, "record_checkpoint", logged("record", script.record_checkpoint))
        monkeypatch.setattr(script, "load_model", logged("load", script.load_model))
        monkeypatch.setattr(ckpt, "verify_manifest", logged("verify", ckpt.verify_manifest))

    def run(self, *args, out="out"):
        self.script.main(["--pair", "tiny", "--tokens", str(self.tokens), "--stride", "2",
                          "--out", str(self.tmp / out), *args])
        return self.tmp / out

    def names(self):
        return [e[0] for e in self.events if e[0] in ("record", "load", "verify")]


@pytest.fixture
def harness(tmp_path, monkeypatch, tiny_src3, tiny_tgt, tiny_tokens):
    return _ScriptHarness(tmp_path, monkeypatch, tiny_src3, tiny_tgt, tiny_tokens)


def test_script_hf_revision_hashes_then_loads_the_pinned_commit(harness):
    out = harness.run("--which", "target", "--revision", "main")
    assert harness.names() == ["record", "load", "verify"], "hash before load, re-verify after the dump"
    assert harness.events[0] == ("record", ModelRef("org/tgt", revision="main"))
    assert ("snapshot", "org/tgt", "main", False) in harness.events
    load = next(e for e in harness.events if e[0] == "load")
    assert load[1] == ModelRef("org/tgt", revision=harness.commit)       # the commit that was hashed
    assert ("AutoModelForCausalLM", "org/tgt", harness.commit) in harness.events
    meta = json.loads((out / "meta.json").read_text())
    want = build_manifest(harness.hub["org/tgt"])
    assert meta["checkpoint"] == {"kind": "hf", "model_id": "org/tgt", "revision": "main",
                                  "resolved_commit": harness.commit, "local_checkpoint": None,
                                  "manifest_file": MANIFEST_FILE, "manifest_digest": want["digest"]}
    assert meta["model"] == "org/tgt" and str(harness.tmp) not in (out / "meta.json").read_text()
    assert json.loads((out / MANIFEST_FILE).read_text()) == want
    harness.events.clear()
    assert KVDump.load(out).n_seqs == 2                                  # auto-verified from the HF cache
    assert ("snapshot", "org/tgt", harness.commit, True) in harness.events and harness.names() == ["verify"]


def test_script_unpinned_hf_side_is_still_pinned_to_what_was_hashed(harness):
    harness.run("--which", "source")
    load = next(e for e in harness.events if e[0] == "load")
    assert load[1] == ModelRef("org/src", revision="b" * 40)


def test_script_local_checkpoint_is_loaded_and_recorded_without_paths(harness, monkeypatch):
    local = harness.runs / "step-100"
    out = harness.run("--which", "source", "--local-path", str(local))
    assert harness.names() == ["record", "load", "verify"]
    load = next(e for e in harness.events if e[0] == "load")
    assert load[1] == ModelRef("org/src", local_path=local)
    assert ("AutoModelForCausalLM", "dir", "step-100", True) in harness.events   # local_files_only
    assert not any(e[:2] == ("AutoModelForCausalLM", "org/src") for e in harness.events)
    text = (out / "meta.json").read_text()
    meta = json.loads(text)
    assert meta["model"] == "org/src" and meta["checkpoint"]["kind"] == "local"
    assert meta["checkpoint"]["local_checkpoint"] == "step-100"
    assert meta["checkpoint"]["manifest_digest"] == build_manifest(local)["digest"]
    assert str(harness.tmp) not in text and str(Path.home()) not in text
    with pytest.raises(ckpt.CheckpointUnavailable):
        KVDump.load(out)
    monkeypatch.setenv(ckpt.CHECKPOINT_ROOT_ENV, str(harness.runs))
    assert KVDump.load(out).n_seqs == 2


def test_script_invalidates_the_dump_when_the_checkpoint_changes_mid_dump(harness, monkeypatch):
    local = harness.runs / "step-100"
    real_dump = harness.script.dump_kv

    def dump_then_trainer_overwrites(*a, **kw):
        real_dump(*a, **kw)
        _flip_last_byte(next(local.glob("*.safetensors")))
    monkeypatch.setattr(harness.script, "dump_kv", dump_then_trainer_overwrites)
    with pytest.raises(SystemExit, match="checkpoint changed during the dump"):
        harness.run("--which", "source", "--local-path", str(local))
    out = harness.tmp / "out"
    assert not (out / "meta.json").exists() and (out / "meta.json.INVALID").exists()
    with pytest.raises(FileNotFoundError):
        KVDump.load(out, checkpoint_dir=local)


def test_script_refuses_to_overwrite_a_dump_from_a_different_checkpoint(harness):
    local = harness.runs / "step-100"
    harness.run("--which", "source", "--local-path", str(local))
    _flip_last_byte(next(local.glob("*.safetensors")))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        harness.run("--which", "source", "--local-path", str(local))


def test_script_rejects_revision_with_local_path(harness):
    with pytest.raises(SystemExit, match="ambiguous"):
        harness.run("--which", "source", "--revision", "main", "--local-path", str(harness.runs / "step-100"))
    assert harness.names() == []                               # refused before hashing or loading anything
