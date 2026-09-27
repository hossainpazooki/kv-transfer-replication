"""Checkpoint provenance for KV dumps (bridge spec G3).

A dump is only traceable if we know exactly which weights produced it. `record_checkpoint` resolves
a ModelRef to the directory the weights are read from (a local checkpoint, or the HF cache snapshot
of the resolved commit) and hashes every relevant file into a manifest; `dump_kv` writes that
manifest next to meta.json and records its digest there. `verify_dump_checkpoint` re-hashes a
checkpoint and raises CheckpointMismatch on any difference -- never a warning, never a default.

Manifest format (deterministic: entries sorted by path, canonical JSON for the digest):
    {"format": "kvt-checkpoint-manifest/1", "algorithm": "sha256",
     "files": [{"path": "model.safetensors", "size": 123, "sha256": "..."}, ...],
     "digest": sha256 of the canonical JSON of the three keys above}
The checkpoint's directory is NOT part of the manifest, so the same bytes at a different path
verify.

meta.json's "checkpoint" block never holds a filesystem path (it would leak the home directory and
means nothing on another machine). It records what identifies the weights -- the manifest digest --
plus what is needed to find them again: an HF (model_id, resolved_commit), which the HF cache
resolves on any machine, or a local checkpoint's directory NAME, looked up under
$KVT_CHECKPOINT_ROOT or an explicit checkpoint_dir. KVDump.load verifies every dump that carries
this block; a checkpoint that cannot be found is refused, not skipped.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from kvt.pairs import ModelRef

MANIFEST_FORMAT = "kvt-checkpoint-manifest/1"
MANIFEST_FILE = "checkpoint_manifest.json"
# The weights plus the two files that decide how they are read: the shard index (tensor -> shard)
# and config.json (architecture, RoPE). Same weights under a different config give different K/V.
EXTRA_FILES = ("model.safetensors.index.json", "config.json")
HF_PATTERNS = ["*.safetensors", *EXTRA_FILES]
_CHUNK = 16 << 20


CHECKPOINT_ROOT_ENV = "KVT_CHECKPOINT_ROOT"


class CheckpointMismatch(ValueError):
    """The checkpoint on disk is not the one a manifest / dump records."""


class CheckpointUnavailable(CheckpointMismatch):
    """The checkpoint a dump records cannot be found, so the dump cannot be verified (refused)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _digest(format_: str, algorithm: str, files: list) -> str:
    return hashlib.sha256(_canonical({"format": format_, "algorithm": algorithm, "files": files})).hexdigest()


def relevant_files(checkpoint_dir: Path) -> list[Path]:
    """Top-level *.safetensors shards plus the index and config, sorted by name.

    Refuses a directory with no safetensors (e.g. pytorch_model.bin only: nothing we can hash as
    the spec asks) and an index naming a shard that is not there (a partially-copied checkpoint).
    """
    d = Path(checkpoint_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"checkpoint directory {d} does not exist")
    shards = sorted(p for p in d.glob("*.safetensors") if p.is_file())
    if not shards:
        raise CheckpointMismatch(f"no *.safetensors files in {d}; cannot build a checkpoint manifest")
    index = d / "model.safetensors.index.json"
    if index.is_file():
        wanted = set(json.loads(index.read_text()).get("weight_map", {}).values())
        missing = sorted(wanted - {p.name for p in shards})
        if missing:
            raise CheckpointMismatch(f"{index} names shards that are not in {d}: {missing}")
    extras = [d / n for n in EXTRA_FILES if (d / n).is_file()]
    return sorted(shards + extras, key=lambda p: p.name)


def build_manifest(checkpoint_dir: Path) -> dict:
    files = [{"path": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)}
             for p in relevant_files(checkpoint_dir)]
    return {"format": MANIFEST_FORMAT, "algorithm": "sha256", "files": files,
            "digest": _digest(MANIFEST_FORMAT, "sha256", files)}


def manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def check_manifest_integrity(manifest: dict) -> None:
    """The manifest is a known format and its digest matches its own entries (not hand-edited)."""
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("algorithm") != "sha256":
        raise CheckpointMismatch(f"unknown manifest format {manifest.get('format')!r} / "
                                 f"algorithm {manifest.get('algorithm')!r}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise CheckpointMismatch("manifest lists no files")
    if manifest.get("digest") != _digest(manifest["format"], manifest["algorithm"], files):
        raise CheckpointMismatch("manifest digest does not match its own file entries")


def verify_manifest(manifest: dict, checkpoint_dir: Path) -> None:
    """Re-hash `checkpoint_dir` and raise CheckpointMismatch naming every file that differs."""
    check_manifest_integrity(manifest)
    want = {f["path"]: f for f in manifest["files"]}
    got = {f["path"]: f for f in build_manifest(checkpoint_dir)["files"]}
    problems = [f"missing {p}" for p in sorted(want.keys() - got.keys())]
    problems += [f"unexpected {p}" for p in sorted(got.keys() - want.keys())]
    problems += [f"changed {p}" for p in sorted(want.keys() & got.keys()) if want[p] != got[p]]
    if problems:
        raise CheckpointMismatch(f"checkpoint {checkpoint_dir} does not match the manifest: " + "; ".join(problems))


def resolve_checkpoint_dir(ref: ModelRef) -> tuple[Path, str | None]:
    """(directory the weights are read from, resolved HF commit or None for a local checkpoint).

    An HF ref resolves through the HF cache snapshot, whose directory name IS the commit hash; the
    files are the ones from_pretrained reads, so hashing them hashes what was loaded.
    """
    if ref.local_path is not None:
        return ref.local_path.resolve(), None
    if Path(ref.model_id).is_dir():             # a path given as the id, as from_pretrained accepts
        return Path(ref.model_id).resolve(), None
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(ref.model_id, revision=ref.revision, allow_patterns=HF_PATTERNS))
    return snap, snap.name


@dataclass(frozen=True)
class CheckpointRecord:
    ref: ModelRef
    checkpoint_dir: Path
    resolved_commit: str | None
    manifest: dict

    def provenance(self) -> dict:
        """The "checkpoint" block of meta.json -- path-free, see the module docstring."""
        local = self.resolved_commit is None
        # A model_id that is itself a directory (from_pretrained accepts one) is a path: keep its name only.
        model_id = self.ref.model_id if self.ref.local_path is not None or not local else Path(self.ref.model_id).name
        return {
            "kind": "local" if local else "hf",
            "model_id": model_id,
            "revision": self.ref.revision,
            "resolved_commit": self.resolved_commit,
            "local_checkpoint": self.checkpoint_dir.name if local else None,
            "manifest_file": MANIFEST_FILE,
            "manifest_digest": self.manifest["digest"],
        }

    def load_ref(self) -> ModelRef:
        """The ref to load weights with: an HF ref pinned to the commit that was hashed, so the
        branch cannot move between hashing and loading."""
        if self.resolved_commit is None:
            return self.ref
        return ModelRef(self.ref.model_id, revision=self.resolved_commit)

    def verify(self) -> None:
        verify_manifest(self.manifest, self.checkpoint_dir)


def record_checkpoint(ref: ModelRef) -> CheckpointRecord:
    d, commit = resolve_checkpoint_dir(ref)
    return CheckpointRecord(ref, d, commit, build_manifest(d))


def locate_checkpoint(prov: dict, checkpoint_dir=None) -> Path:
    """Where to find the checkpoint a provenance block names, on THIS machine.

    `checkpoint_dir` wins. Otherwise an HF checkpoint is the cached snapshot of its resolved commit
    (never downloaded here -- a dump load must not fetch gigabytes), and a local one is
    $KVT_CHECKPOINT_ROOT/<local_checkpoint>. Anything not found raises CheckpointUnavailable.
    """
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    kind = prov.get("kind")
    if kind == "hf":
        from huggingface_hub import snapshot_download
        try:
            return Path(snapshot_download(prov["model_id"], revision=prov["resolved_commit"],
                                          allow_patterns=HF_PATTERNS, local_files_only=True))
        except Exception as e:
            raise CheckpointUnavailable(
                f"{prov['model_id']}@{prov['resolved_commit']} is not in the local HF cache ({type(e).__name__}); "
                f"download it or pass checkpoint_dir to verify this dump") from e
    if kind == "local":
        root = os.environ.get(CHECKPOINT_ROOT_ENV)
        if not root:
            raise CheckpointUnavailable(
                f"dump was made from local checkpoint {prov.get('local_checkpoint')!r}; set {CHECKPOINT_ROOT_ENV} to "
                f"the directory holding it, or pass checkpoint_dir, to verify this dump")
        d = Path(root) / prov["local_checkpoint"]
        if not d.is_dir():
            raise CheckpointUnavailable(f"local checkpoint {prov['local_checkpoint']!r} not found under {root}")
        return d
    raise CheckpointMismatch(f"unknown checkpoint kind {kind!r} in provenance")


def verify_dump_meta(dump_dir, meta: dict, checkpoint_dir=None) -> dict:
    """Verify a dump whose meta.json (already parsed) carries a "checkpoint" block; return the manifest.

    Refuses a manifest file that is missing, hand-edited, or not the one meta.json recorded, a
    checkpoint that cannot be found, and any difference between the manifest and its files.
    """
    dump_dir = Path(dump_dir)
    prov = meta.get("checkpoint")
    if not prov:
        raise CheckpointMismatch(f"{dump_dir}/meta.json has no checkpoint provenance (a pre-G3 dump?)")
    mpath = dump_dir / prov.get("manifest_file", MANIFEST_FILE)
    if not mpath.is_file():
        raise CheckpointMismatch(f"{mpath} is missing")
    manifest = json.loads(mpath.read_text())
    check_manifest_integrity(manifest)
    if manifest["digest"] != prov.get("manifest_digest"):
        raise CheckpointMismatch(f"{mpath} digest {manifest['digest']} != meta.json's {prov.get('manifest_digest')}")
    verify_manifest(manifest, locate_checkpoint(prov, checkpoint_dir))
    return manifest


def verify_dump_checkpoint(dump_dir, checkpoint_dir=None) -> dict:
    """Explicit verification of a dump directory; a dump without provenance is refused here."""
    dump_dir = Path(dump_dir)
    return verify_dump_meta(dump_dir, json.loads((dump_dir / "meta.json").read_text()), checkpoint_dir)
