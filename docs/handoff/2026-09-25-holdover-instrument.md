# Handoff — holdover-instrument: Pair revision/local_path, device(), --dtype, CI + --probe

2026-09-25 (UTC 2026-09-25T04:25Z). **Newest commit this brief describes: `47bdd39`**
(`feat(models): device() prefers cuda, mps, cpu; KVT_DEVICE override`), branch
`holdover-instrument`, equal to `origin/holdover-instrument`. **The working tree is NOT clean:**
six files carrying tasks 1 and 3 of the four-task spec are uncommitted (see Current state).
Session: Claude Code `a0bc4703-7c59-4b7f-bcf2-e44daa6e7ffa`.

Spec executed: the operator's four-task prompt (Pair revision + local path; device() with
Apple silicon and `KVT_DEVICE`; `--dtype`; CI + console entry points + `--probe`), branched from
`063f402`. Downstream consumer is the lag-ladder repo, which re-pins by a ledger entry after the
operator merges. Nothing under `data/`, `results/`, `mappers/`, `docs/ledger.md` was written.

## Current state

**`built` — all four tasks, as a working tree.** The tree at HEAD plus the uncommitted six
files is byte-identical (modulo CRLF) to the stage the build finished at, and the whole suite is
green there: 165 passed (was 153 at `063f402`).
re-verify: `.venv/Scripts/python -m pytest -q 2>&1 | tail -1` — expect `165 passed`.
re-verify: `git diff --stat -- kvt scripts tests | tail -1` — expect `6 files changed, 98 insertions(+), 6 deletions(-)`
(scoped to code paths: an unscoped diff is dirtied by this brief's own index rows)
(the uncommitted tasks 1 + 3: `kvt/pairs.py kvt/models.py kvt/data.py tests/test_pairs.py
tests/test_data.py tests/test_models.py`). If this prints nothing, the operator has committed
them since; re-anchor on the new HEAD and skip the first Open item.

**`built` — pushed commits `e54c102` (task 4 files) and `47bdd39` (task 2 files), and BOTH ARE
BROKEN TREES.** The operator ran the commit block out of order: task 4 first, then task 2,
without tasks 1 and 3. `e54c102` committed the finished `scripts/dump_kv.py` (which imports
`DTYPES` from `kvt.models` and calls `Pair.with_revision`) against a `kvt/` that has neither.
At `47bdd39`: `import scripts.dump_kv` raises `ImportError: cannot import name 'DTYPES'`, and
the suite is `1 failed, 156 passed` (the failure is the `--probe` test). The per-task
bisectability the four-commit plan was for is therefore already lost on the pushed branch.
re-verify: `git show 47bdd39:scripts/dump_kv.py | grep -c DTYPES` — expect `2`;
`git show 47bdd39:kvt/models.py | grep -c "DTYPES ="` — expect `0`.
re-verify (pushed): `git rev-parse --short origin/holdover-instrument` — expect `47bdd39`
(read-only against the local ref; `git fetch` first if you want the remote's word).

**`built` — the console entry points.** `kvt-dump --help` and `kvt-score-positions --help`
exit 0 from `/tmp` with `PYTHONPATH` unset. This needed a reinstall (`pip install -e .`),
which the operator's own venv must repeat if it predates the `pyproject.toml` change.
re-verify: `(cd /tmp && env -u PYTHONPATH ~/dev/kv-transfer-replication/.venv/Scripts/kvt-dump.exe --help | head -1)` — expect `usage: kvt-dump [-h] --pair`.

**`built` — `--help` works at all.** At `063f402` `python scripts/dump_kv.py --help` crashed
(`ValueError: unsupported format character ','`) because the `--out` help text contains a bare
`20%`. Escaped to `20%%` inside the task-1 change to that file. This is the one edit outside
the spec's letter; it was required by two of the spec's done lines.
re-verify: `git show 063f402:scripts/dump_kv.py | grep -c "20%,"` — expect `1`;
`grep -c "20%%," scripts/dump_kv.py` — expect `1`.

**`built` — per-task patches, session-scoped.** `task1.patch` … `task4.patch` and the stage
snapshots `s0/`…`s4/` sit in this session's scratchpad
(`%LOCALAPPDATA%\Temp\claude\C--Users-hossa-dev\a0bc4703-7c59-4b7f-bcf2-e44daa6e7ffa\scratchpad\stages`).
They were verified to apply in sequence to a clean index at `063f402` and each intermediate
stage was green (159 / 162 / 164 / 165). They are NOT the right tool any more: the branch's
history no longer starts from `063f402`, so `task1.patch` and `task3.patch` would conflict on
`scripts/dump_kv.py`. Treat the folder as evidence, not as a plan; it may already be gone.

**`planned` — CI has never run.** `.github/workflows/ci.yml` parses with PyYAML and is pushed
in `e54c102`, but no Actions run was observed from this machine. Its first run at `47bdd39`
would fail for the ImportError above; it should pass once tasks 1 + 3 land.

**`planned` (operator's, out of scope)** — merge, and the lag-ladder ledger entry that re-pins
to the merged SHA.

## Locked decisions

- **`probe` is written into meta.json by the script after `dump_kv` returns, not by `dump_kv`.**
  Reason: task 3's done line pins the DEFAULT key set to the old nine keys plus exactly
  `revision`, `local_path`, `load_dtype`, and task 4's touch list excludes `kvt/data.py`.
  `tests/test_data.py::test_dump_records_load_dtype_and_default_key_set_is_old_plus_three`
  is the gate.
- **`local_path` wins over `revision` and the revision is NOT forwarded with it.** A local
  checkout carries its own revision; forwarding a Hub revision to `from_pretrained(local_dir)`
  is at best ignored. `Pair.resolve` encodes this; `tests/test_pairs.py` pins it.
- **`revision` is forwarded to `from_pretrained` only when not None.** So the default call is
  byte-for-byte what it was, which is what "existing behaviour unchanged" required.
- **`KVT_DEVICE` beats availability; unknown values raise naming `cuda, mps, cpu`.** Spec item.
- **K/V stay float16 on disk regardless of `--dtype`.** Spec item; the save cast was not touched
  and `test_data.py` asserts `layer00.npz["K"].dtype == float16` after a bfloat16 load.
- **Pushed history is not rewritten without an explicit ask** (global `git.md`). The broken
  intermediate commits are therefore repaired by a forward commit, not by reordering — unless
  the operator says otherwise (see Open / next for the trade).
- **`docs/` was not edited by the build.** This brief and its learnings entries are the only
  additions, written by `/rigor:handoff`, untracked until the operator commits them.

## Reuse map

- `kvt/models.py`: `DEVICES`, `DTYPES` (name → torch dtype), `device()`, `load_model(..., revision=, dtype=)`,
  `load_tokenizer(..., revision=)`.
- `kvt/pairs.py`: `Pair.resolve(which) -> (id_or_path, revision)`, `with_revision`, `with_local_path`
  (both `dataclasses.replace`, registry entries never mutate).
- `scripts/dump_kv.py`: `probe_forward(model, seq, load_dtype) -> dict` (one 64-token forward;
  peak bytes by device type, `-1` on cpu); `PROBE_TOKENS`.
- **Test pattern for running a script end-to-end with no download:** save the synthetic model
  (`tiny_tgt.save_pretrained(dir)`), pass `--local-path dir` so BOTH sides of the pair resolve to
  it, `--tokens` to a saved `.npy`, `--out` to tmp, and `KVT_DEVICE=cpu`. See
  `tests/test_scripts.py::test_dump_kv_probe_records_device_dtype_attn_and_peak`.
- **Staging from patches on this CRLF checkout:** `git apply --cached <patch>` applies to the
  LF-normalized index and never touches the working tree, so line endings cannot break it.
- **Testing a historical tree without committing:** `git worktree add --detach <dir> <sha>`, then
  `PYTHONPATH=<dir> .venv/Scripts/python -m pytest -q -p no:cacheprovider` from inside it. The
  editable install does not shadow this (see learnings, editable-finder entry).

## Invariants

- **meta.json default keys** = `model n_layers n_kv d_h rope_theta stride seq_len n_seqs rope`
  + `revision local_path load_dtype`, nothing else unless `--probe`. Archived dumps are read by
  `KVDump.load` via these names; renaming one silently orphans every dump under `data/kv`.
- **Existing tests unmodified.** `git diff 063f402 -- tests/ | grep -E "^-[^-]"` must stay empty.
  The spec forbids weakening an assertion to pass.
- **No bare `%` in any argparse help string** in `scripts/dump_kv.py`; `--help` is a done-line
  and a CI-visible surface. Argparse expands help with `%`-formatting.
- **Adding a top-level package or console script requires `pip install -e .` again.** The
  setuptools editable finder bakes its package map at install time.
- **No writes under `data/`, `results/`, `mappers/`, `docs/ledger.md`** from this branch. The
  untracked `results/mapper/qwen3-0.6b-to-1.7b/n420/` predates the branch and is not ours.
- **ASCII-only stdout** (Windows cp1252 console); files UTF-8.

## Open / next

1. **Land tasks 1 + 3 on top of `47bdd39` so the pushed branch is green.** Two honest options;
   the second is the default under `git.md`:
   - *Rewrite (needs the operator's explicit say-so, and a force-push):* reset the branch to
     `063f402`, re-apply the four scratchpad patches in order with `git apply --cached`, commit
     each. Cheap only while nothing downstream has pinned `e54c102`/`47bdd39`; lag-ladder pins by
     ledger entry, so check that ledger first.
   - *Forward commit (default):* the six uncommitted files are tasks 1 and 3 interleaved in
     `kvt/models.py`, `kvt/data.py`, `tests/test_data.py`, `tests/test_models.py`; a clean split
     needs hunk editing (the `dump_kv` signature hunk carries both). One combined commit:
     ```bash
     cd ~/dev/kv-transfer-replication
     git add kvt/pairs.py kvt/models.py kvt/data.py tests/test_pairs.py tests/test_data.py tests/test_models.py
     git commit -m "feat(pairs,models): revision/local_path on Pair and --dtype; completes e54c102"
     .venv/Scripts/python -m pytest -q   # expect 165 passed
     git push
     ```
2. **Watch the first CI run** on that push; it is the only unverified done-line surface
   (`pytest -q` on ubuntu, CPU torch). Then merge (operator).
3. Commit this brief and the five learnings entries (`docs/handoff/2026-09-25-*.md`,
   `docs/learnings/2026-09-25-*.md`, and the two index rows).

Blocker: none. The scratchpad patches are convenience only; everything needed is in the
working tree and in HEAD.
