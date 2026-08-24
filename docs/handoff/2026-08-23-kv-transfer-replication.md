# Handoff — CPU replication of Heo et al. 2026 cross-model KV cache transfer

2026-08-23 (UTC 2026-08-24T03:50Z). **Newest commit this brief describes: NONE — the repository
has zero commits.** `git init -b main` was run but no history exists, because the operator
reserves history-writing. Measure drift from the working tree as described below, not from a
SHA. Closing this gap is the first item under Open / next, and it is the one thing the session's
adversarial pass had to mark UNEVALUABLE.

Paper: Heo et al. (NVIDIA), arXiv [2608.03893](https://arxiv.org/abs/2608.03893), 4 Aug 2026.
No code release. Everything here is reconstructed from the equations.

## Current state

**Built — the `kvt` package and its scripts, 42 tests green.** Nine modules
(`pairs`, `models`, `rope`, `cache`, `data`, `ridge`, `mapper`, `hellaswag`, plus `__init__`)
and seven scripts. Built task-by-task under subagent-driven-development; every task passed an
independent spec+quality review, three needed a fix pass.
`re-verify:` `.venv/Scripts/python.exe -m pytest -q` — expect `42 passed`.

**Built — all four replication steps have RUN.** Results live in `docs/ledger.md` as Runs 1-4
with per-run wall-clock. Dumps 3m25s + 8m36s, probe 22m39s, mapper fit 13m04s, HellaSwag n=50
16m13s, HellaSwag n=500 170m57s.
`re-verify:` `cat results/hellaswag/qwen3-0.6b-to-1.7b/summary.md` — expect the seven-condition
table with native 0.598 and mapped-k1 0.548.

**Built — the headline result.** In-sample K R^2 rises with k (0.7783 / 0.8816 / 0.9607 at
k=1/4/8) while held-out R^2 collapses (0.6814 / 0.5907 / 0.0984) and HellaSwag acc_norm falls
(0.548 / 0.498 / 0.312). Selecting k on the metric the paper reports picks the worst of the
three. Cause is interpolation: p/n = 0.800 at k=8.
`re-verify:` `.venv/Scripts/python.exe -c "import json; d=json.load(open('results/mapper/qwen3-0.6b-to-1.7b/r2.json'))['k']; print([(k, round(d[k]['K_r2_train_layer_mean'],4), round(d[k]['K_r2_heldout_layer_mean'],4)) for k in ('1','4','8')])"`

**Built — adversarial verification, 3/3 claims survived.** An independent skeptic recomputed
every number from raw artifacts (never from `docs/ledger.md`, `summary.json`, or `r2.json`) and
ran two prescribed attacks. See `docs/ledger.md` § "Adversarial verification".
`re-verify:` read `docs/ledger.md` § Adversarial verification; each claim there names the
command that reproduces it.

**Built — the two correctness gates that make everything else interpretable.**
(a) cache injection is exact: prefill, extract to tensors, rebuild, forward the suffix
reproduces a native forward to 2.4e-7 on a tiny model and 500/500 argmax on the real 1.7B;
(b) `apply_rope` is bitwise identical to HF's `apply_rotary_pos_emb`.
`re-verify:` `.venv/Scripts/python.exe -m pytest tests/test_cache.py tests/test_rope.py -q`

**Not started — Task 10, attention-output cosine** (`[STRETCH]`). The paper's proposed
replacement for R^2 as a retention predictor. A full brief with code is already written at
`.git/sdd/task-10-brief.md`.

**Not started — the Qwen3-1.7B -> 4B pair** (`[STRETCH]`). Already in the registry as
`qwen3-1.7b-to-4b`; differs in layer count (28 -> 36) like the paper's own pairs, so it would
exercise cross-depth selection that the 28->28 pair cannot.

**Deliberately not attempted.** Prefill-latency measurement (meaningless on CPU), the MLP
mapper, and GSM8K / ARC-C / MMLU. HellaSwag only, per the spec.

## Locked decisions

- **Pair is Qwen3-0.6B -> Qwen3-1.7B.** Both 28 layers, `n_kv=8`, `d_h=128`, `rope_theta=1e6`,
  shared tokenizer — matched-KV, and small enough to run on a CPU-only 32 GB box. Reason still
  holds unless the machine changes.
- **Held-out split is by SEQUENCE, never by token** (last `ceil(0.2*n)` sequences). Adjacent
  tokens are correlated, so a token split leaks and inflates held-out scores — which would
  destroy the one measurement this whole repo exists to make.
- **The agent never writes git history.** Operator rule (`~/.claude/rules/git.md`). Commands are
  emitted for the human. This is why the repo has no commits and why review packages in this
  session were snapshot diffs rather than commit ranges.
- **`KVDump.get()` returns its cached tensor BY REFERENCE and callers must not mutate it.** A
  reviewer proposed cloning; it was rejected on measured grounds — tensors are ~210 MB at real
  scale and `probe_r2` calls `get()` 784 times, so cloning would copy ~164 GB per run. The
  invariant is instead pinned by a test that runs the real consumer and asserts the cache is
  unchanged. Reason holds as long as dumps stay this large.
- **transformers-5 API deltas override the plan's example code**, recorded in
  `.git/sdd/env-notes.md`. Chiefly: read theta via `kvt.pairs.kv_shape`, never `config.rope_theta`.
- **`*.egg-info/` was added to `.gitignore`** even though the plan specified that file verbatim —
  additive only, and the plan predated the editable install.
- **Report retention BOTH raw and floor-normalized.** Raw retention on 4-way multiple choice
  cannot fall below ~25%, so the floor is exactly what hides a catastrophic result.

## Reuse map

- `kvt/pairs.py` — `PAIRS` registry, `kv_shape()`, `check_matched_kv()`. **Always** get
  `rope_theta` through `kv_shape()`; it raises rather than defaulting.
- `kvt/rope.py` — `apply_rope` / `strip_rope` (exact inverses, HF-pinned) and the
  `_tokens_first` variants that take real per-token positions, so stride-subsampled dumps work.
- `kvt/cache.py` — `get_layer_kv` / `cache_to_arrays` / `build_cache` / `forward_with_cache`.
  This is the only place transformers-version drift is allowed to land; `get_layer_kv` carries a
  dual path and the `key_cache` branch is intentional dead code for older versions.
- `kvt/data.py` — `KVDump` (lazy, `.get(kind, layer)`, `.split(frac)`) and `dump_kv`. Read the
  no-mutation contract in `get`'s docstring before using it.
- `kvt/ridge.py` — `fit_ridge` (chunked float64 Gram, centered, bias), `r2_score` (scores against
  the mean of whichever set is passed), `probe_r2`.
- `kvt/mapper.py` — `Mapper.formula_params()` reproduces the paper's Appendix D counts exactly
  (three Table 12 rows checked); `select_top_k`, `fit_mapper`, `apply_mapper`.
- `kvt/hellaswag.py` — `providers()` gives every condition including the two null controls;
  `summarize_records` does floor normalization and Wilson intervals.
- `scripts/summarize_hellaswag.py` — `load_and_validate_records()` is importable and is the
  fail-closed gate; call it rather than globbing JSONL yourself.
- **Test fixtures**: `tests/conftest.py` builds tiny randomly-initialised Qwen3 models
  (`tiny_src` 2 layers, `tiny_tgt` 3, `tiny_src3` 3 for identity controls). The whole suite runs
  offline in under 4 seconds — keep it that way; only the real runs touch real weights.
- `.git/sdd/env-notes.md` — the verified transformers-5 deltas. Read before writing new code.
- `.git/sdd/task-*-brief.md`, `task-*-report.md` — per-task requirements and evidence, including
  the unstarted Task 10.

## Invariants

- **`tests/test_cache.py::test_injecting_own_cache_reproduces_native_logits` must pass.** If the
  cache round trip is not exact, a broken harness is indistinguishable from a mapper that does
  not work, and every downstream number becomes uninterpretable.
- **`native-injected` must equal `native`** in any evaluation run. It is the same gate at real
  scale. If it diverges, stop — do not interpret the other conditions.
- **Feature layout must agree between `build_features` (fit) and `apply_mapper.feats`
  (inference)**: layer-major, then head, then `d_h`. Disagreement produces no error and no
  failing test except the k>1 gate — it just silently returns garbage. Guarded by
  `test_self_mapper_is_a_noop_at_k_gt_1_pinning_feature_order`; do not weaken it back to k=1.
- **`kvt.pairs._rope_theta` must raise, never default.** A consistently-wrong theta cancels in a
  strip/re-apply round trip, so no test would catch it.
- **The summarizer must refuse rather than emit NaN.** Any condition with zero records, a
  mismatched `idx` set, a duplicate `idx`, or a count mismatch is a refusal. It already fired on
  a real case during this session's first run.
- **Every reported number is recomputed from a raw artifact by a summarize script.** Nothing in
  `docs/ledger.md` may be restated from stdout or from another summary.
- **Status tags are mandatory** on every result section: `[VALIDATED]` / `[BASELINE]` /
  `[STRETCH]` / `[FUTURE]` / `[SUPERSEDED]`.
- **Console output stays ASCII** (this box's console is cp1252) and scripts stay POSIX-portable.
- **Never mutate a file another agent currently holds.** Violated once this session — a
  mutation test on `kvt/mapper.py` collided with a fix agent verifying the same file. The agent
  correctly reported the phantom edit instead of ignoring it, and had to redo its checks.

## Open / next

**First thing: commit.** The repo has zero history, which is the sole UNEVALUABLE in the
adversarial pass — provenance between the n=50 and n=500 runs cannot be established from the
files alone. The grouped commit commands were emitted to the operator at the end of the build
session (six conventional commits plus `git remote add` and `git push -u`). Nothing else should
be built on top of an uncommitted tree.

**Then, in order of value:**

1. **Length generalization is the real gap.** Content-space RoPE mapping is claimed to buy
   reuse across context lengths, and *nothing here tests it* — every evaluation is
   short-context. This is also why the H1b refutation says nothing for or against the paper.
   Testing it needs evaluation at several context lengths with a fit done at one.
2. **Task 10, attention-output cosine.** Brief ready at `.git/sdd/task-10-brief.md`. With one
   pair there is no correlation to compute; the deliverable is whether cosine orders the
   conditions the way accuracy does while R^2 does not.
3. **Re-run with nested subsamples.** `rng.choice(N, size=n)` is not nested across `n`, so this
   session's smoke and full runs share only 9 of 50 examples. Draw the largest set once and take
   prefixes.
4. **A 200-sequence calibration run** would move k=8 from p/n 0.800 to 0.160 and should partly
   restore it — a direct test that the collapse is calibration size and not the method. Budget
   ~40 min of dumping plus a longer mapper fit; the 8192-dim solve is the bottleneck.

**Blockers.** None technical. The machine is CPU-only with no NVIDIA GPU, so the n=500 eval is a
~3 hour job and anything larger should be planned around that. Disk: `data/` and `mappers/`
hold 5.7 GB, both gitignored and regenerable in about 25 minutes.

**Two live caveats a newcomer must not lose:**
- mapped-k8 (0.312) sits nominally *below* the identity null (0.362), which would mean the k=8
  mapper is worse than injecting the raw unmapped cache — but the intervals overlap, so it is an
  open question, **not** a finding.
- k=1 vs k=4 is unresolved at n=500 (intervals overlap). The statistically defensible claim is
  the k=1 vs k=8 endpoint gap only, and the "rank correlation ±1.0 over three points" phrasing
  was dropped as a trivial restatement of monotonicity.
