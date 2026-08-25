# Handoff — cache-economics follow-on (WP1 composition, WP2 calibration curve, WP3 length)

2026-08-24 (UTC 2026-08-25T02:10Z). **Newest commit this brief describes: `8e286b2`**
(`docs: cache-economics follow-on spec (WP1-WP4)`), `main` in sync with `origin/main`.

**Everything below that is tagged `built` is UNCOMMITTED working tree.** `8e286b2` contains
only the spec. The entire follow-on build — nine changed tracked files, twenty-plus new ones,
and 97 new tests — sits uncommitted, so measure drift from the working tree, not from that SHA.
Closing that gap is the first item under Open / next and is the single largest risk here: this
work is one `git checkout` from gone.

Paper: Heo et al. (NVIDIA), arXiv [2608.03893](https://arxiv.org/abs/2608.03893). No code
release. Spec: `docs/superpowers/specs/2026-08-24-cache-economics-followon-spec.md`.
Plan: `docs/superpowers/plans/2026-08-24-cache-economics-followon.md` (local-only, gitignored).

## Current state

**`built` — the `kvt` extensions and five scripts, 139 tests green** (was 42 at `0487bd6`).
Ten tasks under subagent-driven development, each independently reviewed; then a final
whole-branch review that returned **NOT ready** with two blockers, both since fixed and
mutation-verified.
`re-verify:` `.venv/Scripts/python.exe -m pytest -q` — expect `139 passed`.

**`built` — Run 5, rope-space vs content-space mapper at the calibration length.** The two
variants differ in RoPE handling and nothing else; the H-G3 control (V carries no RoPE, so V
must be identical between them) holds bit-exactly.
`re-verify:`
`.venv/Scripts/python.exe -c "import json; c=json.load(open('results/mapper/qwen3-0.6b-to-1.7b/r2.json'))['k']['1']; r=json.load(open('results/mapper/qwen3-0.6b-to-1.7b/rope/r2.json'))['k']['1']; print(round(c['K_r2_heldout_layer_mean'],4), round(r['K_r2_heldout_layer_mean'],4), c['V_r2_heldout_layer_mean']==r['V_r2_heldout_layer_mean'])"`
— expect `0.6814 0.7065 True`.

**`built` — Run 6, paired McNemar tests on data that already existed.** No new inference. It
**resolved** a question the ledger had recorded as open (k=1 vs k=4) and correctly **declined**
to upgrade another (k=8 vs identity, which fails Bonferroni for its family of six).
`re-verify:`
`.venv/Scripts/python.exe -c "from pathlib import Path; from scripts.summarize_hellaswag import load_and_validate_records; from kvt.stats import correctness_by_idx, mcnemar_exact; r=load_and_validate_records(Path('results/hellaswag/qwen3-0.6b-to-1.7b'),500); c={k:correctness_by_idx(v) for k,v in r.items()}; m=mcnemar_exact(c['mapped-k1'],c['mapped-k4']); print('%.3e'%m['p'], m['b'], m['c'])"`
— expect `1.264e-03 41 16`.

**`built`, INTERRUPTED — Run 7, prefix-conditioned perplexity.** Killed mid-run. P=512 finished
all five conditions; P=1024 has only the two native conditions; **P=2048 never started**. The
completed slice does enforce the injection gate on a second instrument for the first time.
`re-verify:` `.venv/Scripts/python.exe scripts/summarize_perplexity.py --pair qwen3-0.6b-to-1.7b`
— expect P=512 `native` and `native-injected` both `16.1154`, `content-k1` 36.10%, `rope-k1`
37.96%, `identity` 440.59%.

**`built`, NOT RUN — WP1 composition.** `compose()` plus a gate proving the closed form equals
applying two mappers in sequence; `providers(extra=)`; `--extra` and `--compare` CLIs. The
accuracy leg cannot run here: the Qwen3-4B checkpoint is not downloaded (~8 GB), and its
HellaSwag leg is ~6 CPU-hours.
`re-verify:` `.venv/Scripts/python.exe -m pytest tests/test_compose.py -q` — expect `11 passed`;
and `ls ~/.cache/huggingface/hub | grep 4B` returns nothing, which is why it has not run.

**`built`, NOT RUN — WP2 calibration curve.** Fitting, curve-point emission and the summarizer
exist. Blocked on data: the 420-sequence dump completed `source` (28/28 layers, 12 GB, valid)
and lost **all** of `target` when killed at 85%.
`re-verify:`
`.venv/Scripts/python.exe -c "from pathlib import Path; d=Path('data/kv/qwen3-0.6b-to-1.7b-n420'); [print(w, len(list((d/w).glob('layer*.npz')))) for w in ('source','target')]"`
— expect `source 28` and `target 0`.

**`in-progress` — a Workflow drafting the README "Research directions" section and a commit
plan** (run id `wf_b4381cd5-bad`). Recon returned; drafting and adversarial verification were
still running when this brief was written. It is **not** load-bearing: its output is text for a
human to place, and nothing else depends on it. If its results are gone, re-run or write the
section by hand from `docs/ledger.md`.

**`planned` — WP4, the cross-pair transfer predictor.** Needs ≥6 directed pairs × 3 k. Not
designed beyond the spec's sketch. Do not start before WP1–WP3 have results.

## Locked decisions

- **Held-out is by SEQUENCE, never by token.** Adjacent tokens are correlated; a token split
  leaks and inflates exactly the held-out number this repo exists to measure.
- **`KVDump.get()` returns its cached tensor by reference; callers must never mutate it.**
  Cloning was rejected on measured grounds: 52.4 MB per tensor × 784 `get()` calls ≈ 41 GB of
  copying per run. (The docstring previously said 210 MB / 164 GB — that was the *un-strided*
  size, 4× too high, corrected 2026-08-24. The decision stands; only its evidence was wrong.)
- **Cache is LRU-bounded at `CACHE_LIMIT = 24` inside `fit_mapper`/`mapper_r2`, and the default
  stays unlimited.** Unbounded, the cache reaches 49 GB at 420 sequences on a 32 GB box and the
  fit dies at every k. The default is unlimited because every committed number was produced
  that way. Eviction is value-neutral: an evicted tensor reloads bit-identically.
- **WP2 reuses the 50-sequence probe's `r2_*_train.npy` arrays for layer selection; it does not
  re-probe.** Those cover sequences [0, 40), which sit inside every training prefix and are
  disjoint from the held-out range [400, 420). Re-probing would be *worse*: `probe_r2` splits by
  fraction, so on a 420-sequence dump its held-out range is [336, 420) and overlaps. Note this
  safety is a property of which file the script happens to load, not of anything that enforces
  it.
- **Every WP2 curve point is recomputed from ONE dump, including n=50.** Dumps of identical
  tokens differ by up to one fp16 ULP across torch thread counts, so splicing Run 2's committed
  0.6814 in as the curve's first point could make a 1e-04-scale difference between points a
  thread-count artifact rather than an effect of n.
- **P = 2048 is the minimum meaningful length test.** Calibration sequences are 1024 tokens, so
  at P ≤ 1024 the mapper is applied only at positions it was fitted on. The pre-registration
  originally required {512, 1024} and this was corrected *before* any run.
- **There is no identity control for unequal-depth pairs.** Injecting a 28-layer cache into a
  36-layer model is undefined and `providers()` correctly raises. Matched-KV does **not** imply
  an identity null exists — it exists only at equal depth. `source` alone is the floor for WP1.
- **Multiple comparisons are corrected.** The Run 6 family of six uses Bonferroni (8.3e-03),
  not 0.05. This is why k=8 vs identity (p=0.047) stayed open.
- **The agent never writes git history.** Commands are emitted for the operator. This is why
  nothing here is committed and why review packages were snapshot diffs, not commit ranges.

## Reuse map

- `kvt/data.py` — `seq_range_mask(lo, hi)` for nested prefixes against a fixed held-out set
  (`split(frac)` cannot express this, its held-out moves with `n_seqs`); `set_cache_limit(n)` /
  `clear_cache()`.
- `kvt/mapper.py` — `compose(a, b)` (closed form; `k` derived from `selected.shape[1]` because
  it feeds `formula_params`, a parameter-count claim); `space` field selecting content vs rope;
  `CACHE_LIMIT`.
- `kvt/stats.py` — `correctness_by_idx` / `mcnemar_exact`. **Use this, not Wilson intervals, for
  condition-vs-condition.** Wilson is unpaired and cannot resolve differences these experiments
  produce on identical example sets.
- `kvt/perplexity.py` — `fixed_continuation_windows` + `slice_for_prefix`. **Always slice; never
  re-chunk per prefix length**, or each P scores different text.
- `kvt/hellaswag.py` — `providers(source, target, mappers, extra=None)`; `extra` maps a full
  condition name to `(model, Mapper)` and may prefill from a different model.
- `scripts/summarize_perplexity.py`, `scripts/summarize_curve.py`,
  `scripts/summarize_hellaswag.py` — the fail-closed summarizers. Call them; never glob JSONL
  and average.
- `scripts/compose_mapper.py` — saves the composed mapper only after its H-C3 gate passes.
- `.git/sdd/fo-task-*-brief.md`, `fo-*-report.md`, `fo-review-*.md` — per-task requirements,
  evidence, and every review, including the final one that returned NOT ready.
- `docs/learnings/` — 13 entries. Four from this session are about the *verification layer*
  rather than the code, and are the ones most likely to save the next session time.

## Invariants

- **`native-injected` must equal `native`** in every evaluation run, on every target model and
  at every prefix length. Both harnesses now enforce it; `summarize_perplexity.py` refuses if
  the condition is absent rather than skipping the gate.
- **Feature layout must agree between `build_features` (fit) and `apply_mapper.feats`
  (inference).** Disagreement produces no error and no failing test except the k>1 gate. Do not
  weaken `test_self_mapper_is_a_noop_at_k_gt_1_pinning_feature_order` back to k=1.
- **`compose()` must agree with applying two mappers in sequence** to max abs 2e-3. They are
  independent computations; disagreement is a bug, and a wrong `compose` would produce a poor
  R² that reads as *confirmation* of pre-registered H-C2.
- **`fit_mapper.py` refuses the untagged output path** when `--space`, `--n-train` or
  `--dump-root` is set, and `dump_kv.py` refuses to overwrite a dump with a different
  `n_seqs`. Both guard artifacts backing published numbers. Do not remove either.
- **`kvt.pairs._rope_theta` must raise, never default.** A consistently-wrong theta cancels in a
  strip/re-apply round trip, so no test would catch it.
- **Every reported number is recomputed from a raw artifact by a summarize script**, and
  deltas are computed from raw values, not from rounded table entries. Violated once this
  session and corrected (+0.0251 → +0.0252).
- **Status tags are mandatory**; console output stays ASCII (cp1252).
- **Before trusting any check, ask what it would print if the thing it guards were broken.**
  Three guards this session could not fail by construction, and two verification commands
  reported success for work that had not happened. See
  `docs/learnings/2026-08-24-guards-inherit-the-assumptions-of-what-they-guard.md` and
  `docs/learnings/2026-08-24-verifications-that-cannot-fail.md`.

## Open / next

**First: commit.** Nothing from this build is in history. Grouped commands were emitted to the
operator; if lost, regenerate from `git status --short` with one commit per concern, keeping the
two overwrite-guard fixes and the reporting fail-open fix separate from features, and the new
results (`results/mapper/.../rope/`, `results/perplexity/`) separate from code.

**Then, in order of value:**

1. **P = 2048 perplexity — the only run that answers a question nobody has answered.** Everything
   else pending is a stronger version of something already known. Command:
   `scripts/eval_perplexity.py --pair qwen3-0.6b-to-1.7b --prefix-lens 2048 --n-windows 12
   --content-mapper mappers/qwen3-0.6b-to-1.7b/k1 --rope-mapper mappers/qwen3-0.6b-to-1.7b/rope/k1`.
   Expect ~1 h on CPU. Note Run 5 put rope-space **ahead** at the calibration length, so H-G1 is
   now a *crossover* question, and at 12 windows the P=512 difference was not significant
   (9/12, sign test p=0.146) — plan for more windows if the effect is small.
2. **Re-run the 420-sequence target dump** (~2 h; source is intact). Use
   `--out data/kv/qwen3-0.6b-to-1.7b-n420/target`. Then WP2's curve is a sequence of
   `fit_mapper.py --n-train N --holdout 400 420 --dump-root ... --tag nN` runs.
3. **WP1** needs the 4B checkpoint downloaded first. Its R² leg is affordable; its HellaSwag leg
   is ~6 CPU-hours.

**Blockers.** CPU-only, no GPU (`torch.cuda.is_available()` is False) — a GPU would compress
WP1–WP3 from roughly a day to under an hour, except the float64 ridge solve, which is at parity
on consumer cards. Disk: `data/` holds ~15 GB, gitignored and regenerable.

**Two caveats a newcomer must not lose:**
- `docs/handoff/2026-08-23-kv-transfer-replication.md:161` says k=1 vs k=4 is unresolved. **Run 6
  resolved it.** That brief is immutable, so the correction lives here and in the ledger's
  `[SUPERSEDED by Run 6]` markers.
- Nothing in this build tested length generalization. Run 5 compares the two mapper variants
  *at* the calibration length and Run 7 only reached P=512, below it. The paper's actual claim
  for content-space mapping remains untested by anyone, including the paper.
