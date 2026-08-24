# Cache economics follow-on — implementation context

Written 2026-08-24 for the implementing session (Opus). It will be evaluated afterwards by
an independent controller session against the **Evaluation contract** at the end of this
document, so everything the contract names is a requirement, not a suggestion.

Read in this order before writing anything: this file; `README.md`; `docs/ledger.md`
(especially Hypotheses, Protocol decisions, Run 2, Run 4, Adversarial verification);
`docs/handoff/2026-08-23-kv-transfer-replication.md` (Locked decisions, Reuse map,
Invariants); `.git/sdd/env-notes.md` (transformers-5 deltas — untracked, exists only on
this machine); `docs/learnings/LEARNINGS.md`. Then plan with the writing-plans skill and
implement with subagent-driven development, as the first build did.

## 1. Where the repository stands (verified 2026-08-24)

- Eight commits on `main`, latest `0487bd6`, in sync with `origin/main`; `docs/handoff/` and
  `docs/learnings/` are committed. Whatever is untracked when you start is the operator's to commit.
- `pytest -q` → `42 passed` in about 4 s, offline, on tiny randomly-initialised Qwen3
  models from `tests/conftest.py`. Keep the suite offline and fast.
- One pair run end to end: `qwen3-0.6b-to-1.7b`. Artifacts on disk:
  `data/tokens/qwen3-0.6b-to-1.7b_n50_len1024_seed0.npy`, `data/kv/qwen3-0.6b-to-1.7b/{source,target}/`,
  `mappers/qwen3-0.6b-to-1.7b/k{1,4,8}/`, `results/{probe,mapper,hellaswag}/qwen3-0.6b-to-1.7b/`.
  `data/` and `mappers/` are gitignored (5.7 GB).
- The headline result you are building on: on this pair, in-sample K R² rises with k
  (0.7783 / 0.8816 / 0.9607) while held-out R² collapses (0.6814 / 0.5907 / 0.0984) and
  HellaSwag acc_norm falls (0.548 / 0.498 / 0.312); native 0.598, identity null 0.362.
  Cause: p/n = 0.800 at k = 8 with 10,240 calibration tokens. Every one of those numbers
  is recomputable from `results/` — recompute them, do not restate them.
- Machine: CPU-only, 16 cores, 32 GB RAM, Windows, cp1252 console. No GPU. Qwen3-4B in
  fp32 is ~16 GB and fits; Qwen3-8B does not fit in fp32 and is out of scope.

## 2. What is being asked

The slide deck "Cache economics" pitches this repo as the initial implementation of
cross-model KV transfer and proposes, as its research direction, *a pre-deployment
predictor for whether a pair will transfer*. Three follow-on directions were chosen; this
brief turns them into work packages in dependency and value order. WP1 is the one to
finish if only one finishes.

| WP | direction | what it decides | status on entry |
|---|---|---|---|
| WP1 | **Composition** — do linear mappers compose across a chain? | whether a family of n sizes needs n−1 mappers or n(n−1) | `[STRETCH]` designed here, not run |
| WP2 | **Calibration learning curve** — retention as a function of calibration tokens and k | whether the k = 8 collapse is calibration size or the method | `[STRETCH]` designed here, not run |
| WP3 | **Length generalization** — content-space vs RoPE-space mappers past the calibration length | whether the paper's central design choice (strip/re-apply RoPE) is motivated | `[STRETCH]` designed here, not run |
| WP4 | **Transfer predictor** — which calibration-only quantity predicts retention across pairs | the slide deck's own ask; needs WP1–WP2's pairs and Task 10 | `[FUTURE]`; do not start unless WP1–WP3 are recorded |

The paper (Heo et al., arXiv 2608.03893) reports none of these. WP2 is the direct test of
this repo's own headline; WP1 and WP3 test claims the paper makes implicitly (cascading is
a chain; content space is for length) without measuring.

## 3. Constraints that carry over unchanged

These are the first build's locked decisions and invariants. Violating any of them makes
the run uninterpretable, and the evaluator will check each.

- **Never write git history.** No `git commit`, `git push`, `git add`. Emit the commands
  for the operator at each checkpoint and keep working.
- **Held-out is by sequence, never by token.** Adjacent tokens are correlated.
- **`KVDump.get()` returns its cached tensor by reference; never mutate it.** Pinned by
  `tests/test_data.py`; clone if you must write.
- **Read `rope_theta` only through `kvt.pairs.kv_shape()`.** It raises rather than
  defaults. `config.rope_theta` does not exist in transformers 5.
- **`native-injected` must equal `native`** in every evaluation run, on every target model
  you introduce (4B included). If it diverges, stop and report; interpret nothing else.
- **Do not weaken `test_self_mapper_is_a_noop_at_k_gt_1_pinning_feature_order`.** It is the
  only thing that catches a fit/inference feature-order mismatch, which fails silently.
- **Summaries fail closed.** Extend `scripts/summarize_hellaswag.py::load_and_validate_records`
  or copy its pattern; never glob JSONL and average. Zero records, duplicate `idx`,
  mismatched `idx` sets, or a count short of `--expect-n` are refusals.
- **Every reported number is recomputed from a raw artifact by a summarize script.**
  Nothing in `docs/ledger.md` may be restated from stdout or from another summary.
- **Status tags on every result section:** `[VALIDATED]` / `[BASELINE]` / `[STRETCH]` /
  `[FUTURE]` / `[SUPERSEDED]`. `[VALIDATED]` only after an independent skeptic pass.
- **Pre-register before running.** Hypotheses and decision rules for each WP go into
  `docs/ledger.md` under a dated heading *before* the first run of that WP starts. The
  evaluator will compare the ledger's pre-registration text against file mtimes in
  `results/`.
- **Console output stays ASCII.** `print()` on this box is cp1252. Files are UTF-8.
- **Never mutate a file another agent holds.** If you run mutation tests, do it on files
  no subagent is currently editing or verifying.
- **Report retention raw and floor-normalized** (`(acc − chance)/(target − chance)`,
  chance 0.25 for HellaSwag) and Wilson 95% intervals. For *paired* comparisons between
  conditions scored on the same examples, use McNemar's exact test on the per-example
  correctness vectors — CI overlap at n = 500 (±4 pp) cannot resolve 3 pp differences,
  a paired test can. Say which test produced which p-value.

## 4. WP1 — Composition `[STRETCH]`

### Question

Let A = 0.6B→1.7B (exists, k = 1), B = 1.7B→4B, C = 0.6B→4B. Is B∘A as good as C?
If yes, a family with n sizes needs n−1 mappers along a chain, not n(n−1) directed pairs
of 4–12 GB each; and cascading (the slide deck's motivating pattern) is a chain whose
compounding error is measured, not assumed away.

### Pre-registered hypotheses (write these into the ledger verbatim, then run)

- **H-C1 (composition holds, the paper-favourable reading):** on held-out sequences,
  K R² of (B∘A)(x₀.₆ʙ) against y₄ʙ is within 0.05 of C's, and floor-normalized HellaSwag
  retention of `composed-BA-k1` is within 3 pp of `mapped-C-k1` with McNemar p > 0.05.
- **H-C2 (the skeptic's prior):** composition loses more than 0.05 held-out K R² and
  McNemar rejects at p < 0.05 with `mapped-C-k1` ahead. Linear maps fitted with ridge on
  different calibration targets do not commute with error; each hop adds its own residual.
- **H-C3 (gate):** operational composition — `apply_mapper(B, apply_mapper(A, x))` — agrees
  with a closed-form product mapper to max abs 1e-3 on fp32 tensors. Mind the block
  structure: each per-head weight consumes *all* source heads (`k·n_kv·d_h → d_h`, see
  `build_features` and `Mapper.formula_params`), so for 4B target layer l and B's selected
  1.7B layer s = sel_B[l, 0] (k = 1), first stack A's n_kv per-head maps for 1.7B layer s
  into one `[n_kv·d_h × n_kv·d_h]` block W_A(s) (with bias b_A(s) of length n_kv·d_h),
  then W_C[l,h] = W_A(s) · W_B[l,h] and b_C[l,h] = b_A(s) · W_B[l,h] + b_B[l,h] in the
  row-vector convention `y = x·W + b` that `kvt.ridge.predict` uses; at k > 1 the same
  block-wise over each of B's k selected layers, and the composed mapper's own selection
  is sel_A[sel_B[l, :], 0]. The RoPE re-apply after A and strip before B cancel in content
  space up to float error, which is what the 1e-3 tolerance is for. If the two paths
  disagree by more, one of them is wrong; find which before interpreting anything.

Either of H-C1 / H-C2 is a result. Decide by the rule, not by which is more pleasing.

### Design

1. Registry: add `qwen3-0.6b-to-4b` to `kvt.pairs.PAIRS`. `qwen3-1.7b-to-4b` is already
   there. Run `check_matched_kv` on the loaded configs for both — **do not trust this
   brief's claim that Qwen3-4B has `n_kv=8`, `d_h=128`, 36 layers**; assert it.
2. Dumps: Qwen3-4B on the **same 50 sequences** (`data/tokens/qwen3-0.6b-to-1.7b_n50_len1024_seed0.npy`
   — the tokenizer is shared, so reuse the file; assert that the 4B tokenizer's vocab
   matches before reusing). Same stride, same split (last 10 sequences held out). Assert
   `positions` and `seq_idx` are identical across all three dumps.
3. Probe + mappers for B and C at k = 1. k = 1 because it is the best k at this calibration
   size (Run 2/4); do k = 4 only if time allows and say so. Cross-depth (28 → 36 layers)
   selection is exercised here for the first time — check that `select_top_k` handles
   `L_s ≠ L_t` and that `Mapper.formula_params` still matches Appendix D for an unequal
   layer count.
4. Composition: implement `compose(A, B) -> Mapper` (closed form) and test it against the
   two-call path on tiny models (H-C3 at tiny scale, then again on real dumps).
5. Held-out R² on the 4B target for: C direct, B∘A closed-form, B∘A operational (the last
   two must agree per H-C3).
6. HellaSwag n = 500, seed 0, on the 4B target. Conditions, all scored on the same 500
   examples with the same subsample code path as Run 4: `native-4b`, `native-injected-4b`
   (gate), `source-0.6b` (0.6B alone), `identity` (raw 0.6B cache into 4B), `mapped-C-k1`,
   `composed-BA-k1`, and `mapped-B-k1` from a 1.7B prefill (the single-hop baseline from
   the middle model — it bounds what the chain could possibly achieve).
7. Summarize with `--expect-n 500`; McNemar `mapped-C-k1` vs `composed-BA-k1`.

### Budget (CPU, estimated from Run 1–4 wall-clocks; write the actuals in the ledger)

4B dump ~25 min; two probes (28×36 cells) ~1 h each; three mapper fits ~15 min each;
HellaSwag on 4B, 7 conditions, ~5–6 h. Roughly one CPU-day. Run the long evaluation in the
background and do not poll it — arm a Monitor or `run_in_background` and stop.

## 5. WP2 — Calibration learning curve `[STRETCH]`

### Question

Retention(n_calib, k) on the 0.6B→1.7B pair. Does k = 8 recover as p/n falls, and where
does k = 8 held-out R² cross k = 1? This is the direct test of the repo's headline claim
that the collapse is calibration size, not the method.

### Pre-registered hypotheses

- **H-L1:** held-out K R² at k = 8 rises monotonically in n over {50, 100, 200, 400}.
- **H-L2 (paper-favourable):** at n = 400 (p/n = 0.10), k = 8 held-out K R² ≥ k = 1
  held-out K R², and `mapped-k8` floor-normalized retention ≥ `mapped-k1` − 3 pp.
- **H-L3 (alternative):** k = 8 has not crossed k = 1 by n = 400. Then cross-layer
  selection costs accuracy on this pair even outside the interpolation regime, and the
  paper's k-selection result does not transfer to 0.6B→1.7B. A result, not a failure.
- **H-L4 (noise floor, re-check):** the pure-noise in-sample R² ≈ p/n at each (n, k);
  quote p/n beside every in-sample R² in the table.

### Design

1. **Fixed held-out set, nested training prefixes.** Draw 420 sequences with
   `prepare_tokens.py --n-seqs 420 --seed 0`. Assert that rows 0–49 equal the existing
   n = 50 token file byte-for-byte (`iter_fineweb_sequences` takes the first n after a
   seeded shuffle, so this should hold — if it does not, stop, record why, and do not
   proceed with a non-nested design). Held-out = sequences 400–419 (20 sequences, 5,120
   stride-4 tokens). Training sets = prefixes 0–49, 0–99, 0–199, 0–399. Pass explicit
   `train_mask` / held-out masks to `fit_mapper` and `mapper_r2`; do **not** use
   `KVDump.split`, whose held-out set moves with n.
2. Dump source and target once on all 420 sequences (~100 min); probe once on the n = 400
   training prefix for selection (selection is shared across n so that n is the only
   variable — say so in the ledger).
3. Fit k ∈ {1, 4, 8} at each of the four n. 12 mappers.
4. Held-out R² for all 12 against the fixed 20 sequences.
5. HellaSwag n = 500 on the 1.7B target for the mapped conditions only (`native` and
   `native-injected` from Run 4 are reusable **only if** the example subsample is
   byte-identical — assert the `idx` set matches Run 4's JSONL; otherwise re-score them).
   Do n ∈ {100, 200} first (required), n = 400 second (`[STRETCH]` within the WP): 3
   mapped conditions × ~25 min each per n.
6. Also re-run the existing n = 50 configuration under the new fixed held-out set so the
   curve has a comparable first point; the original Run 2 numbers (held-out = seqs 40–49)
   stay in the ledger as `[SUPERSEDED]` for the curve, not deleted.

### Budget

Dumps ~100 min, probe ~1.5 h, 12 fits ~2 h (the 8192-dim solve dominates and scales with
n only in the Gram accumulation), HellaSwag ~75 min per n. About one and a half CPU-days.

## 6. WP3 — Length generalization `[STRETCH]`

### Question

The content-space mapper (strip RoPE → map → re-apply at true positions) is claimed to
generalize past the calibration length because W is position-free. The RoPE-space mapper
(fit on rotated K, apply without strip/re-apply) has W entangled with the positions it saw.
H1b showed no difference at short context. Is there one past 1,024 tokens?

### Pre-registered hypotheses

- **H-G1 (paper-favourable):** relative perplexity degradation vs native is roughly flat
  in prefix length P for the content-space mapper and grows with P beyond the calibration
  length for the RoPE-space mapper; at P = 4,096 the gap between them exceeds 5% relative
  perplexity.
- **H-G2 (alternative):** both are flat, or both degrade equally. Then strip/re-apply buys
  nothing on this pair at up to 4× calibration length and the serving path can drop it.
- **H-G3 (V control):** V has no RoPE, so V's held-out R² is identical between the two
  mapper variants by construction. If it is not, the two variants differ in something
  other than RoPE handling — a bug, stop.

### Design

1. Instrument: WikiText-2 prefix-conditioned perplexity (slide 4's auxiliary eval). For
   each document window of length P + 256: source prefills the first P tokens, the cache
   is mapped and injected, the target scores NLL of the next 256 tokens (the target sees
   token P as its first input, per protocol A1). Report mean NLL → perplexity, per
   condition, per P ∈ {512, 1024, 2048, 4096}. ≥ 40 windows per P, held out from
   calibration (WikiText-2 is not FineWeb-Edu, so disjointness is by construction — say
   so).
2. Conditions: `native` (target prefills P itself), `native-injected` (gate, must equal
   native at every P), `identity`, `content-k1` (the existing mapper, applied with
   `apply_mapper` at true positions), `rope-k1` (new: fitted on `K_rope` features directly
   with the same selection and λ, applied to rotated K with no strip/re-apply; V path
   identical to `content-k1`).
3. The RoPE-space fit must reuse `build_features` with `kind="K_rope"` — do not write a
   second feature builder; the feature-order invariant applies.
4. Memory: a 4,096-token prefill of Qwen3-1.7B in fp32 on CPU is fine; the 4B is not
   needed here.

### Budget

Small next to WP1/WP2: one extra mapper fit, and 5 conditions × 4 lengths × 40 windows of
one forward pass each, ~2–3 h.

## 7. WP4 — Transfer predictor `[FUTURE]`

Do not build until WP1–WP3 are recorded and the operator says go. What it needs from
them: the three 0.6B/1.7B/4B pairs in both directions (six directed pairs × k ∈ {1,4,8} =
18 points), each with held-out R², attention-output cosine (Task 10, brief at
`.git/sdd/task-10-brief.md`, unbuilt), the top-1 minus top-2 layer-selection margin, and
floor-normalized retention. The question is which calibration-only quantity predicts
retention *across pairs*, not just across k within one pair. The pre-registration:
in-sample R² anti-predicts (rank correlation ≤ 0), held-out R² predicts (≥ +0.5), cosine's
correlation is to be measured with no prior. Reverse-direction pairs (1.7B→0.6B etc.)
need only the registry entry and a dump; they are matched-KV by symmetry.

## 8. Ledger, learnings, handoff

- New runs continue the numbering: Run 5 onward, one heading per run, with pair, n,
  split, wall-clock, artifacts, the table, and the verdict against the pre-registered
  hypotheses by rule.
- Each refuted prediction, each gate that fired, each thing you had to correct goes in
  `docs/ledger.md` § "What fired / what is blocked" and, if it generalizes, as a new dated
  entry in `docs/learnings/` with the seven fields (`ts:`, `commit:`, `session:`,
  `status:`, `fact:`, `basis:`, `re-verify:`) and a row in `LEARNINGS.md`. Entries are
  immutable; supersede, never edit.
- Finish with a new dated brief in `docs/handoff/` and a row in `HANDOFF.md`. Do not edit
  the 2026-08-23 brief.
- Update `README.md`'s status table and the timeline's *record* section for anything that
  actually ran; leave the *proposal* section's plan as it was, marking gates as passed /
  failed / not reached.

## 9. Evaluation contract — what the controller will check afterwards

The evaluating session will not read your summary first. It will:

1. Run `pytest -q` and require every pre-existing test still present and green, plus new
   tests for `compose`, the fixed-held-out masks, the `rope-k1` feature path, and the
   extended summarizer refusals. It will mutation-test at least one new gate (e.g. swap
   the order of `compose`'s arguments; corrupt one `train_mask`) and require a failure.
2. Recompute every number in the new ledger sections from `results/` and `mappers/`
   artifacts — never from `summary.json`, `r2.json`, or the ledger — and require a match
   to the printed precision.
3. Check `native-injected == native` on every target model at every P from the raw
   per-example records, not the summary.
4. Check pre-registration order: the ledger's hypothesis text for each WP must predate
   that WP's first result artifact. Hypotheses added after a result are not
   pre-registered; the section will be re-tagged.
5. Check WP2's nesting claim by comparing the token files directly, and the held-out
   disjointness by recomputing masks from `seq_idx`.
6. Check H-C3 (closed-form vs operational composition) by re-running the comparison.
7. Check that the RoPE-space and content-space conditions differ in code path only at
   RoPE handling (diff the providers), and that V R² is identical between them (H-G3).
8. Dispatch an independent skeptic per load-bearing claim with the instruction to refute
   it, and mark only survivors `[VALIDATED]`.
9. Check the status tags, the ASCII console rule, that no git history was written by the
   session, and that the README's record/proposal split is still honest.

Anything the contract names that you could not do: say so in the handoff under
"Open / next" with the reason, rather than leaving it for the evaluator to discover.
Scaling the work down is the operator's call; reporting what was left out is yours.
