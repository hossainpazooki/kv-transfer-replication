# Needle-position benchmark for length generalization — design `[PROPOSED]`

Status: **design, nothing here has run.** Written 2026-08-30 from a brainstorm with the
operator; the decisions and their reasons are recorded in §1 so the choices can be
re-litigated later against what was actually argued, not against memory.

This adds a third evaluation surface to the repository. The two that exist — HellaSwag
swap-in (`kvt/hellaswag.py`) and prefix-conditioned perplexity (`kvt/perplexity.py`) —
cannot answer the question WP3 actually poses. HellaSwag contexts are under 100 tokens.
Perplexity reaches long prefixes (Run 8) but measures next-token NLL, which says the mapped
cache *predicts the distribution*, not that the target can *use* what is in the prefix. A
retrieval task where the answer lives in the prefix, past the calibration length, is the
missing instrument.

## 1. Decisions made in the 2026-08-30 brainstorm

| question | decision | reason |
|---|---|---|
| Which gap do new benchmarks close first — a long-context task (Gate L), cheap high-throughput scoring for the predictor work (Gate T), or the paper's own benchmark suite? | **Long-context task first.** | It is the property content space claims to buy; Run 8 reached it only through perplexity. The other two remain open and are not addressed here. |
| Synthetic retrieval vs natural long-document QA? | **Synthetic needle only.** | Position is the variable the two mapper spaces differ on; exact-match scoring pairs per example so `mcnemar_exact` applies unchanged and n is small; the native baseline sits near ceiling so degradation has headroom. Natural QA has noisy scoring, uncontrolled lengths, and a 1.7B baseline that may leave no headroom. Its weakness — a mapper could preserve one salient number and destroy the prose — is covered by perplexity, which is its complement; both go in the note. |
| Independent variable: prefix length (Run 8's shape), needle position at fixed length, or the full grid? | **Position sweep at fixed P=4096, with P=1024 / position 512 as the in-calibration control.** | See §2. A length sweep moves prefix length and needle position together and cannot attribute a drop to either. |
| Haystack text: FineWeb-Edu held-out sequences, WikiText-2, or repeated-sentence filler? | **WikiText-2 test split**, the same stream `kvt/perplexity.py` chunks. | Correction to what was first proposed in the brainstorm (FineWeb-Edu held-out). The perplexity instrument already chose WikiText-2 precisely because a different corpus needs no leakage argument against the FineWeb-Edu calibration set; the same reasoning applies here and keeping the two WP3 instruments on one corpus makes their results comparable. Repeated-sentence filler is rejected because near-periodic keys make the mapping easier than any real text. |
| Mapper checkpoints? | **`k1` content and `k1` rope, the Run 8 checkpoints.** | Every fitted artifact is shared with Run 8; only the benchmark differs, so a disagreement between the two results is about the instrument, not the mapper. |

## 2. Why position, not length — the mechanism this design reads off

`apply_mapper` (`kvt/mapper.py`) is the only place the two spaces differ. Content space strips
source RoPE from K, applies the linear map, and re-applies target RoPE at the true position.
Rope space applies the linear map to the rotated K directly. V is mapped identically in both.

A rope-space mapper `W` is fitted on keys rotated by their position. RoPE rotates each 2-D
pair of head dimensions by `p * theta_i`. For the high-frequency pairs, the 1024 calibration
positions already sweep the full circle many times, so `W` has seen every angle and should
extrapolate. For the low-frequency pairs — wavelengths longer than 1024 tokens — positions
>= 1024 present rotation angles that never occurred in the fit, and `W` is unconstrained
there: its output is whatever the ridge solution does off-distribution.

Two testable consequences:

1. Rope-space corruption is a property of an entry's **absolute position**, not of prefix
   length. Entry 700 in a 4096-token prefix is rotated exactly as entry 700 in a 1024-token
   prefix.
2. The corruption should **grow with position** past 1024 rather than step, because the
   low-frequency angle drifts further from the fitted range the further out it goes. Run 8's
   gap (0.7 pt at P=1024, 5.7 pt at P=2048) is consistent with this but cannot distinguish it
   from "long prefixes are generally worse" — its length sweep moved both at once.

Content space has no such regime: the fit never sees a rotation, and target RoPE is applied
at the true position by the same `apply_rope` the injection gate checks bitwise. It predicts
a flat line across position.

Holding P fixed and moving only the needle holds every entry's rotation fixed except the one
thing being varied, so the prediction is read directly off the x-axis.

## 3. The grid

| P | needle absolute position | role |
|---|---|---|
| 1024 (control) | 512 | in-calibration; both mappers **must** agree here. Anchors the difference-of-differences, the role P=1024 played in Run 8. |
| 4096 | 256, 768 | needle inside the calibration range, followed by 3000+ entries that are out of range under rope space. **Critical cells, exploratory** (§5, H-N4). |
| 4096 | 1280, 2048, 3584 | needle itself out of range. Predicted rope-space drop, growing with position. **Confirmatory** (§5, H-N1). |

Conditions per cell, the Run 8 five: `native`, `native-injected` (gate), `identity` (null:
source cache injected unmapped), `content-k1`, `rope-k1` — the exact names `eval_perplexity.py`
gives its `extra` conditions, so summaries of the two instruments line up column for column.

**Pairing.** The same 48 (haystack window, key) examples are used in every cell; only the
needle's position moves. Every cell is therefore paired with every other cell by example,
which is what makes the difference-of-differences a paired statistic across 48 examples
rather than a comparison of two unrelated samples.

**Fallback grid.** If the native ceiling pre-check (§6) fails at P=4096, the grid becomes
P=2048 with positions {256, 768, 1280, 1664, 1920} and the same control. Both grids are
written into the pre-registration so the choice between them is made by the pre-check, not
by the result.

## 4. Protocol

**Example construction.** Haystack windows are drawn from WikiText-2 test by the existing
deterministic front-of-stream chunker (`_chunk`), sized at 4096 + slack, so a re-run selects
the same text. Windows 0-47 are the test set; windows 48-71 are reserved for the native
pre-check and timing smoke and never enter a reported cell. The needle is one sentence,
`The pass key is <6 digits>. Remember it.`, tokenized separately and spliced into the token
stream so that its **first token sits at absolute index `p`** and the whole prefix is exactly
`P` tokens (the haystack contributes `P - len(needle)` tokens). Keys are 6 digits with no
leading zero, drawn from a seeded RNG recorded in the artifact, unique across examples. For
the P=1024 control the haystack is the trailing 1024 tokens of the same window
(`slice_for_prefix` pattern), so the control shares text with the P=4096 cells.

**Query.** `\nQuestion: What is the pass key?\nAnswer: The pass key is`, processed
**natively by the target** after injection. The query is the user turn; the question is
whether the mapped cache is usable by a target that prefills its own question. This mirrors
the paper's "prefill by source, decode by target" setting and HellaSwag's A1 protocol.

**Injection (protocol A1, unchanged from hellaswag/perplexity).** For a prefix of `P`
tokens, the cache covers `ids[:P-1]`; the target is fed `ids[P-1]` followed by the query
tokens at positions `P-1 ..`, then greedily decodes up to `len(key tokens) + 2` tokens.
`native` forwards `ids[:P]` + query whole. The Qwen tokenizer emits one token per digit, so
a 6-digit key is 6 decode steps.

**Scoring.** An example is correct iff the decoded text, whitespace-stripped, starts with the
key. Correctness is a per-example boolean, so `kvt/stats.py::mcnemar_exact` applies
unchanged.

**Gate.** `native-injected` must produce **identical decoded token ids** to `native` on
every example in every cell (first-step argmax equality is implied). Any mismatch stops the
run: a broken harness is indistinguishable from a broken mapper.

**Null validity.** `identity` must fail. If the unmapped source cache retrieves the needle at
>= 20 % in any cell, the task is solvable from garbage and the benchmark is invalid — stop.

## 5. Pre-registered hypotheses

These are copied **verbatim** into `docs/prereg/run9-needle.md` and committed before any
content/rope output exists (§7). The spec is the design; the prereg file is the contract.

- **H-N0 (gate).** `native-injected == native` decoded ids on all 48 x 6 example-cells.
  Otherwise stop; nothing below is interpretable.
- **H-N1 (primary, confirmatory).** Pooled over the three out-of-range cells at P=4096
  (positions 1280, 2048, 3584; 144 paired example-cells), `content` beats `rope`:
  `b` (content right, rope wrong) exceeds `c` (the reverse) with exact two-sided McNemar
  `p < 0.05`. One test; no correction.
- **H-N2 (secondary).** Per-cell McNemar content vs rope at each of 1280, 2048, 3584, and a
  descriptive check that rope-space accuracy is non-increasing across 1280 -> 2048 -> 3584.
  Three tests at Bonferroni `0.05 / 6 = 0.0083` (six secondary tests planned in total: three
  here, one in H-N3, two in H-N4). The ordering is reported, not tested.
- **H-N3 (control).** At P=1024 / position 512, `content` and `rope` are each within 10 pp
  of `native`, and content vs rope McNemar is not significant at 0.0083. If this fails, the
  Run 8 equivalence at the calibration length did not transfer to this task, and the
  **difference-of-differences becomes the reported primary**: per example `w`,
  `s_w(cell) = content_correct - rope_correct` in {-1, 0, 1}; `DoD_w = mean over the three
  out-of-range cells of s_w - s_w(control)`; exact sign test across the 48 examples.
- **H-N4 (exploratory, no direction predicted).** Rope-space accuracy at positions 256 and
  768 in P=4096. Two named readings, fixed in advance: within 10 pp of `native` -> "damage is
  local to out-of-range entries" (a rope-to-1024 / content-past-1024 hybrid becomes a live
  idea); at least 25 pp below `native` -> "out-of-range entries corrupt retrieval of in-range
  needles through attention". Between those -> inconclusive. Content vs rope McNemar in
  these two cells is reported at 0.0083 but carries no prediction.
- **H-N5 (null).** `identity` accuracy < 20 % in every cell. Otherwise stop (§4).

**Power, from the test's own arithmetic.** With 144 pooled pairs, `b = 12, c = 2` gives
`p = 0.013`; that is a net 7 pp gap and the smallest effect this design resolves. A
30 pp gap (`b ~ 45, c ~ 2`) is decisive by orders of magnitude. Per cell, n=48 resolves
gaps of roughly 20 pp and no finer — which is why the primary pools.

**What each outcome means.**
- H-N1 confirmed, H-N3 holds: the mechanism in §2 is supported on a retrieval task, not just
  on NLL. Run 8's gap is position-local corruption.
- H-N1 confirmed, H-N3 fails: content beats rope everywhere on this task; DoD decides whether
  the advantage *grows* past calibration.
- H-N1 not confirmed with content also degrading past 1024: position is not the story; Run 8's
  gap has another cause and the note says so.
- H-N1 not confirmed with neither degrading: the needle is too easy for k1 mappers at these
  lengths; the perplexity result stands alone and this instrument is reported as insensitive.

## 6. Pre-checks, in order, all before launch

1. **Native ceiling pre-check.** Windows 48-71 (24 examples, disjoint from the test set),
   `native` only, P=4096, needle at 2048. Accept P=4096 if native accuracy >= 90 %
   (>= 22/24). Otherwise the fallback grid at P=2048 (§3), same threshold; if that also
   fails, stop and record — the target cannot do the task natively at these lengths and no
   mapper comparison is meaningful. The outcome is appended to the prereg file and
   committed **before** the sweep.
2. **Timing smoke.** 2 examples from windows 48-71, all five conditions, at the chosen P and
   one position. Its purpose is the wall-clock number for the ledger; it runs only after the
   prereg commit, so no content-vs-rope output exists on disk before the rule (the Run 8
   lesson, GPU plan §1 rule 2).

**Cost estimate, to be replaced by the smoke.** Per example-cell: one source prefill of P
tokens, one target prefill (for `native`), then five short target forwards over the query
plus ~8 decode steps. The injected conditions never run the target over the prefix, which is
why this is cheaper per point than perplexity, where every condition scored a 256-token
continuation. Guess from Run 8's measured 157 s for 2 windows x 5 conditions at P=2048:
1-2 min per example-cell at P=4096 on the CPU box, so 48 x 6 = 288 example-cells is
5-10 h. **Cell order**, so an interruption leaves the informative cells: control (1024/512),
then 4096/2048, 4096/3584, 4096/1280, 4096/768, 4096/256.

## 7. Pre-registration mechanism — first use of the GPU plan §1 rule

Run 8's rule preceded its data by eleven seconds, provable only from the session log
(`docs/ledger.md`, provenance paragraph under the Run 8 design). This run is the first to
use the mechanism adopted in response:

1. `docs/prereg/run9-needle.md` holds §5 verbatim plus the grid, the pre-check rule and
   (after step 6.1) its outcome. It is **committed by the operator** before launch.
2. `scripts/eval_needle.py --prereg-sha <commit>` stamps that SHA into every record.
3. `scripts/summarize_needle.py` refuses a results directory whose records carry more than
   one `prereg_sha`, or none; it prints the SHA in its header so the ledger entry quotes it.
   (Ancestry checking against `git` is deliberately not done here — the summarizer must not
   shell out to git — the operator verifies `git merge-base --is-ancestor` by hand when
   writing the ledger entry.)

## 8. Code

New, disjoint from existing modules:

- `kvt/needle.py` — `make_examples(tokenizer, n, P, seed)`, `splice(haystack_ids, needle_ids,
  p, P)`, `encode_query(tokenizer)`, `decode_greedy(model, cache, ids, past_len, max_new)`,
  `is_correct(decoded_text, key)`, and `providers`-style condition wiring reusing
  `kvt.hellaswag.providers` (which already builds native-injected / identity / mapped from a
  source, target and mapper dict).
- `scripts/eval_needle.py` — `--pair --P --positions --n --tag --prereg-sha --content-mapper
  --rope-mapper --seed`. Writes
  `results/needle/<pair>/<tag>/P<P>/pos<p>/<condition>.jsonl`, one record per example:
  `example_id, window_idx, key, position, condition, decoded_ids, decoded_text, correct,
  prereg_sha`. Refuses to overwrite a cell directory holding a different `n` or a different
  `prereg_sha` (the `eval_perplexity.py --tag` guard, generalised). Runs cells in the §6
  order.
- `scripts/summarize_needle.py` — fail-closed in the perplexity summarizer's family: every
  cell must hold all five conditions with identical example-id sets; gate exact on every
  example; `identity` under 20 %; a cell holding only the gate pair is **refused** with the
  message `[GATE ONLY]` — this is the Run 8 summarizer fail-open (ledger, "What fired"),
  fixed at birth here. Emits the §5 tests via `mcnemar_exact`, the DoD sign test, and a
  markdown table per cell, plus `summary.json`.
- **Same-family fix, separate commit:** `scripts/summarize_perplexity.py` gains the same
  refusal for a prefix-length directory holding only `native` and `native-injected`. Small,
  and the ledger records it as "not yet fixed"; leaving it open while building its sibling
  correctly would be inconsistent.

Tests (`tests/test_needle.py`), each with the check that would fail if the thing it guards
were broken:

- `splice` puts the needle's first token at exactly `p` and returns exactly `P` tokens, for
  every grid position including `p + len(needle) > P` (must raise).
- `is_correct` accepts the key with leading whitespace, rejects a 5-digit prefix of it and
  rejects the key appearing after other text.
- Keys are unique and never begin with `0`.
- Summarizer refuses: a missing condition; mismatched example ids; a gate mismatch; a
  gate-only cell; mixed `prereg_sha`. **Mutation-tested**: with each refusal commented out,
  the corresponding test must fail (the k=1 self-mapper gate lesson).
- `--prereg-sha` appears in every written record.
- End-to-end on the `tiny_src3` / `tiny_tgt` random-weight fixtures in `tests/conftest.py`
  (equal layer counts, so `identity` is constructible): native-injected ==
  native decoded ids (the gate passes on a harness that is correct by construction).

Ridge, RoPE, mappers, cache injection: untouched. `kvt/hellaswag.py::providers` is reused,
not modified.

## 9. Records this run produces

- `docs/prereg/run9-needle.md` — committed first (§7).
- `results/needle/qwen3-0.6b-to-1.7b/<tag>/` — the raw artifacts; every number in the
  ledger recomputes from them via the summarizer.
- `docs/ledger.md` — "Run 9" entry, `[BASELINE]` on completion, `[BASELINE, PARTIAL]` if
  interrupted, quoting the prereg SHA and the ancestry check.
- `docs/learnings/` — an entry for anything that fires, in the existing form.
- `README.md` — WP3 row updated from "length sweep not run" to what Run 9 found, tagged
  honestly; the "What it does" flow gains a needle node beside perplexity.

## 10. What this design does not do

- Does not test natural long-document QA. The "single salient number" objection stands and
  is answered by perplexity, not by this instrument.
- Does not address Gate T (cheap scoring for the predictor) or paper-suite fidelity; both
  remain open follow-ons.
- Does not run on GPU. It is designed for the CPU box; the GPU plan's bf16 question is
  orthogonal and its gate re-characterisation (Phase 0) would have to pass before this runs
  there.
- Does not touch the second pair. If Run 9 lands, the same script runs on
  `qwen3-1.7b-to-4b` once its mappers exist.
