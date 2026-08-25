# Ledger

## Hypotheses (pre-registered, before any run)

- **H1 (structure):** the stripped-K probe heatmap shows a diagonal ridge; stripped-K R² > RoPE-K R²; K R² > V R² by a visible margin (paper: ~0.2 at 14B→32B; magnitude at 0.6B→1.7B unknown).
- **H2 (overfit floor):** for pure noise, in-sample R² ≈ p/n. Probe: p = 128, n_train ≈ 40 seqs × 256 tok = 10,240 → floor ≈ 0.0125 (small). Top-k at k=8: p = 8·8·128 = 8,192 vs n_train = 10,240 → floor ≈ 0.8, i.e. **the k=8 in-sample R² lift at 50 sequences is mostly interpolation**; held-out R² is the number to trust. At 200 sequences p/n ≈ 0.2, still not small. (The paper's own regime: 128K tokens vs p = 8,192 → 6.4%; k = 20 → 16%.)
- **H3 (downstream):** identity control (raw source cache injected, no mapper) lands near chance; mapped k=8 retains more HellaSwag than k=1; raw and floor-normalized retention are both reported and diverge when retention is low.
- **H4 (assumption to test, not the paper's claim):** the last-token protocol below (target sees only the final context token; all earlier context arrives as mapped cache) reproduces native accuracy *exactly* when fed the target's own cache. If not, the injection path is wrong, and nothing downstream is interpretable.

## Protocol decisions

- **A1 last-token protocol.** For a context of tokens `c[0..n-1]` and ending `e`, the source prefills `c[0..n-2]`, the cache is mapped and injected, and the target processes `c[n-1] + e` at positions `n-1, n, …`. The target needs the logit at position `n-1` to score `e[0]`, and we do not have the target's hidden state there from a cache alone. The paper says "the full prompt is prefilled by the source and mapped" without stating how the first continuation logit is obtained; A1 is the minimal faithful reading.
- **A2 tokenization.** lm-eval convention: `whole = enc(ctx + cont)`, `n_ctx = len(enc(ctx))`, continuation tokens are `whole[n_ctx:]`. HellaSwag `preprocess()` is copied from lm-eval verbatim (Task 8).
- **A3 subsample.** HellaSwag validation subsampled to N=500 with seed 0. Wilson 95% CI at N=500 is about ±4 pp; this resolves retain-vs-collapse, not sub-2 pp differences. Say so wherever numbers are quoted.
- **A4 identity control.** Because the pair is matched-KV, the raw source cache is shape-compatible with the target. Injecting it unmapped is the null: it measures how much of any retention comes from the mapper rather than from shape compatibility alone.
- **A5 R² definition.** Per target head: `1 − Σ‖y − ŷ‖² / Σ‖y − ȳ‖²` with sums over tokens and all `d_h` dims pooled; `ȳ` is the mean of the set being scored (train mean for train R², held-out mean for held-out R²). Head-averaged = arithmetic mean over `n_kv` heads.

## Follow-on hypotheses (pre-registered 2026-08-24, before any WP1-WP3 result artifact)

Plan: `docs/superpowers/plans/2026-08-24-cache-economics-followon.md`.
Spec: `docs/superpowers/specs/2026-08-24-cache-economics-followon-spec.md`.
Written by the controller before any run, and before the code implementing WP1-WP3 was
finished, so the ordering is checkable against file mtimes in `results/`.

**WP1 composition.** A = 0.6B->1.7B, B = 1.7B->4B, C = 0.6B->4B fitted directly.

- **H-C1 (composition holds):** held-out K R^2 of B(A(x)) is within 0.05 of C's, AND
  floor-normalized HellaSwag retention of `composed-BA-k1` is within 3 pp of `mapped-C-k1`
  with McNemar p > 0.05. If so, a family of n sizes needs n-1 mappers along a chain rather
  than n(n-1) directed pairs.
- **H-C2 (composition degrades):** the held-out K R^2 gap exceeds 0.05 and McNemar rejects
  at p < 0.05 with `mapped-C-k1` ahead. Each hop adds its own residual.
- **H-C3 (gate, not a result):** closed-form `compose(A, B)` agrees with `apply_mapper`
  applied twice to max abs 2e-3. Failure means a bug; do not interpret any composed number
  until it passes.
- **Correction to the spec, found while reading the code.** The spec listed an `identity`
  control for this pair. There is none, and there cannot be: injecting a 28-layer cache
  into a 36-layer model is undefined and `kvt/hellaswag.py` correctly raises. **Matched-KV
  does not imply an identity null exists** -- that null is available only at equal depth,
  which is why Run 4 had one and WP1 cannot. `source` (0.6B scored alone) is the floor
  reference instead. Recorded here because the missing null genuinely weakens WP1 relative
  to Run 4, and that should not be discovered later and quietly absorbed.

**WP2 calibration learning curve.** Held-out FIXED at sequences [400, 420); training
prefixes [0, n) for n in {50, 100, 200, 400}; k in {1, 4, 8}. Selection is computed ONCE and
shared across all n, so n is the only variable.

**Corrected 2026-08-24** (this paragraph originally said selection is computed "on the n=400
prefix", which contradicts the implementation and the "What fired" entry below): selection is
REUSED from the existing 50-sequence probe's **train** arrays over sequences [0, 40), which
`scripts/fit_mapper.py` loads unconditionally. Those sequences sit inside every training
prefix and are disjoint from the held-out range [400, 420), so the selection is leak-free and
genuinely constant across the curve. Re-probing the 420-sequence dump would have been worse,
not better: `probe_r2` splits by FRACTION, so its held-out range would be [336, 420) and would
overlap the curve's held-out set.

- **H-L1:** held-out K R^2 at k=8 rises monotonically in n.
- **H-L2 (the collapse was calibration size):** at n=400 (p/n = 0.10), k=8 held-out K R^2
  >= k=1 held-out K R^2, and mapped-k8 floor-normalized retention >= mapped-k1 - 3 pp.
- **H-L3 (the collapse was the method):** k=8 has not crossed k=1 by n=400. Then cross-layer
  selection costs accuracy on this pair even outside the interpolation regime, and the
  paper's k-selection result does not transfer to 0.6B->1.7B.
- **H-L4:** p/n is quoted beside every in-sample R^2 in the curve table.

**Nesting verified before the curve was run.** `iter_fineweb_sequences` takes the first n
qualifying documents after a seeded shuffle, so the n=420 draw should extend the n=50 draw
rather than replace it. Confirmed rather than assumed: `np.array_equal(n50, n420[:50])` is
True (both `len=1024`, seed 0). This matters because the previous HellaSwag runs were NOT
nested (`rng.choice` is not nested across n -- see
`docs/learnings/2026-08-24-rng-choice-not-nested-across-n.md`), and repeating that mistake
would make the curve a comparison of different corpora rather than a curve in n.

**Amendment to WP2 before any WP2 result exists (2026-08-24).** The 420-sequence dump is
byte-nested in its TOKENS (verified exactly) but NOT bit-identical to the 50-sequence dump in
its KV VALUES: it was produced at `--threads 12` while the original used the default, and BLAS
reduction order plus fp16 storage puts the two up to one fp16 ULP apart, growing with depth
(layer 27 max abs 3.125e-02, relative 3.1e-04, mean 7.1e-06). See
`docs/learnings/2026-08-24-kv-dumps-are-not-bit-reproducible-across-thread-counts.md`.
Consequence, adopted as a protocol decision rather than discovered later: **every point on the
WP2 curve is recomputed from the single 420-sequence dump, including the n=50 point.** Run 2's
committed figures (k=1 held-out K R^2 = 0.6814 etc.) must NOT be spliced in as the curve's first
point, because a 1e-04-scale difference between two curve points could otherwise be a
thread-count artifact rather than an effect of n. This costs one extra fit and removes a
confound that would have been invisible.

**WP3 length generalization.** Fit at 1k context; evaluate WikiText-2 prefix-conditioned
perplexity across P.

**Correction to this pre-registration, made 2026-08-24 BEFORE any WP3 perplexity run.** It
originally listed P in {512, 1024} as required and {2048, 4096} as `[STRETCH]`. That is
backwards, and would have produced a "length generalization" result that tested no such thing.
The calibration sequences are 1024 tokens, so the mapper only ever saw positions 0-1023. At
P = 1024 the cache covers ids[:1023] and `apply_mapper` runs at positions 0-1022 -- **entirely
inside** the fitted range. P = 512 and P = 1024 are therefore BASELINE points, comparable to
Run 5's R^2, not tests of extrapolation. The first P at which the mapper is applied to unseen
positions is **P = 2048** (positions 1024-2046 are extrapolated), so 2048 is the minimum
meaningful test of H-G1/H-G2 and is now REQUIRED, with 4096 as the stretch point. Recorded
rather than silently re-scoped, because "we ran the length test at 512 and 1024" would have
been a correct-shaped lie: real numbers, real code, answering a different question than the
one named.

- **H-G1 (content space buys length):** relative degradation vs native is flat in P for the
  content-space mapper and grows with P for the rope-space mapper; the gap exceeds 5%
  relative perplexity at the largest P run.
- **H-G2 (it does not):** both flat, or both degrade equally. Then strip/re-apply buys
  nothing at up to 4x calibration length on this pair and the serving path can drop it.
- **H-G3 (control, not a result):** V carries no RoPE, so V's weights and V's held-out R^2
  must be IDENTICAL between the two variants. If they differ, the variants differ in
  something other than RoPE handling -- a bug.

Decision rule for all three work packages: state the verdict against the rule as written
above, before considering which outcome is the more interesting one to report.

## Runs

### Run 1 — calibration dumps + single-source OLS probe (replication steps 1-2) `[BASELINE]`

2026-08-23. Pair Qwen3-0.6B -> Qwen3-1.7B (matched-KV: both 28 layers, n_kv 8, d_h 128,
rope_theta 1e6, verified from the loaded configs). 50 FineWeb-Edu sequences x 1024 tokens,
stride-4 subsampled -> 12,800 token observations; held out by sequence (last 10 of 50) ->
**10,240 train / 2,560 held-out**. CPU fp32, 16 cores.

Wall-clock: source dump 3m25s, target dump 8m36s, probe 22m39s.

Dump alignment was verified rather than assumed: source and target dumps have identical
`positions`, identical `seq_idx`, and identical train masks, both (12800, 8, 128) per layer.

Head-averaged R^2, best cell chosen by held-out score. Every figure below is recomputed
from `results/probe/qwen3-0.6b-to-1.7b/r2_<kind>_<split>.npy`.

| cache kind | best held-out cell | train R^2 | held-out R^2 | diag mean train | diag mean held-out | mean gap |
|---|---|---|---|---|---|---|
| K_rope     | (src 25, tgt 25) | 0.7751 | 0.7568 | 0.6782 | 0.6606 | 0.0292 |
| K_stripped | (src  0, tgt  0) | 0.7923 | 0.7606 | 0.6569 | 0.6284 | 0.0402 |
| V          | (src 19, tgt 19) | 0.5649 | 0.5473 | 0.4698 | 0.4361 | 0.0415 |

Heatmap: `results/probe/qwen3-0.6b-to-1.7b/figure2.png`.

**Verdicts against the pre-registered hypotheses.**

- **H1a (diagonal structure) — HELD.** The best held-out cell lies exactly on the diagonal for
  all three cache kinds: (25,25), (0,0), (19,19). Linear structure between the two models'
  caches is strongest at matching depth, as the paper reports.
- **H1b (stripped-K should beat RoPE-K) — NOT CONFIRMED. My prediction was wrong.** On the
  diagonal mean, RoPE-contaminated K scores *higher* than stripped K (0.6606 vs 0.6284,
  delta -0.0322); at the single best cell stripped K wins by a trivial +0.0038. So stripping
  RoPE did not improve fit quality here.
  This is not a contradiction of the paper — it corroborates the paper's own RoPE ablation,
  which found "-all RoPE" (rotation coupled at both fit and inference) to be within noise on
  short context. Our source and target share `rope_theta` and see identical positions, so the
  rotation applied to both sides is the same and stripping it removes no confound. The paper's
  content-space mapping is claimed to buy **length generalization**, not short-context accuracy;
  we do not test length generalization here, so we neither confirm nor refute that claim.
  The over-sharp prediction was mine, not the paper's.
- **H1c (K more predictable than V) — HELD, and close to the paper's magnitude.** Diagonal-mean
  held-out K minus V is **+0.192** (stripped) / **+0.225** (RoPE-K). The paper reports K being
  roughly 0.2 R^2 more predictable than V on Qwen3 14B->32B. Matching that separation at
  1/20th the parameter scale is the strongest single point of agreement in this run.
- **H2 at probe scale — HELD.** With p/n = 128/10,240 = 0.0125 the overfit floor is negligible,
  and observed train-minus-held-out gaps are correspondingly small: mean 0.029 / 0.040 / 0.042,
  worst single cell 0.116. **So at probe scale the paper's in-sample-only R^2 is trustworthy.**
  The gap that matters is predicted to appear in the top-k mapper, where p/n reaches 0.800 —
  measured in Run 2, not here.

### Run 2 — top-k cross-layer ridge mappers, k in {1, 4, 8} (replication step 3) `[BASELINE]`

2026-08-23, same dumps as Run 1. Ridge lambda = 0.01, float64 centered solve, selection by
head-averaged probe R^2 averaged over stripped-K and V, shared across heads and across K/V.
Fit time for all three k: 13m04s. Recomputed from `results/mapper/qwen3-0.6b-to-1.7b/r2.json`.
Layer-averaged R^2 over the 28 target layers:

| k | p = k*n_kv*d_h | p/n | K train | K **held-out** | K gap | V train | V **held-out** | V gap |
|---|---|---|---|---|---|---|---|---|
| 1 | 1,024 | 0.100 | 0.7783 | 0.6814 | +0.097 | 0.6596 |  0.5133 | +0.146 |
| 4 | 4,096 | 0.400 | 0.8816 | 0.5907 | +0.291 | 0.8206 |  0.3361 | +0.485 |
| 8 | 8,192 | 0.800 | **0.9607** | **0.0984** | **+0.862** | 0.9476 | **-0.6412** | +1.589 |

**This is the result the replication was built to produce, and H2 is confirmed emphatically.**

Read the in-sample column alone — the metric the paper reports — and the mapper improves
monotonically with k: 0.78 -> 0.88 -> 0.96. Read the held-out column and it collapses
monotonically: 0.68 -> 0.59 -> 0.10, with V falling to **-0.64**, i.e. worse than predicting
the held-out mean. At k=8 the two columns disagree by 0.86 R^2 on keys and 1.59 on values.

A practitioner who selected k on in-sample R^2 at this calibration size would pick k=8, the
worst of the three. The apparent gain from cross-layer selection is, at 50 sequences, almost
entirely interpolation: p/n = 0.800 at k=8, and the controller's independent noise simulation
(no signal at all) returns in-sample R^2 = 0.8016 at exactly that p/n.

**This does NOT refute the paper.** The paper calibrates on ~128K tokens, 12.5x our 10,240,
putting its k=8 p/n at about 0.064 where this effect is small. What the run establishes is
that the paper's headline metric is only safe well away from the interpolation regime, and
that a small-scale replication reporting in-sample R^2 would draw the opposite conclusion
about k while looking more successful. Calibration size is load-bearing, and the paper's
choice of 128K tokens is doing real work that its in-sample-only reporting makes invisible.

Also confirmed on the real fitted mappers: `n_weight_params == formula_params` exactly for
all three k (58,720,256 / 234,881,024 / 469,762,048), so the paper's Appendix D size formula
holds for actually-constructed mappers, not just arithmetic.

Selection sanity checks: at k=1 the chosen source layer is the **identity map** for all 28
target layers (target i picks source i) — an independent, per-layer confirmation of the
diagonal structure. At k=4, selections cluster locally around matching depth
(e.g. target 13 -> [13, 14, 12, 16]).

### Run 3 — HellaSwag swap-in, n=50 smoke (replication step 4) `[SUPERSEDED by Run 4]`

> **Read Run 4 instead for any number in this section.** Run 3 was a 50-example smoke test
> and one of its headline figures did not survive the 500-example run: it put mapped-k8 at
> acc_norm 0.160 with floor-normalized retention **-0.310** (i.e. below chance), and Run 4
> measures 0.312 / **+0.178** (i.e. above chance). The +/-14 pp interval at n=50 was doing
> exactly what it warns about. The section is kept unedited below as the record of what was
> claimed before the larger sample corrected it.

2026-08-23, 16m13s for all seven conditions. Mappers from Run 2. HellaSwag validation
subsampled to 50 with seed 0. **n=50 gives roughly +/-14 pp Wilson intervals, so read only
the large separations here**; the n=500 run supersedes this for anything finer. All figures
recomputed from the per-example JSONL by `scripts/summarize_hellaswag.py`.

| condition | acc_norm | 95% CI | retention raw | retention floor-norm |
|---|---|---|---|---|
| native (Qwen3-1.7B)      | 0.540 | [0.404, 0.670] | 1.000 | 1.000 |
| native-injected (gate)   | 0.540 | [0.404, 0.670] | 1.000 | 1.000 |
| source alone (Qwen3-0.6B)| 0.360 | [0.241, 0.499] | 0.667 | 0.379 |
| identity (null control)  | 0.300 | [0.191, 0.438] | 0.556 | 0.172 |
| mapped-k1                | 0.460 | [0.330, 0.596] | 0.852 | **0.724** |
| mapped-k4                | 0.360 | [0.241, 0.499] | 0.667 | 0.379 |
| mapped-k8                | 0.160 | [0.083, 0.285] | 0.296 | **-0.310** |

**Gate H4 holds on real weights.** native-injected reproduces native to a maximum
log-probability difference of 8.4e-05, with per-example argmax identical on 50/50 examples and
identical acc_norm. The injection path is sound, so the other rows are interpretable.

**The mapper does real work at k=1.** 0.460 against an identity null of 0.300 and a chance
floor of 0.25 — floor-normalized retention 0.724. Cross-model KV transfer genuinely functions
on this pair; that part of the paper replicates.

**And now the central result, joining Run 2 to Run 3:**

| k | in-sample K R^2 | held-out K R^2 | HellaSwag acc_norm | floor-norm retention |
|---|---|---|---|---|
| 1 | 0.7783 | 0.6814 | 0.460 | +0.724 |
| 4 | 0.8816 | 0.5907 | 0.360 | +0.379 |
| 8 | **0.9607** | 0.0984 | **0.160** | **-0.310** |

Downstream accuracy is **perfectly rank-correlated with held-out R^2 (+1.0) and perfectly
rank-ANTI-correlated with in-sample R^2 (-1.0)** across the three settings. The configuration
with the best reconstruction score the paper's methodology would report — k=8, in-sample
R^2 = 0.96 — performs **below chance** downstream, with negative floor-normalized retention.
Selecting k on in-sample R^2 at this calibration size does not merely fail to help; it inverts
the ranking and picks the actively harmful configuration.

Note also what floor normalization does here. Raw retention for k=8 reads 0.296 — "about 30%
retained", which sounds like degradation. Floor-normalized it is **-0.310**: worse than
answering at random. Raw retention on a 4-way task cannot fall below ~25% however broken the
cache is, and that arithmetic floor is exactly what conceals a catastrophic result. This is
the paper's own Appendix F point, reproduced from the other direction.

The k=1 vs k=8 separation is significant even at n=50 (CIs [0.330, 0.596] vs [0.083, 0.285]
do not overlap). The k=1 vs k=4 difference is not resolved at this sample size.
**[SUPERSEDED by Run 6 for n=500]** — that sentence reasons from UNPAIRED intervals. A paired
McNemar test on the n=500 data resolves k=1 vs k=4 at p = 1.26e-03. It says nothing about n=50.

### Run 4 — HellaSwag swap-in, n=500 (replication step 4, authoritative) `[BASELINE]`

2026-08-23, 170m57s for all seven conditions on CPU. Mappers from Run 2. HellaSwag validation
subsampled to 500 with seed 0; Wilson 95% intervals are about +/-4 pp. Every figure recomputed
from the per-example JSONL by `scripts/summarize_hellaswag.py`, which was run with
`--expect-n 500` so a short or mismatched condition would have refused to summarize.

| condition | acc_norm | 95% CI | retention raw | retention floor-norm |
|---|---|---|---|---|
| native (Qwen3-1.7B)       | 0.598 | [0.554, 0.640] | 1.000 | 1.000 |
| native-injected (gate)    | 0.598 | [0.554, 0.640] | 1.000 | 1.000 |
| source alone (Qwen3-0.6B) | 0.474 | [0.431, 0.518] | 0.793 | 0.644 |
| identity (null control)   | 0.362 | [0.321, 0.405] | 0.605 | 0.322 |
| mapped-k1                 | 0.548 | [0.504, 0.591] | 0.916 | **0.856** |
| mapped-k4                 | 0.498 | [0.454, 0.542] | 0.833 | 0.713 |
| mapped-k8                 | 0.312 | [0.273, 0.354] | 0.522 | 0.178 |

**Gate H4 holds at scale.** native-injected reproduces native on **500/500** per-example
argmax decisions, max log-probability difference 1.75e-04. The injection path is exact, so
every other row measures the mapper rather than the harness.

**Cross-model KV transfer works on this pair.** mapped-k1 retains **85.6%** floor-normalized
(0.548 vs native 0.598), well clear of the identity null at 0.322 and of the source model's
own 0.644. The paper's core claim replicates at 1/20th scale.

**The central finding, restated on the authoritative sample:**

| k | in-sample K R^2 | held-out K R^2 | acc_norm | floor-norm retention |
|---|---|---|---|---|
| 1 | 0.7783 | 0.6814 | 0.548 | +0.856 |
| 4 | 0.8816 | 0.5907 | 0.498 | +0.713 |
| 8 | **0.9607** | 0.0984 | **0.312** | +0.178 |

Downstream accuracy still falls monotonically as in-sample R^2 rises: rank correlation with
held-out R^2 is +1.0, with in-sample R^2 -1.0. The k=1 vs k=8 separation is robust —
[0.504, 0.591] against [0.273, 0.354], no overlap. **Choosing k by the reconstruction score
the paper reports would select k=8, which is the worst of the three by a wide margin.**

**Correction against Run 3, stated plainly.** The n=50 smoke reported k=8 *below chance* with
negative floor-normalized retention (-0.310). That does not survive: at n=500, k=8 scores
0.312 with CI [0.273, 0.354], comfortably above the 0.25 floor, floor-normalized **+0.178**.
So mapped-k8 is badly degraded, not anti-predictive. The claim was put too strongly before the
larger sample existed.

**And a correction to that correction, found by the adversarial pass.** It is too generous to
write the Run 3 / Run 4 discrepancy off as "sampling noise". Two facts, both recomputed:
(a) the two runs do **not** score nested example sets — `rng.choice(len(ds), size=n)` draws
differently for different n, so only **9 of the 50** smoke examples appear in the 500;
(b) comparing 8/50 correct against 156/500 gives **p ~= 0.025** (Fisher exact 0.024, normal
approximation 0.0251), i.e. *outside* the conventional 0.05 threshold rather than comfortably
inside noise. The n=500 figure is authoritative — ten times the sample, and its summary was
produced under the `--expect-n 500` gate — but the honest statement is that the two runs
disagree by more than sampling alone comfortably explains, on largely disjoint example sets.
Anyone re-running this should use nested subsamples so smoke and full runs are comparable.

**One suggestive result that is NOT established.** mapped-k8 (0.312) sits nominally *below*
the identity null (0.362) — i.e. the k=8 mapper would appear worse than injecting the raw
unmapped source cache. Their intervals overlap ([0.273, 0.354] vs [0.321, 0.405]), so at 95%
this is not resolved. Recorded as an open question, not a finding.

Also unresolved at n=500: mapped-k1 vs mapped-k4 ([0.504, 0.591] vs [0.454, 0.542] overlap).
The monotone trend across k=1/4/8 is carried by the k=8 endpoint.
**[SUPERSEDED by Run 6]** — RESOLVED. The overlapping Wilson intervals are unpaired and cannot
use the fact that both conditions score the same 500 examples; the paired McNemar test gives
p = 1.26e-03 (41 flips vs 16), surviving Bonferroni for its family of six. The monotone trend
is therefore carried by EVERY adjacent pair, not only by the k=8 endpoint.

Caveat carried: dumps are stored float16. Re-roping a stripped key recovers the original to
6.1e-05 max error, i.e. ~5e-4 relative, which caps attainable R^2 near 0.9999997 — far above
any value reported here, so it does not affect these conclusions.

### Run 5 — rope-space vs content-space mapper at k=1 (WP3, part 1 of 2) `[BASELINE]`

2026-08-24. Same 50-sequence dumps and same selection as Run 2, so the ONLY difference between
the two rows below is where the ridge is fitted: content space (strip source RoPE, map,
re-apply target RoPE — the paper's design) versus rope space (map the rotated keys directly).
lam = 0.01, k = 1, p/n = 0.100 for both. Recomputed from
`results/mapper/qwen3-0.6b-to-1.7b/r2.json` and `.../rope/r2.json`, not restated from stdout.

| space | K train | K **held-out** | V train | V held-out |
|---|---|---|---|---|
| content | 0.7783 | 0.6814 | 0.6596 | 0.5133 |
| rope    | 0.7842 | **0.7065** | 0.6596 | 0.5133 |

**H-G3 (the control) HOLDS exactly.** V carries no RoPE, so V's numbers must be identical
between the two variants or the variants differ in something beyond RoPE handling. They are
bit-identical: train delta 0.00e+00, held-out delta 0.00e+00. The comparison is therefore
genuinely controlled, which is the precondition for reading the K row at all.

**Rope space fits BETTER at short context: +0.0252 held-out K R^2 (0.70653 vs 0.68136).** This
extends Run 1's H1b result from the probe level to the mapper level. H1b found that stripping
RoPE made the single-source probe slightly worse (diagonal-mean held-out 0.6284 stripped vs
0.6606 un-stripped); the same ordering now holds for the fitted top-k mapper. So on this pair,
at the calibration length, the paper's content-space design costs a little reconstruction
quality rather than buying any.

**What this does NOT show.** This is R^2 on the calibration corpus at the calibration length
(1024 tokens). It is **not** a test of H-G1/H-G2, which are about generalization *past* that
length and require the WikiText-2 perplexity sweep at P > 1024. Nothing here confirms or
refutes the paper's actual claim for content space. If anything it sharpens the WP3 test: with
rope space ahead at short context, a content-space win at longer context would be a genuine
crossover rather than a widening of an existing lead — a stronger result than a flat comparison
would have given, in either direction.

### Run 6 — paired McNemar tests on the committed n=500 HellaSwag data `[BASELINE]`

2026-08-24. No new model inference: this recomputes from the existing per-example JSONL in
`results/hellaswag/qwen3-0.6b-to-1.7b/`, which `load_and_validate_records` guarantees covers
an identical 500-example set for every condition — the precondition for a paired test.
Correctness is `acc_norm` (byte-length-normalized argmax), the repo's headline metric.
Exact two-sided McNemar; `b` = examples A got right and B wrong, `c` = the reverse.

| comparison | acc A | acc B | b | c | McNemar p |
|---|---|---|---|---|---|
| mapped-k1 vs mapped-k4 | 0.548 | 0.498 | 41 | 16 | 1.26e-03 |
| mapped-k4 vs mapped-k8 | 0.498 | 0.312 | 114 | 21 | 1.16e-16 |
| mapped-k1 vs mapped-k8 | 0.548 | 0.312 | 136 | 18 | 1.33e-23 |
| mapped-k8 vs identity  | 0.312 | 0.362 | 61 | 86 | 4.74e-02 |
| mapped-k1 vs source    | 0.548 | 0.474 | 62 | 25 | 9.06e-05 |
| native vs native-injected | 0.598 | 0.598 | 0 | 0 | 1.00 |

Six comparisons, so the Bonferroni threshold is 0.05/6 = **8.3e-03**. Verdicts below are stated
against that, not against 0.05.

**Run 4's "unresolved" k=1 vs k=4 is now RESOLVED: k=1 beats k=4, p = 1.26e-03.** Run 4 recorded
this as open because the Wilson intervals overlap ([0.504, 0.591] vs [0.454, 0.542]). Those
intervals are UNPAIRED and cannot use the fact that both conditions score the same 500 examples.
The paired test can: 41 examples flipped one way against 16 the other. It survives Bonferroni.
So the monotone decline across k = 1, 4, 8 is now established on every adjacent pair, not only
on the k=1 vs k=8 endpoints. **This strengthens the repository's central claim** — selecting k
on in-sample R^2 picks the worst of the three — since the ordering no longer rests on the
endpoints alone.

**Run 4's "suggestive but NOT established" mapped-k8 vs identity stays NOT established.** The
paired test gives p = 4.74e-02, which is below 0.05 but **above the 8.3e-03 Bonferroni
threshold for this family of six**. Reporting it as significant would be exactly the
multiple-comparisons error this ledger should catch. Direction is consistent with the earlier
suspicion (identity ahead, 86 flips against 61), so the k=8 mapper may well be worse than
injecting the raw unmapped cache — but one marginal p-value inside a family of six is not the
evidence for it. It stays an open question, as Run 4 recorded it.

**The injection gate is exact on decisions, not merely on averages.** `native` and
`native-injected` differ on **zero** of 500 examples (b = c = 0). Run 4 reported matching
accuracy and a max log-probability difference of 1.75e-04; this adds that no individual
decision differs either, which is the stronger statement.

### Run 7 — WikiText-2 prefix-conditioned perplexity, P=512 only (WP3, part 2) `[BASELINE, INTERRUPTED]`

2026-08-24. **This run was killed partway and is INCOMPLETE.** P=512 finished all five
conditions (12 windows each); P=1024 has only `native` and `native-injected`; P=2048 never
started. Reported anyway because the completed slice enforces a gate that had never been run,
and because the partial result is worth recording accurately rather than discarding.

| prefix_len | condition | perplexity | % degradation vs native |
|---|---|---|---|
| 512 | native | 16.1154 | 0.00% |
| 512 | native-injected | 16.1154 | **0.00%** (gate) |
| 512 | content-k1 | 21.9325 | 36.10% |
| 512 | rope-k1 | 22.2331 | 37.96% |
| 512 | identity | 87.1186 | 440.59% |
| 1024 | native | 15.5183 | 0.00% |
| 1024 | native-injected | 15.5183 | **0.00%** (gate) |

**The injection gate holds on a second, independent instrument.** `native-injected` reproduces
`native` to four decimal places at both prefix lengths. Until now that invariant had only ever
been checked through the HellaSwag harness; `scripts/summarize_perplexity.py` now enforces it
here too (rtol 1e-4) and refuses to summarize if `native-injected` is absent. The README's
claim that both gates are re-checked in every evaluation run is true again.

**The mapper does substantial work.** The identity null — the raw unmapped 0.6B cache injected
into the 1.7B — degrades perplexity by 441%, against 36% for the mapped cache. Shape
compatibility alone buys very little, consistent with Run 4's HellaSwag finding.

**Content-space vs rope-space at P=512 is NOT established, despite the aggregate looking
decisive.** Content (36.10%) appears to beat rope (37.96%) by 1.86 points. Per window, though,
content is better on only **9 of 12**, and an exact two-sided sign test gives **p = 0.146**.
Twelve windows cannot resolve a difference this size. Recording the aggregate alone would have
been a third instance of this repository's own recurring failure — a summary statistic that
looks like a result until it is tested pairwise (see Run 6, where the reverse happened and a
paired test rescued a real effect from overlapping intervals).

**This says NOTHING about H-G1 or H-G2 either way.** P=512 is far below the 1024-token
calibration length, so the mapper is applied only at positions it was fitted on. Per the
correction recorded in the pre-registration above, the first informative point is P=2048, which
did not run. WP3's length question remains **open**.

## Adversarial verification `[VALIDATED]`

Three load-bearing claims were handed to an independent skeptic instructed to refute them,
recomputing from raw artifacts and never from `docs/ledger.md`, `summary.json`, or the
summarize script. The controller separately recomputed the same quantities with its own code.
Both passes agree.

**Claim 1 — the overfit floor. SURVIVES a strong attack.** p/n = 0.8000 and all six R^2 values
were reproduced bit-for-bit from `mappers/*.safetensors` plus the dumps, bypassing `r2.json`
entirely. The decisive test was the alternative explanation: if the held-out collapse were
distribution shift between the first 40 and last 10 sequences rather than overfitting, then
flipping which sequences are held out should change it. Re-fitting k=8 with the **first** 10
sequences held out instead of the last gives K held-out R^2 = **0.0766**, against 0.0984
originally. The collapse is unchanged. Distribution shift is refuted; interpolation stands.

**Claim 2 — downstream accuracy anti-tracks in-sample R^2. SURVIVES, with the claim narrowed.**
acc_norm and the Wilson intervals reproduce exactly; all seven conditions score an identical
500-element `idx` set; and unnormalized accuracy corroborates the same ordering
(0.428 / 0.392 / 0.270). Two honest narrowings the skeptic insisted on and which are adopted
here: a rank correlation of +/-1.0 over **three** points is a restatement of monotonicity, not
independent evidence, and should not be quoted as though it were a correlation result; and only
the **k=1 vs k=8** endpoint gap is statistically defensible, since k=1 vs k=4 intervals overlap.
**[SUPERSEDED by Run 6]** — the second clause no longer holds: k=1 vs k=4 is defensible too,
at McNemar p = 1.26e-03. The first clause (rank correlation over three points is not
independent evidence) stands unchanged.

**Claim 3 — the harness gate. SURVIVES a stronger check than was asked for.** 500/500 argmax
agreement and a 1.75e-04 maximum log-probability difference both confirmed. The sharper
question is whether the gate is merely lucky: a perturbation that small could still flip a
decision where two choices are nearly tied. Computing the top1-minus-top2 margin for all 500
examples, the **tightest** margin is 3.405e-03 — about **19.4x** the observed perturbation —
and no example sits within even 2x of it. The gate has real headroom rather than getting away
with it.

Two further checks that close off ways the verification itself could have been hollow:
- **`native-injected` is not a soft gate.** It and every `mapped-k*` condition route through
  the identical `score_with_cache_provider -> build_cache -> forward_with_cache` scaffolding,
  differing only in which provider supplies the (K, V) — the target running itself, versus
  `source_cache()` + `apply_mapper()`. So it genuinely exercises the same injection path the
  mapped conditions use rather than bypassing it, and the inference "the other conditions
  measure the mapper, not the harness" is warranted.
- **The held-out R^2 is not biased by a train-mean artifact.** `kvt/ridge.py`'s `r2_score`
  computes its total sum of squares against the mean of *whichever set is being scored*, so
  held-out R^2 is measured against the held-out mean. Verified by inspection, independent of
  the tests that assert it.
- Under the flipped split the value collapse deepens as well: V held-out R^2 = **-0.7245**
  versus -0.6412 on the original split.

**Provenance gap, unevaluable.** The repository has no commits, so the relationship between the
Run 3 and Run 4 artifacts cannot be established from history — only from the files as they
stand. Committing (the commands are prepared for the operator) closes this.

## What fired / what is blocked

- **Two background runs were killed mid-flight (2026-08-24); the loss profile was asymmetric
  and worth knowing.** The 420-sequence dump had finished `source` (28/28 layers, 12 GB, valid)
  and was 85% through `target` — and lost **all** of the target, because `dump_kv` accumulates
  every layer for every sequence in memory and writes only after the loop completes. Roughly 75
  minutes of compute produced zero bytes for that half. Two consequences: (a) the function
  should write incrementally, or at minimum callers should know a kill at 99% is as costly as a
  kill at 1%; (b) it also holds ~12 GB resident at this scale, which is a second reason to
  stream. **What did NOT go wrong is worth stating too:** `KVDump.load` refuses an incomplete
  dump with `FileNotFoundError` rather than loading a truncated one, so there is no
  silent-corruption path — checked explicitly rather than assumed.


- **A published delta in this ledger was computed from ROUNDED inputs, in a section that
  claimed otherwise (caught by the final whole-branch review, 2026-08-24).** Run 5 originally
  read "+0.0251 held-out K R^2 (0.7065 vs 0.6814)" thirteen lines below a sentence stating
  "Recomputed from `results/mapper/.../r2.json` and `.../rope/r2.json`, not restated from
  stdout." Recomputed from the raw artifacts the difference is
  0.7065349557641573 - 0.6813557346883706 = 0.025179, i.e. **+0.0252**; the published +0.0251
  is exactly what subtracting the two 4-dp table entries gives. The magnitude is trivial and
  no conclusion changes. What is not trivial is that the ledger asserted a provenance the
  number did not have -- the same class as
  `docs/learnings/2026-08-24-hand-computed-test-fixtures-are-unrun-code.md`, arithmetic done
  by eye inside a document whose whole claim is that arithmetic is not done by eye. Corrected
  in place to +0.0252 and recorded here rather than silently amended.


- **Checked for held-out leakage in WP2's layer selection before running, and it is clean
  (2026-08-24).** `select_top_k` is driven by `r2_sel` in `scripts/fit_mapper.py`, which
  loads `results/probe/<pair>/r2_K_stripped_train.npy` and `r2_V_train.npy` -- the probe's
  **train** arrays, computed over sequences [0, 40) of the 50-sequence dump. Those sit inside
  every WP2 training prefix and are disjoint from the pre-registered held-out range
  [400, 420), so the selection never sees held-out data. WP2 therefore REUSES the existing
  selection rather than re-probing the 420-sequence dump. That is not only cheaper (~1.5 h
  saved) but strictly safer: `probe_r2` splits by FRACTION (`src.split(0.2)`), so a probe on
  a 420-sequence dump would hold out [336, 420) -- overlapping the WP2 held-out range -- and
  had selection been switched to the heldout arrays it would have leaked. Recorded because
  the leak-free property here is a consequence of which array the script happens to load,
  not of anything that enforces it.
- **Two latent hazards found while checking the above, neither blocking, both unfixed.**
  (a) `scripts/probe.py` hardcodes `data/kv/<pair>/{source,target}` and has no flag for an
  alternative dump root, so it cannot probe the 420-sequence dump at all. (b) `probe_r2` in
  `kvt/ridge.py` walks all source x target layer pairs with no cache bound; at 420 sequences
  that is 56 tensors per kind at 440 MB = 24.6 GB, and it loops over three kinds without
  evicting between them. Task 10 bounds `fit_mapper`/`mapper_r2` but NOT `probe_r2`. Neither
  blocks WP2 because WP2 does not re-probe, but a future probe at scale will hit both.


- **WP2 would have OOMed after a 100-minute dump, caught by arithmetic before the run rather
  than by a crash after it (2026-08-24).** `fit_mapper` and `mapper_r2` walk every target
  layer calling `KVDump.get`, which caches every tensor it loads and evicts nothing for the
  duration of the loop. The cache therefore grows to `2*L_s + 2*L_t` = up to 112 tensors.
  One tensor is 52.4 MB at 50 sequences but **440 MB at 420**, so the footprint goes from
  5.9 GB (why the committed Run 2 worked) to **49.3 GB on a 32 GB box** -- at k=1 just as
  much as at k=8, since the union of selected source layers spans all 28 either way.
  `clear_cache()` had been added but `grep` over the project confirms it had **no caller at
  all**, and the loop that needs it is inside `fit_mapper` where no script can reach it.
  Fixed by bounding the cache (LRU, cap 24 -> 10.6 GB peak) with the default left unlimited
  so the committed numbers keep reproducing. Recorded because the failure mode was a plan
  that was correct about *what* to run and silently wrong about whether it could run.


- **A load-bearing number in a locked decision was 4x too high, caught 2026-08-24 by a
  task reviewer and confirmed by recomputation.** `KVDump.get`'s docstring justified the
  "never clone, return by reference" decision with "~210 MB each at the real run size ...
  roughly 164 GB per run". Measured from `data/kv/qwen3-0.6b-to-1.7b/source` layer 0, a
  cached float32 tensor is **52.4 MB** (12,800 x 8 x 128), so 784 `get()` calls would copy
  **41.1 GB**, not 164 GB. The 210 MB figure is exactly the UN-STRIDED size (209.7 MB
  computed) -- it forgot that dumps are stride-4 subsampled. The decision itself is
  unchanged (41 GB of copying is still prohibitive), but the evidence for it was wrong, and
  it had been carried forward verbatim into the 2026-08-23 handoff brief's Locked decisions.
  Found because a reviewer noticed that this figure and a new 440 MB figure for 420
  sequences did not reconcile; the 440 MB figure is correct (440.4 MB computed). Recorded
  rather than quietly corrected, because a wrong number that survives into a handoff as
  justification is the failure mode this ledger exists to catch.


- **The fail-closed summarizer guard fired on a real case, not a hypothetical.** Invoking
  `scripts/summarize_hellaswag.py` while `mapped-k8.jsonl` existed but was still being written
  produced a refusal: "condition(s) ... produced zero scored examples: ['mapped-k8']; a run
  that scored nothing must not be summarized as a valid empty result." Without that guard the
  run would have emitted a NaN row for k=8 into the results table, which reads as data.
- **H1b was refuted — the prediction was the replicator's, not the paper's.** Stripping RoPE
  did not improve probe fit (diagonal-mean held-out 0.6284 stripped vs 0.6606 un-stripped).
  Recorded in Run 1 rather than quietly dropped.
- **A "below chance" claim was withdrawn and then its withdrawal was itself qualified.** See
  Run 4. The n=50 smoke said mapped-k8 fell below chance; n=500 says otherwise; and the
  discrepancy between the two runs is p ~= 0.025 on largely disjoint example sets, so neither
  "below chance" nor "just noise" is the right summary.
- **Blocked / not attempted:** attention-output cosine (`[STRETCH]`, Task 10) and the
  Qwen3-1.7B -> 4B pair (`[STRETCH]`) were not run. Length generalization, the property the
  paper's content-space mapping actually claims to buy, is **not tested here** — every
  evaluation is short-context, so Run 1's H1b result says nothing for or against that claim.
- **Not attempted by design:** prefill-latency measurement (meaningless on CPU), the MLP
  mapper, and GSM8K / ARC-C / MMLU. HellaSwag only, per the spec.
- **WP1's accuracy leg is now runnable, but has not been run (2026-08-24).** The gap was
  apparatus, not data: `scripts/eval_hellaswag.py` gained `--extra NAME=MAPPER_PATH[@MODEL]`
  (feeding `providers(extra=)`) and `scripts/summarize_hellaswag.py` gained `--compare A B`,
  which computes `kvt.stats.mcnemar_exact` over `kvt.stats.correctness_by_idx` and writes it
  into `summary.json`/`summary.md`. This closes the path to the McNemar p-value H-C1/H-C2 are
  stated in terms of. It cannot be run on this machine today: the Qwen3-4B checkpoint is not
  downloaded, and both `mapped-C-k1` and the `composed-BA-k1` middle-model prefill need it.
  Building the apparatus is not a result -- no WP1 number exists yet.
- **Mutation-tested the two Task-9 gates against `kvt/mapper.py`; both fired (2026-08-24).**
  Mutation 1 (`compose`'s `wk.append(...)` swapped to `wk.insert(0, ...)`, breaking the
  source-layer concatenation order) caused `tests/test_compose.py::
  test_closed_form_equals_operational_composition` to fail exactly where predicted, at
  `k_b >= 2` (`[1-2]` and `[2-3]` failed, `[1-1]` and `[2-1]` passed since a single-row
  vstack has no order to break). Mutation 2 (`_k_kind` returning `"K_stripped"` for both
  `space="content"` and `space="rope"`, so a rope-space fit silently reads un-rotated keys)
  caused `tests/test_space.py::test_fit_in_rope_space_uses_rotated_keys` to fail as
  predicted (`m_content.W_K[0]` and `m_rope.W_K[0]` came out `np.allclose`, i.e.
  indistinguishable, when they must differ). Both mutations were reverted from a
  pre-mutation copy and `kvt/mapper.py`'s MD5 (`e8dc48b777cfa0ac68247b167095cffc`) was
  confirmed identical before and after each. Neither gate is vacuous.

## The five questions, answered from this replication

1. **What makes a pair matched-KV, and why is it necessary but not sufficient?** Necessary:
   identical KV head count and per-head dimension, plus a shared tokenizer, so the source cache
   is shape-compatible with the target's attention. Verified here for Qwen3-0.6B -> 1.7B
   (both `n_kv=8`, `d_h=128`, 28 layers, identical vocab of 151,669). Not sufficient, and this
   run shows why directly: the `identity` control injects the raw, shape-compatible source
   cache and scores 0.362 against a native 0.598 and a chance floor of 0.25. Shape
   compatibility alone buys very little; the learned map is what does the work.
2. **Which component carries accuracy and which carries length generalization?** From our data
   we can only speak to the first, and our answer differs from the paper's at this scale: for
   us cross-layer selection *costs* accuracy monotonically (k=1 -> 8 drops acc_norm 0.548 ->
   0.312), because at 10,240 calibration tokens the extra features are interpolation rather
   than signal. The paper, at 128K tokens, finds the opposite. Length generalization we did not
   test at all — see the blocked list above.
3. **Why does floor normalization change the story?** Because raw retention on 4-way multiple
   choice cannot fall below ~25% however broken the cache is. mapped-k8 reads 0.522 raw
   retention, which sounds like partial degradation; floor-normalized it is 0.178. The
   arithmetic floor is exactly what conceals a bad result, which is the paper's Appendix F
   point observed from the measurement side.
4. **Why does R^2 fail to predict downstream retention, and what replaces it?** Our sharper
   finding: it depends entirely on *which* R^2. In-sample R^2 anti-predicts here (0.7783,
   0.8816, 0.9607 against accuracies 0.548, 0.498, 0.312), while held-out R^2 orders the three
   correctly. The paper's r = -0.20 for R^2 against retention is measured in-sample; a held-out
   split is the cheap fix, and the paper's proposed replacement, attention-output cosine, we
   did not measure (Task 10, `[STRETCH]`).
5. **Under what serving pattern is the prefill saving real?** Not answerable from this run — no
   latency was measured, and on CPU it would not transfer. From the paper: only where the
   source prefill was going to happen regardless (cascading, mid-conversation model switching,
   routing). It is not a general prefill accelerator, and the mapper is 4-12 GB per *directed*
   pair, so a family with n deployed sizes faces n(n-1) such artifacts.
