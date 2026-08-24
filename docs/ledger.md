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

Caveat carried: dumps are stored float16. Re-roping a stripped key recovers the original to
6.1e-05 max error, i.e. ~5e-4 relative, which caps attainable R^2 near 0.9999997 — far above
any value reported here, so it does not affect these conclusions.

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
