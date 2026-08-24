# Spec — small-scale replication of "Cross-Model KV Cache Transfer in LLM Families"

Paper: Heo et al. (NVIDIA), arXiv [2608.03893](https://arxiv.org/abs/2608.03893),
4 Aug 2026, cs.LG. *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form
Linear Mapping for Prefill Reuse.* No code release (checked 2026-08-23); the
replication is from the equations.

Status of this document: **spec, not results.** Nothing below is built.

## What we are replicating (the optional replication track)

Scoped to consumer hardware, matched-KV pair **Qwen3-0.6B → Qwen3-1.7B**
(stretch: Qwen3-1.7B → Qwen3-4B, which differs in layer count like the paper's pairs).

1. Dump K and V per layer/head for ~50 FineWeb-Edu sequences from both models.
2. Reproduce the Figure 2 heatmap with single-source OLS, on train **and held-out**
   tokens separately. The train/held-out gap is the thing the paper does not foreground.
3. Add top-k cross-layer selection at k ∈ {1, 4, 8} and confirm the R² lift.
4. Swap the mapped cache into the target and score HellaSwag with a small harness.

Step 2 alone is a defensible negative-control-style result: a reconstruction
metric that looks strong while downstream accuracy collapses is the
"fail-closed on unevaluable" failure mode seen from the measurement side.

## Paper facts pinned for the build (extracted from the arXiv HTML 2026-08-23)

Re-verify anything load-bearing against the PDF before quoting it in a ledger;
these came through a fetch-and-summarize pass, not a hand read.

**Matched-KV**: source and target share tokenizer, KV head count `n_kv`, and
per-head dim `d_h`. Layer count, hidden dim, attention-head count may differ.

**Probe (§2, Figure 2)**: per (source layer l', target layer l, head h),
single-source OLS `C̃_t = C_s W + b`, W ∈ R^{d_h×d_h}, X and Y centered, bias
included. Scored by R² averaged over target heads. Reported R² is **in-sample**
on the calibration tokens — no held-out set. Cache types: K with RoPE, K
stripped of RoPE, V. Best cell 0.81 (Qwen3 14B→32B, stripped K); single source
layer explains ~56% of key variance and ~32% of value variance; rises to 79% / 65%
with multiple source layers.

**Mapper (§3)**:
- Eq. 3: `K̂_t^{l,h} = X_K^l W_K^{l,h} + b_K^{l,h}`, same for V.
- Eq. 4: `W* = (XᵀX + λI)⁻¹ XᵀY`, λ = 0.01, float32 covariance / float64 analysis.
- Eq. 5: `X_K^l = [K̄_s^{l₁} ‖ … ‖ K̄_s^{l_k}]` — concatenation of **all KV heads**
  of the k selected source layers, so the feature width is `k · n_kv · d_h`.
- Top-k source layers per target layer chosen by head-averaged probe R²
  averaged over stripped-K and V. All heads in a target layer share the same
  selected source layers (this is what allows cross-head information flow).
  Same selection for K and V.
- RoPE: strip source RoPE, map in content space, re-apply target RoPE:
  `K̂_t = (K_s R_s(t)⁻¹ W_K + b_K) R_t(t)`. V is not rotated.
- Calibration: 500 FineWeb-Edu sequences × 1,024 tokens, stride-4 subsampled
  → ~128K token observations per target head. Fit time 47–87 min per pair on 8×H100.

**Appendix D storage formula**: weight params `= 2 · L_t · n_kv_t · (k · n_kv_s · d_h_s) · d_h_t`.
Check: Qwen3 14B→32B, k=8: `2·64·8·(8·8·128)·128 = 1,073,741,824` ≈ 1.07 B ✓ matches
Table 12 (4 GB at fp32). Other rows: 8B→32B k=12 1.61 B / 6 GB; Llama 8B→70B k=20
3.36 B / 12 GB; Ministral 3B→8B k=all(26) 1.85 B / 7 GB; 8B→14B k=12 1.01 B / 4 GB;
3B→14B k=20 1.68 B / 6 GB.

**Evaluation**: lm-evaluation-harness defaults, completion mode; acc_norm for
ARC-C/HellaSwag/MMLU. Qwen3 checkpoints are the post-trained ones (not `-Base`).
Floor normalization: `(acc − chance) / (target − chance)`, chance = 25% for 4-way.
k selected per pair by mean of ARC-C/HellaSwag/MMLU; GSM8K, CoQA, latency held out;
self-selection inflation bounded ≤ 2.49 pp (Appendix H).

**Headline numbers**: Tier 1 retention 97.6 / 87.5 / 76.2 / 72.8 % (Qwen3 14B→32B,
Qwen3 8B→32B, Ministral 3B→8B, Llama 8B→70B); Tier 2 44.2 / 41.6 % (Ministral
3B→14B, 8B→14B) → 14.7 / 11.1 % floor-normalized. GSM8K: 95.6 / 68.8 / 36.6 / 18.2 /
3.2 / 1.6. RoPE ablation on 14B→32B HellaSwag: full 80.70, −inference RoPE 75.39,
−all RoPE 80.73 (within noise). k=8→1: K R² 0.79→0.56, ARC-C 61.6→27.7.
Attention-output cosine vs HellaSwag retention r=+0.57 across 12 pair-evals;
calibration K R² r=−0.20. MLP mapper: −0.3 / −1.5 pp where ridge works, +24.3 /
+36.8 pp on the two Ministral failures.

## Model configs verified from HF `config.json` (2026-08-23)

| model | layers | hidden | attn heads | KV heads | d_h | rope_theta |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 28 | 1024 | 16 | 8 | 128 | 1e6 |
| Qwen/Qwen3-1.7B | 28 | 2048 | 16 | 8 | 128 | 1e6 |
| Qwen/Qwen3-4B   | 36 | 2560 | 32 | 8 | 128 | 1e6 |

0.6B→1.7B and 1.7B→4B are both matched-KV. All three tie word embeddings
(0.6B, 1.7B) or not (4B) — irrelevant to KV. `rope_scaling` is null on all three.

## Hardware this must run on

Hossain's local box: Windows 11, 16 cores, 32 GB RAM, **no NVIDIA GPU**, Python
3.14 only (no `uv`). Everything is CPU fp32. A GPU is a speed-up, not a requirement.

## Five questions the artifact should help answer cold

1. What makes a pair matched-KV, and why is it necessary but not sufficient?
2. Which mapper component carries accuracy (top-k cross-layer selection) and which
   carries length generalization (content-space / RoPE-stripped fit)?
3. Why does floor-normalized retention change the story on Ministral 3B→14B?
4. Why does R² fail to predict downstream retention, and what replaces it?
5. Under what serving pattern is a 2.7–25× prefill saving actually realized?
