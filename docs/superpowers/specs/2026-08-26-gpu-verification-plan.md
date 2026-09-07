# GPU verification plan — the CPU findings, re-tested at the scale they were about

2026-08-26. Status: **PROPOSAL.** Nothing here has run. Every claim below is tagged with what
the CPU work established and what a GPU run would add; a GPU run that merely repeats the CPU
number faster is not on this list.

The CPU replication produced findings *about* a regime it could not reach. The central one —
in-sample R² anti-predicts accuracy — was measured at 10,240 calibration tokens because CPU
forced it, then explicitly declared to say nothing about the paper's ~128K-token regime. GPU
time is worth spending exactly where it changes what a claim is *about*, and nowhere else.

## 0. Before anything: inventory, and one methodological blocker

**Inventory.** Run on every machine and paste the output into `docs/ledger.md` under a
"GPU inventory" heading before any run is designed against it:

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

The plan below is parameterised by tier, because the tiers can run different things:

| tier | typical card | fits in bf16 | can run |
|---|---|---|---|
| A | one 24 GB consumer (3090/4090) | 0.6B, 1.7B, 4B, 8B (16 GB) — one model resident at a time for 8B | Phases 1–3, 5 |
| B | one 80 GB (A100/H100) | everything up to 32B (64 GB) | all phases incl. the paper's own pair |
| C | several of B | the same, in parallel; seeds and pairs fan out | all, with error bars |

**The blocker: the injection gate is defined as bit-exact, and bf16 is not.** Every CPU result
rests on `native-injected == native` — 500/500 identical argmax decisions, max log-prob
difference 1.75e-04 in fp32. On a GPU in bf16, with non-deterministic reductions, the two
paths will *not* be bit-identical, and a gate that fails on every run is as useless as one
that never fires. So Phase 0 exists to **re-characterise the gate before it guards anything**.
This is not optional and it is the first thing that runs on each tier.

## 1. Non-negotiables carried over, plus two learned today

Carried over unchanged (see `docs/handoff/`): held-out by sequence; the no-clone `KVDump`
contract; the LRU cache; every number recomputed by a summarize script; status tags;
Bonferroni fixed in advance; the `--tag` overwrite guards.

**New, from the 2026-08-26 verification pass on the team page:**

1. **Pre-registration must be provable from git, not from file mtimes and not from the
   transcript.** Today's Run 8 rule was written 11 seconds before launch — genuinely prior,
   but provable only from the session log, because `docs/ledger.md` is one continuously
   edited file whose mtime is always "now". Rule: each run gets its own
   `docs/prereg/<run-id>.md` (hypotheses, statistic, threshold, stop conditions), **committed
   before launch**, and the launch command takes `--prereg-sha <commit>`; that SHA is stamped
   into every result record, and the summarizer refuses any result whose SHA is not an
   ancestor of HEAD or whose commit timestamp is not earlier than the artifact's. That makes
   the timeline in the team page's Figure 2 a check that can actually run.
2. **No smoke run on the same conditions before the rule is fixed.** Today's timing smoke put
   2 windows of P=2048 content-vs-rope output on disk before the rule existed. Timing smokes
   run on `native` only, or on a pair not under test.

## 2. The claims, and what a GPU changes about each

| # | CPU claim (established) | What it is *about* at CPU scale | What GPU adds | Phase |
|---|---|---|---|---|
| C1 | In-sample R² anti-predicts accuracy; held-out predicts | 10,240 tokens, p/n up to 0.80 | The paper's regime: does k=8 *recover* at p/n ≈ 0.06? Is the anti-correlation a small-calibration artifact or a property of the metric? | 2 |
| C2 | Calibration size is load-bearing | one point (n=50) | The full learning curve to 400K tokens, three seeds | 2 |
| C3 | Content-space mapping generalises past calibration length (Run 8, PARTIAL) | one pair, k=1, 32 windows, P ≤ 2048, perplexity only | P to 16384, 128 windows, second pair, k ∈ {1,4}, a long-context *task* not just perplexity | 1 |
| C4 | Composition B∘A vs direct C (apparatus only; never run) | nothing — 4B never loaded | The actual n−1 vs n(n−1) question, plus a 3-hop chain to 8B | 3 |
| C5 | Injection gate exact | fp32, CPU | Whether the gate survives bf16 at all, and what tolerance it needs | 0 |
| C6 | The paper's headline numbers (14B→32B) | never attempted | Direct replication of the paper's own pair | 4 (tier B) |
| C7 | Which calibration-only quantity predicts transfer across pairs | one pair, so no correlation possible | 6–12 directed pairs × 3 k | 5 |

## 3. Code changes required (small, and all before Phase 0)

The package is already device-agnostic — `kvt/models.py::device()` picks CUDA — but four
things assume CPU:

| change | where | why |
|---|---|---|
| `--dtype {fp32,bf16}` flag threaded to `load_model` | `kvt/models.py`, every script | bf16 is the reason to use a GPU; fp32 stays the reference for Phase 0's comparison |
| Batched forward in `dump_kv` | `kvt/data.py:40` (batch=1 loop) | at batch 1 a GPU is bandwidth-idle; batch 16–32 is where the 50–150× lives |
| **Streaming writes** in `dump_kv` | `kvt/data.py` | it currently buffers every layer of every sequence and writes at the end — a kill at 99% loses everything (`docs/learnings/2026-08-25-buffered-writers-have-a-flat-loss-profile.md`). Write per-layer shards every N sequences. |
| Optional torch-float64 ridge on tier B | `kvt/ridge.py` | the Gram is accumulated in numpy float64 on CPU. Consumer cards run fp64 at 1:64 — keep it on CPU there. A100/H100 run fp64 at 1:2 — move it. At n = 400K tokens the CPU solve is hours. |
| `--prereg-sha` stamping + summarizer ancestry check | all `eval_*`, `summarize_*` | §1 |
| Gate tolerance as a recorded parameter, not a constant | `scripts/summarize_hellaswag.py`, `summarize_perplexity.py` | Phase 0 sets it; every later run reports against it |

Estimated: one day of work, all offline-testable against the tiny fixtures, before any GPU is
touched.

## 4. Phases, in order, each ending at a gate

### Phase 0 — gate re-characterisation (every tier; ~1 GPU-hour)

Run `native` and `native-injected` on HellaSwag n=500 and on the perplexity windows, in
**fp32 on GPU** and in **bf16**, with `torch.use_deterministic_algorithms(True)` on and off.
Record, per condition: argmax agreement out of 500, max and p99 |Δ log-prob|, and whether
two identical runs agree with each other.

Pre-registered decision rule: the gate for all later phases is
`argmax agreement ≥ 498/500 AND p99 |Δ logprob| ≤ 10 × the bf16 native-vs-native run-to-run
value`, with both thresholds written down *from the fp32-GPU and bf16 measurements*, not
chosen after seeing mapped conditions. If bf16 cannot meet even 495/500 against itself, all
later phases run in fp32 and the throughput estimates below halve.

**Stop condition:** if fp32-on-GPU does not reproduce the CPU 500/500, something is wrong
with the port, not with bf16 — halt and find it.

### Phase 1 — length generalisation at scale (C3; tier A ≈ 3 GPU-hours, B ≈ 1)

Extends Run 8, which met its rule at P=2048 on one pair with 32 windows.

- Pairs: 0.6B→1.7B (existing mappers) and 1.7B→4B (new, fit once).
- k ∈ {1, 4}: k=4 is where the CPU work saw accuracy fall; does length interact with it?
- P ∈ {1024, 2048, 4096, 8192, 16384}, 128 windows, shared continuation (already built).
- Three seeds of calibration draw → three mapper fits per (pair, k), so the curve has error
  bars instead of being n=1.
- **A task, not only perplexity:** one long-context QA set where the answer depends on a
  token beyond position 1024 (a needle-style set built from the same WikiText windows is
  enough; nothing external needed). Perplexity generalising and *retrieval* generalising are
  different claims.

Pre-registered statistic, unchanged from Run 8: the difference-of-differences
`d(P) − d(1024)` paired across windows, threshold Bonferroni 0.05 / (number of P × pairs × k).
Leg 2: content-vs-rope gap ≥ 5 pp at the largest P.

**What would kill C3:** the gap stops growing past 4096, or reverses on the second pair, or
grows in perplexity but not in the task. Each is a result.

### Phase 2 — the learning curve into the paper's regime (C1, C2; tier A ≈ 6 GPU-hours + CPU solves, B ≈ 2)

This is the one that decides whether the repository's central finding is about *the metric*
or about *small calibration*.

- Pair 0.6B→1.7B. Calibration n ∈ {50, 100, 200, 400, 800, 1600} sequences × 1024 tokens,
  stride 4 → 12.8K to 410K token rows. At k=8 that spans p/n = 0.80 down to **0.02**, passing
  through the paper's 0.064.
- Fixed held-out set of 100 sequences drawn *after* the largest training prefix, never
  reused for selection (the leakage argument in `docs/ledger.md` § "What fired").
- k ∈ {1, 4, 8, 20}. k=20 is the paper's Table value; it was unaffordable on CPU.
- Every point recomputed from ONE dump (the fp16-ULP finding), three seeds of the
  sequence draw.
- HellaSwag n=2000 (not 500) so paired tests can see 2 pp; McNemar on every adjacent k.

Pre-registered: **H-L2** — the p/n at which k=8 held-out R² crosses k=1 lies in [0.05, 0.2],
and HellaSwag accuracy crosses at the same point within one grid step. **H-L3** — no crossing
by p/n=0.02; then cross-layer selection costs accuracy on this pair regardless of calibration
and the paper's k result does not transfer. Both are reportable.

Compute note: dumps are minutes on GPU. The **fits** are the cost — the 8192×8192 Gram at
410K rows is ~40× the CPU work of today's 13-minute fit. Tier A: leave it on CPU overnight.
Tier B: torch-float64 on the card, ~20 minutes per k.

### Phase 3 — composition (C4; tier A ≈ 8 GPU-hours, B ≈ 3)

Never run; the apparatus is complete and reviewed.

- A = 0.6B→1.7B, B = 1.7B→4B, C = 0.6B→4B direct, all at k ∈ {1, 4}, calibration at the
  Phase-2 point where k=4 was safe (so composition is not confounded with interpolation).
- H-C3 gate first (closed-form vs two-call) on real dumps, not synthetic matrices.
- HellaSwag n=2000 on the 4B: `native-4b`, `native-injected-4b`, `source-0.6b`, `mapped-C`,
  `composed-BA`, `mapped-B-from-1.7B`. McNemar `composed-BA` vs `mapped-C`.
- **3-hop stretch (tier A can do it, 8B fits):** 0.6B→1.7B→4B→8B composed vs 0.6B→8B direct.
  If the 2-hop holds and the 3-hop fails, that bounds how far a chain carries.

Pre-registered H-C1 / H-C2 as in `docs/ledger.md`, threshold 3 pp and McNemar 0.01.

### Phase 4 — the paper's own pair (C6; tier B only, ≈ 6 GPU-hours)

Qwen3-14B→32B, the paper's headline pair. Reproduce its Figure 2 heatmap and its k-selection
result at *its* calibration size. This is the only phase that tests the paper directly rather
than the replication's extrapolation of it. If the CPU finding is a small-calibration artifact
(Phase 2, H-L2), this is where the paper should look right; if it is a property of the metric
(H-L3), in-sample R² should still mis-order k here.

### Phase 5 — the transfer predictor (C7; tier A ≈ 10 GPU-hours, C ≈ 2 wall-clock)

Needs Phases 1–3's pairs. Six directed pairs among {0.6B, 1.7B, 4B} plus six more with 8B
= 12 pairs × 3 k = 36 points. For each: held-out R², in-sample R², attention-output cosine
(Task 10 brief, still unbuilt), selection margin, floor-normalised retention. Pre-registered:
in-sample R² rank-correlates ≤ 0 with retention across pairs; held-out ≥ +0.5; cosine
measured with no prior. **Kill:** held-out orders k within pairs but not pairs against each
other — then cosine is the only survivor and the paper was right for the wrong reason.

## 5. Budget

| phase | tier A (one 24 GB) | tier B (one 80 GB) | notes |
|---|---|---|---|
| 0 | 1 h | 0.5 h | mandatory first |
| 1 | 3 h | 1 h | P=16384 dominates |
| 2 | 6 h GPU + overnight CPU fits | 2 h | fits are the long pole |
| 3 | 8 h | 3 h | 4B/8B HellaSwag at n=2000 |
| 4 | — | 6 h | 32B in bf16 is 64 GB |
| 5 | 10 h | 3 h | mostly reuses 1–3 |
| **total** | **~28 GPU-h + CPU** | **~16 GPU-h** | tier C: ~1 day wall-clock with seeds fanned out |

These are estimates from the CPU wall-clocks and published throughput; the plan's first real
measurement replaces them. None of these numbers goes into `docs/ledger.md` until measured.

## 6. What this plan will not do

- It will not replace the CPU results; it re-tests them at the scale they were about.
- It will not measure prefill latency. That is a serving-systems claim with its own harness,
  and mixing it into an accuracy replication is how the slide deck ended up citing this repo
  for a latency number it never measured.
- It will not start Phase 5 before Phases 1–3 have results, and it will not start any phase
  before Phase 0 has set the gate.

## 7. First three commands on the first GPU machine

```
# 1. inventory into the ledger
nvidia-smi --query-gpu=name,memory.total --format=csv >> docs/ledger.md
# 2. after the §3 code changes land and are committed:
git rev-parse HEAD                      # this is the --prereg-sha for Phase 0
# 3. Phase 0, fp32 first, bf16 second, on the existing 0.6B->1.7B mappers
python scripts/eval_hellaswag.py --pair qwen3-0.6b-to-1.7b --n 500 --dtype fp32 --tag gpu-fp32 --prereg-sha <sha>
python scripts/eval_hellaswag.py --pair qwen3-0.6b-to-1.7b --n 500 --dtype bf16 --tag gpu-bf16 --prereg-sha <sha>
```
