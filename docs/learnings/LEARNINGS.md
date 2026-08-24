# Learnings ledger — index

Pointers only. Every entry is a single dated fact with its own captured basis and a
read-only `re-verify:` line. Entries are immutable: a wrong entry is superseded by a new
dated entry carrying a `kills:` reference, never edited in place.

| date | entry | status | one-line |
|---|---|---|---|
| 2026-08-23 | [transformers5-rope-theta-moved](2026-08-23-transformers5-rope-theta-moved.md) | verified | `config.rope_theta` is gone in transformers 5 (now `rope_parameters["rope_theta"]`), and a wrong theta CANCELS in a strip/re-apply round trip, so no test catches it. |
| 2026-08-23 | [degenerate-parameter-gate-is-blind](2026-08-23-degenerate-parameter-gate-is-blind.md) | verified | A no-op gate set at the degenerate parameter value (k=1) cannot see the bug it exists for; reversing inference-time feature order left all 4 mapper tests passing. |
| 2026-08-23 | [rope-stripping-does-not-improve-short-context-fit](2026-08-23-rope-stripping-does-not-improve-short-context-fit.md) | refuted-assumption | Stripping RoPE made the probe fit slightly worse (0.6284 vs 0.6606), not better; its claimed payoff is length generalization, which short-context R^2 cannot test. |
| 2026-08-23 | [fail-open-survives-in-the-degenerate-case](2026-08-23-fail-open-survives-in-the-degenerate-case.md) | verified | Hardening a fail-open against missing and truncated inputs still let an EMPTY input through, producing a NaN results row with no exception. |
| 2026-08-24 | [in-sample-r2-tracks-p-over-n-on-pure-noise](2026-08-24-in-sample-r2-tracks-p-over-n-on-pure-noise.md) | verified | On pure noise, in-sample R^2 ~= p/n (0.8016 measured at p/n=0.800), so an in-sample R^2 is uninformative unless p/n is quoted alongside it. |
| 2026-08-24 | [heldout-collapse-is-not-distribution-shift](2026-08-24-heldout-collapse-is-not-distribution-shift.md) | verified | Inverting which chunk is held out is a cheap falsification of the distribution-shift alternative; here the collapse persisted (0.0766 vs 0.0984). |
| 2026-08-24 | [rng-choice-not-nested-across-n](2026-08-24-rng-choice-not-nested-across-n.md) | verified | `rng.choice(N, size=n)` is not nested across `n`: a 50-example smoke and a 500-example run shared only 9 items, so they are independent resamples, not a refinement. |
