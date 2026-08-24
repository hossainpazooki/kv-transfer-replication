# kv-transfer-replication

CPU-scale replication of the replication track for *Cross-Model KV Cache Transfer in
LLM Families* (Heo et al., NVIDIA, arXiv 2608.03893) on the matched-KV pair
Qwen3-0.6B -> Qwen3-1.7B.

Four steps: dump K/V for both models on FineWeb-Edu; reproduce the single-source
OLS heatmap on train **and held-out** tokens; fit top-k cross-layer ridge mappers at
k in {1, 4, 8}; swap the mapped cache into the target and score HellaSwag against an
identity-mapper null control, reporting raw and floor-normalized retention.

Status tags: `[VALIDATED]` ran and was independently refuted-and-survived;
`[BASELINE]` ran, numbers in `docs/ledger.md`; `[STRETCH]` designed, not run;
`[FUTURE]` not designed.

| step | status |
|---|---|
| KV dumps | [BASELINE] |
| probe heatmap, train vs held-out | [BASELINE] |
| top-k mapper R^2 lift | [BASELINE] |
| HellaSwag swap-in + null control | [BASELINE] |
| attention-output cosine | [STRETCH] |
| 1.7B -> 4B pair | [STRETCH] |

## Run

    uv venv --python 3.12 .venv && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install -e ".[dev]"
    pytest
    python scripts/prepare_tokens.py --pair qwen3-0.6b-to-1.7b --n-seqs 50
    python scripts/dump_kv.py --pair qwen3-0.6b-to-1.7b --which source
    python scripts/dump_kv.py --pair qwen3-0.6b-to-1.7b --which target
    python scripts/probe.py --pair qwen3-0.6b-to-1.7b && python scripts/plot_probe.py --pair qwen3-0.6b-to-1.7b
    python scripts/fit_mapper.py --pair qwen3-0.6b-to-1.7b --k 1 4 8
    python scripts/eval_hellaswag.py --pair qwen3-0.6b-to-1.7b --n 500
    python scripts/summarize_hellaswag.py --pair qwen3-0.6b-to-1.7b

Every number in `docs/ledger.md` is recomputed from `results/` by a summarize script.
