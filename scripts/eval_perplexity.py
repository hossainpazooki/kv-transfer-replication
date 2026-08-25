"""WP3: prefix-conditioned perplexity vs prefix length, content-space vs rope-space."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvt.hellaswag import providers
from kvt.mapper import Mapper
from kvt.pairs import PAIRS, check_matched_kv
from kvt.perplexity import fixed_continuation_windows, nll_native, nll_with_provider, slice_for_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--prefix-lens", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--n-windows", type=int, default=20)
    ap.add_argument("--cont-len", type=int, default=256)
    ap.add_argument("--content-mapper", default=None)
    ap.add_argument("--rope-mapper", default=None)
    args = ap.parse_args()

    pair = PAIRS[args.pair]
    tok = AutoTokenizer.from_pretrained(pair.source)
    src = AutoModelForCausalLM.from_pretrained(pair.source, dtype=torch.float32).eval()
    tgt = AutoModelForCausalLM.from_pretrained(pair.target, dtype=torch.float32).eval()
    check_matched_kv(src.config, tgt.config)

    extra = {}
    if args.content_mapper:
        extra["content-k1"] = (src, Mapper.load(args.content_mapper))
    if args.rope_mapper:
        extra["rope-k1"] = (src, Mapper.load(args.rope_mapper))
    prov = providers(src, tgt, mappers={}, extra=extra)
    conds = ["native-injected"] + (["identity"] if
             src.config.num_hidden_layers == tgt.config.num_hidden_layers else []) + list(extra)

    # Windows are drawn ONCE at the largest prefix length so every P's continuation is the
    # same text (see kvt/perplexity.py docstring) -- chunking per-P here would confound
    # prefix length with which text was scored.
    windows, max_prefix = fixed_continuation_windows(tok, args.n_windows, args.prefix_lens, args.cont_len)

    for P in args.prefix_lens:
        root = Path("results/perplexity") / args.pair / f"P{P}"
        root.mkdir(parents=True, exist_ok=True)
        for name in ["native"] + conds:
            recs = []
            for w, window in enumerate(windows):
                ids = slice_for_prefix(window, max_prefix, P)
                if name == "native":
                    s, n = nll_native(tgt, ids, P)
                else:
                    s, n = nll_with_provider(tgt, ids, P, prov[name])
                recs.append({
                    "window": w, "prefix_len": P, "sum_nll": s, "n_tokens": n,
                    "max_prefix": max_prefix, "shared_continuation": True,
                })
                print(f"P={P} {name} window {w+1}/{len(windows)} nll/tok={s/n:.4f}")
            (root / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in recs) + "\n")
    print("done")


if __name__ == "__main__":
    main()
