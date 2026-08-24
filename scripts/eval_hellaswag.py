import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from kvt.hellaswag import byte_lengths, load_examples, providers, score_native, score_with_cache_provider
from kvt.mapper import Mapper
from kvt.models import assert_shared_tokenizer, load_model, load_tokenizer
from kvt.pairs import PAIRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", nargs="+", default=["1", "4", "8"])
    ap.add_argument("--conditions", nargs="+",
                    default=["native", "native-injected", "source", "identity", "mapped"])
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    pair = PAIRS[a.pair]
    tok_s, tok_t = load_tokenizer(pair.source), load_tokenizer(pair.target)
    assert_shared_tokenizer(tok_s, tok_t)
    source, target = load_model(pair.source), load_model(pair.target)
    mappers = {f"k{k}": Mapper.load(Path("mappers") / a.pair / f"k{k}") for k in a.k} if "mapped" in a.conditions else {}
    prov = providers(source, target, mappers)
    conds = [c for c in a.conditions if c != "mapped"] + [f"mapped-k{k}" for k in a.k if "mapped" in a.conditions]
    examples = load_examples(a.n, a.seed)
    out_dir = Path("results/hellaswag") / a.pair
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond in conds:
        path = out_dir / f"{cond}.jsonl"
        t0 = time.time()
        with path.open("w") as f:
            for ex in tqdm(examples, desc=cond, ascii=True):
                if cond == "native":
                    lp = score_native(target, tok_t, ex)
                elif cond == "source":
                    lp = score_native(source, tok_s, ex)
                else:
                    lp = score_with_cache_provider(target, tok_t, ex, prov[cond])
                f.write(json.dumps({"idx": ex["idx"], "gold": ex["gold"], "logprobs": lp,
                                    "nbytes": byte_lengths(ex["choices"])}) + "\n")
        print(f"{cond}: wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
