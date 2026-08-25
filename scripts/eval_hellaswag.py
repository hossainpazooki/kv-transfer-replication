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


def parse_extra(specs: list[str]) -> list[tuple[str, str, str]]:
    """Parse repeated --extra NAME=MAPPER_PATH[@MODEL] specs into (name, mapper_path, which_model).

    which_model is "source" (default) or "middle" -- which loaded model prefills that
    condition. Refuses: a spec with no '=', an empty name, an unknown @MODEL value, and a
    duplicate name across the list. Does NOT check @middle against --middle-model; that
    check needs the parsed args and lives in main()."""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"invalid --extra spec {spec!r}: expected NAME=MAPPER_PATH[@MODEL]")
        name, rest = spec.split("=", 1)
        if not name:
            raise ValueError(f"invalid --extra spec {spec!r}: NAME must not be empty")
        if "@" in rest:
            mapper_path, which = rest.rsplit("@", 1)
        else:
            mapper_path, which = rest, "source"
        if which not in ("source", "middle"):
            raise ValueError(
                f"invalid --extra spec {spec!r}: unknown @MODEL {which!r}, "
                "expected 'source' or 'middle'")
        if not mapper_path:
            raise ValueError(f"invalid --extra spec {spec!r}: MAPPER_PATH must not be empty")
        if name in seen:
            raise ValueError(f"duplicate --extra condition name {name!r}")
        seen.add(name)
        out.append((name, mapper_path, which))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", nargs="+", default=["1", "4", "8"])
    ap.add_argument("--conditions", nargs="+",
                    default=["native", "native-injected", "source", "identity", "mapped"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--extra", nargs="+", default=[], metavar="NAME=MAPPER_PATH[@MODEL]",
                    help="register additional conditions providers() supports via extra=. "
                         "@MODEL is 'source' (default) or 'middle'; 'middle' requires "
                         "--middle-model. Repeatable/space-separated, e.g. "
                         "--extra composed-BA-k1=mappers/p/composed-k1@middle")
    ap.add_argument("--middle-model", default=None,
                    help="HF id or local path for the model that prefills @middle conditions. "
                         "Only loaded if some --extra spec asks for it.")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    try:
        extra_specs = parse_extra(a.extra)
    except ValueError as e:
        raise SystemExit(str(e))
    needs_middle = any(which == "middle" for _, _, which in extra_specs)
    if needs_middle and not a.middle_model:
        raise SystemExit(
            "--extra specifies @middle for at least one condition, but --middle-model was not given")
    pair = PAIRS[a.pair]
    tok_s, tok_t = load_tokenizer(pair.source), load_tokenizer(pair.target)
    assert_shared_tokenizer(tok_s, tok_t)
    source, target = load_model(pair.source), load_model(pair.target)
    middle = load_model(a.middle_model) if needs_middle else None
    mappers = {f"k{k}": Mapper.load(Path("mappers") / a.pair / f"k{k}") for k in a.k} if "mapped" in a.conditions else {}
    extra = {name: ((middle if which == "middle" else source), Mapper.load(Path(mapper_path)))
             for name, mapper_path, which in extra_specs}
    prov = providers(source, target, mappers, extra=extra)
    conds = ([c for c in a.conditions if c != "mapped"] + [f"mapped-k{k}" for k in a.k if "mapped" in a.conditions]
             + [name for name, _, _ in extra_specs])
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
