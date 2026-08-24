"""HellaSwag log-likelihood scoring under native, injected, identity and mapped caches.
Protocol A1: source prefills c[0..n-2]; target processes c[n-1] + ending on top of the injected cache."""
import math
import re

import numpy as np
import torch

from kvt.cache import build_cache, cache_to_arrays, forward_with_cache
from kvt.mapper import apply_mapper


def preprocess(text: str) -> str:
    """Copied from lm-evaluation-harness hellaswag utils."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_examples(n: int, seed: int = 0) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    out = []
    for i in sorted(idx.tolist()):
        d = ds[i]
        query = preprocess(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
        out.append({"idx": i, "query": query, "choices": [preprocess(e) for e in d["endings"]], "gold": int(d["label"])})
    return out


def encode(tok, query: str, choice: str):
    cont = " " + choice if not choice.startswith(" ") else choice
    whole = tok(query + cont, add_special_tokens=False)["input_ids"]
    n_ctx = len(tok(query, add_special_tokens=False)["input_ids"])
    return whole, n_ctx


def byte_lengths(choices) -> list[int]:
    return [len(c.encode("utf-8")) for c in choices]


def _pad(seqs: list[list[int]], dev) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [0] * (L - len(s)) for s in seqs], dtype=torch.long, device=dev)


def _cont_logprob(logits: torch.Tensor, ids: list[int], first_pred: int) -> float:
    """logits[t] predicts ids[first_pred + t]; sum log-probs over positions first_pred..len(ids)-1."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for t in range(len(ids) - first_pred):
        total += lp[t, ids[first_pred + t]].item()
    return total


@torch.no_grad()
def score_native(model, tok, ex: dict) -> list[float]:
    dev = next(model.parameters()).device
    encs = [encode(tok, ex["query"], c) for c in ex["choices"]]
    logits = model(input_ids=_pad([w for w, _ in encs], dev)).logits            # [4, L, V]
    out = []
    for i, (whole, n_ctx) in enumerate(encs):
        out.append(_cont_logprob(logits[i, n_ctx - 1 : len(whole) - 1], whole, n_ctx))
    return out


@torch.no_grad()
def score_with_cache_provider(target, tok, ex: dict, provider) -> list[float]:
    dev = next(target.parameters()).device
    encs = [encode(tok, ex["query"], c) for c in ex["choices"]]
    n_ctx = encs[0][1]
    P = n_ctx - 1
    prefix = _pad([w[:P] for w, _ in encs], dev)                                 # [4, P], same length by construction
    kvs = provider(prefix)                                                        # target-layout cache for prefix
    fed = _pad([w[P:] for w, _ in encs], dev)                                     # c[n-1] + ending (+ pad)
    logits = forward_with_cache(target, build_cache(kvs), fed, past_len=P)       # [4, C, V]
    out = []
    for i, (whole, _) in enumerate(encs):
        out.append(_cont_logprob(logits[i, : len(whole) - P - 1], whole[P:], 1))
    return out


def providers(source, target, mappers: dict) -> dict:
    """mappers: {"k1": Mapper, ...}. Every provider maps prefix ids [B,P] -> list[(K,V)] in target layout."""
    n_s, n_t = source.config.num_hidden_layers, target.config.num_hidden_layers

    def native_injected(prefix):
        return cache_to_arrays(target(input_ids=prefix, use_cache=True).past_key_values, n_t)

    def source_cache(prefix):
        return cache_to_arrays(source(input_ids=prefix, use_cache=True).past_key_values, n_s)

    def identity(prefix):
        if n_s != n_t:
            raise ValueError(
                f"identity control needs equal layer counts (A4); use mapped conditions otherwise "
                f"(source has {n_s} layers, target has {n_t} layers)"
            )
        return source_cache(prefix)

    out = {"native-injected": native_injected, "identity": identity}
    for name, m in mappers.items():
        out[f"mapped-{name}"] = (lambda mm: lambda prefix: apply_mapper(
            mm, source_cache(prefix), torch.arange(prefix.shape[1], device=prefix.device)))(m)
    return out


def _wilson(p: float, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    c = p + z * z / (2 * n); d = 1 + z * z / n; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def summarize_records(records: dict[str, list[dict]], chance: float) -> dict:
    out = {}
    for cond, recs in records.items():
        acc = float(np.mean([int(np.argmax(r["logprobs"])) == r["gold"] for r in recs]))
        acc_norm = float(np.mean([int(np.argmax(np.asarray(r["logprobs"]) / np.asarray(r["nbytes"]))) == r["gold"]
                                  for r in recs]))
        out[cond] = {"n": len(recs), "acc": acc, "acc_norm": acc_norm, "ci95": _wilson(acc_norm, len(recs))}
    if "native" in out:
        t = out["native"]["acc_norm"]
        for cond, s in out.items():
            s["retention_raw"] = s["acc_norm"] / t if t else float("nan")
            s["retention_floor_norm"] = (s["acc_norm"] - chance) / (t - chance) if t > chance else float("nan")
    return out
