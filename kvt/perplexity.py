"""Prefix-conditioned perplexity on WikiText-2 -- WP3's length-generalization instrument.

HellaSwag contexts are under 100 tokens, so no HellaSwag number can distinguish a mapper
that generalizes past its calibration length from one that does not. This measures NLL of
a FIXED 256-token continuation while varying only the length of the prefix delivered as a
mapped cache.

The continuation must be the SAME text at every prefix length P, or a trend across P
confounds "longer prefix" with "different text was scored" -- perplexity varies hugely
across text, so that confound alone can produce a spurious slope. `wikitext_windows` /
`_chunk` draw non-overlapping windows from the front of the token stream and are used to
build windows sized for the LARGEST prefix length via `fixed_continuation_windows`;
`slice_for_prefix` then takes the trailing `prefix_len + cont_len` tokens of each window,
so the final `cont_len` tokens -- the scored continuation -- are identical across every P.

Protocol (identical to hellaswag's A1): for a sliced window of length P + C the cache
covers ids[:P-1], the model is fed ids[P-1 : P+C], and NLL is summed over targets
ids[P : P+C]. The native condition forwards ids[:P+C] whole and scores the same targets,
so native-injected == native is an exact gate.
"""
import numpy as np
import torch

from kvt.cache import build_cache, forward_with_cache


def _chunk(ids: np.ndarray, n_windows: int, total_len: int) -> np.ndarray:
    """Non-overlapping windows from the front of a token stream. Deterministic: no RNG,
    so a re-run selects the same text."""
    need = n_windows * total_len
    if len(ids) < need:
        raise ValueError(
            f"need {need} tokens for {n_windows} windows of {total_len}, "
            f"only enough for {len(ids) // total_len}")
    return ids[:need].reshape(n_windows, total_len)


def wikitext_windows(tokenizer, n_windows: int, total_len: int) -> np.ndarray:
    """WikiText-2 (raw, test split) joined and chunked. Disjoint from FineWeb-Edu
    calibration by construction -- different corpus, so no leakage argument is needed."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
    return _chunk(ids, n_windows, total_len)


def fixed_continuation_windows(tokenizer, n_windows: int, prefix_lens, cont_len: int):
    """Windows sized for the LARGEST prefix, so every P scores the SAME continuation tokens.

    Returns (windows, max_prefix). windows is [n_windows, max_prefix + cont_len]. For a given
    P, slice `w[max_prefix - P : max_prefix + cont_len]`: the continuation occupies the final
    cont_len positions of every window regardless of P, so varying P varies ONLY how much
    preceding context is delivered -- which is the thing the experiment claims to vary.
    Chunking per-P instead would make each P score different text, and a trend across P would
    then confound prefix length with which text was scored.
    """
    max_prefix = int(max(prefix_lens))
    return wikitext_windows(tokenizer, n_windows, max_prefix + cont_len), max_prefix


def slice_for_prefix(window, max_prefix: int, prefix_len: int):
    """The sub-window that gives `prefix_len` tokens of context before the shared continuation."""
    if not (1 <= prefix_len <= max_prefix):
        raise ValueError(f"prefix_len {prefix_len} must be in [1, {max_prefix}]")
    return window[max_prefix - prefix_len:]


def _sum_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(-lp.gather(-1, targets[:, None]).sum().item())


@torch.no_grad()
def nll_native(model, ids: np.ndarray, prefix_len: int) -> tuple[float, int]:
    """Full forward over ids[:P+C]; returns (sum NLL over ids[P:], n_targets)."""
    ids = np.asarray(ids, dtype=np.int64)
    P = int(prefix_len)
    if not (1 <= P < len(ids)):
        raise ValueError(f"prefix_len {P} must be in [1, {len(ids)}) for this window")
    dev = next(model.parameters()).device
    t = torch.tensor(ids[None, :], device=dev)
    logits = model(input_ids=t).logits[0]                    # [P+C, V]
    return _sum_nll(logits[P - 1:-1], t[0, P:]), len(ids) - P


@torch.no_grad()
def nll_with_provider(target, ids: np.ndarray, prefix_len: int, provider) -> tuple[float, int]:
    """Cache covers ids[:P-1]; feed ids[P-1:]; score targets ids[P:]."""
    ids = np.asarray(ids, dtype=np.int64)
    P = int(prefix_len)
    if not (1 <= P < len(ids)):
        raise ValueError(f"prefix_len {P} must be in [1, {len(ids)}) for this window")
    dev = next(target.parameters()).device
    t = torch.tensor(ids[None, :], device=dev)
    kvs = provider(t[:, :P - 1])
    fed = t[:, P - 1:]
    logits = forward_with_cache(target, build_cache(kvs), fed, past_len=P - 1)[0]
    return _sum_nll(logits[:-1], t[0, P:]), len(ids) - P
