import torch
from transformers import DynamicCache


def get_layer_kv(cache, layer_idx: int):
    if hasattr(cache, "layers"):                      # transformers >= 4.56 layered cache
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def cache_to_arrays(cache, n_layers: int):
    out = []
    for l in range(n_layers):
        k, v = get_layer_kv(cache, l)
        out.append((k.detach().clone(), v.detach().clone()))
    return out


def build_cache(kvs, batch: int | None = None) -> DynamicCache:
    """kvs: list of (K[B,n_kv,T,d_h], V). If batch is given, expand a B=1 cache to that batch."""
    cache = DynamicCache()
    for l, (k, v) in enumerate(kvs):
        if batch is not None:
            k = k.expand(batch, -1, -1, -1).contiguous()
            v = v.expand(batch, -1, -1, -1).contiguous()
        cache.update(k, v, l)
    return cache


@torch.no_grad()
def forward_with_cache(model, cache: DynamicCache, input_ids: torch.Tensor, past_len: int) -> torch.Tensor:
    """Run `input_ids` at positions past_len..past_len+C-1 on top of an injected cache.
    Right-padding is safe under the causal mask; the caller ignores pad positions."""
    B, C = input_ids.shape
    dev = input_ids.device
    attn = torch.ones(B, past_len + C, dtype=torch.long, device=dev)
    pos = torch.arange(past_len, past_len + C, device=dev)
    out = model(input_ids=input_ids, past_key_values=cache, attention_mask=attn,
                position_ids=pos[None].expand(B, -1), cache_position=pos, use_cache=True)
    return out.logits
