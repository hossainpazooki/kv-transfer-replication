import argparse
import time
from pathlib import Path

import numpy as np
import torch

from kvt.data import dump_kv
from kvt.models import load_model
from kvt.pairs import PAIRS, check_matched_kv
from transformers import AutoConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--which", required=True, choices=["source", "target"])
    ap.add_argument("--tokens", default=None, help="defaults to data/tokens/<pair>_n50_len1024_seed0.npy")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    pair = PAIRS[a.pair]
    check_matched_kv(AutoConfig.from_pretrained(pair.source), AutoConfig.from_pretrained(pair.target))
    tokens = Path(a.tokens or f"data/tokens/{a.pair}_n50_len1024_seed0.npy")
    seqs = np.load(tokens)
    model = load_model(getattr(pair, a.which))
    out_dir = Path("data/kv") / a.pair / a.which
    t0 = time.time()
    dump_kv(model, seqs, a.stride, out_dir)
    print(f"wrote {out_dir} n_seqs={seqs.shape[0]} stride={a.stride} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
