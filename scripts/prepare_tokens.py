"""Tokenize FineWeb-Edu once with the shared tokenizer so both models see identical sequences."""
import argparse
from pathlib import Path

import numpy as np

from kvt.models import assert_shared_tokenizer, load_tokenizer
from kvt.data import iter_fineweb_sequences
from kvt.pairs import PAIRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--n-seqs", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pair = PAIRS[a.pair]
    tok_s, tok_t = load_tokenizer(pair.source), load_tokenizer(pair.target)
    assert_shared_tokenizer(tok_s, tok_t)
    seqs = iter_fineweb_sequences(tok_s, a.n_seqs, a.seq_len, a.seed)
    out = Path("data/tokens") / f"{a.pair}_n{a.n_seqs}_len{a.seq_len}_seed{a.seed}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, seqs)
    print(f"wrote {out} shape={seqs.shape}")


if __name__ == "__main__":
    main()
