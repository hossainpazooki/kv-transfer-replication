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
    ap.add_argument("--out", default=None,
                    help="output dir; defaults to data/kv/<pair>/<which>. Pass an explicit path "
                         "when dumping a DIFFERENT sequence count: the default path holds the "
                         "50-sequence dumps that back Runs 1/2/4, and overwriting them would "
                         "silently change what every re-verify line in docs/ledger.md recomputes "
                         "(KVDump.split(0.2) holds out the last 20%, which moves with n_seqs).")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    pair = PAIRS[a.pair]
    check_matched_kv(AutoConfig.from_pretrained(pair.source), AutoConfig.from_pretrained(pair.target))
    tokens = Path(a.tokens or f"data/tokens/{a.pair}_n50_len1024_seed0.npy")
    seqs = np.load(tokens)
    out_dir = Path(a.out) if a.out else Path("data/kv") / a.pair / a.which
    if out_dir.exists() and (out_dir / "meta.json").exists():
        import json as _json
        prev = _json.loads((out_dir / "meta.json").read_text())
        if int(prev.get("n_seqs", -1)) != int(seqs.shape[0]):
            raise SystemExit(
                f"refusing to overwrite {out_dir}: it holds a {prev.get('n_seqs')}-sequence dump "
                f"and this run would write {seqs.shape[0]}. A different n_seqs changes what "
                f"KVDump.split() holds out, so every number recomputed from this directory would "
                f"change silently. Pass --out with a different path.")
    model = load_model(getattr(pair, a.which))
    t0 = time.time()
    dump_kv(model, seqs, a.stride, out_dir)
    print(f"wrote {out_dir} n_seqs={seqs.shape[0]} stride={a.stride} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
