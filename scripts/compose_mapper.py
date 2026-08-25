"""Build a composed mapper C = b(a(.)) and run gate H-C3 on it.

The gate matters because compose() and apply_mapper-twice are independent computations of
the same quantity. If they disagree the composed R^2 would just look bad, and a bad R^2
reads as a scientific result rather than a bug -- exactly the confusion this repo exists
to avoid.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kvt.mapper import Mapper, apply_mapper, compose


def gate_hc3(a: Mapper, b: Mapper, c: Mapper, n_tok: int = 32, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    L_s = int(max(int(x) for row in a.selected for x in row)) + 1
    kvs = [(torch.randn(1, a.n_kv, n_tok, a.d_h, generator=g),
            torch.randn(1, a.n_kv, n_tok, a.d_h, generator=g)) for _ in range(L_s)]
    pos = torch.arange(n_tok)
    two = apply_mapper(b, apply_mapper(a, kvs, pos), pos)
    one = apply_mapper(c, kvs, pos)
    dk = max(float(torch.max(torch.abs(x[0] - y[0]))) for x, y in zip(two, one))
    dv = max(float(torch.max(torch.abs(x[1] - y[1]))) for x, y in zip(two, one))
    return {"max_abs_diff_K": dk, "max_abs_diff_V": dv, "n_tokens": n_tok,
            "n_source_layers": L_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="mapper source->middle, without extension")
    ap.add_argument("--b", required=True, help="mapper middle->target, without extension")
    ap.add_argument("--out", required=True, help="output path, without extension")
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    a, b = Mapper.load(args.a), Mapper.load(args.b)
    c = compose(a, b)
    rep = gate_hc3(a, b, c)
    rep.update({"a": args.a, "b": args.b, "out": args.out, "k_a": a.k, "k_b": b.k, "k_c": c.k,
                "tol": args.tol})
    worst = max(rep["max_abs_diff_K"], rep["max_abs_diff_V"])
    rep["passed"] = bool(worst < args.tol)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".hc3.json").write_text(json.dumps(rep, indent=2))
    print(f"H-C3 max abs diff K={rep['max_abs_diff_K']:.3e} V={rep['max_abs_diff_V']:.3e} "
          f"tol={args.tol:.1e} -> {'PASS' if rep['passed'] else 'FAIL'}")
    if not rep["passed"]:
        raise SystemExit("H-C3 FAILED: closed-form and operational composition disagree. "
                         "Do not interpret any composed result until this is resolved.")
    c.save(args.out)
    print(f"wrote {args.out}.safetensors  k={c.k}  selected={c.selected.shape}")


if __name__ == "__main__":
    main()
