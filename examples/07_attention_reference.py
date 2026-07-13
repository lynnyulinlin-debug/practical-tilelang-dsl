"""Chapter 7 focused demo: standard attention only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_record_template import set_benchmark_record
from examples.ch07_attention_core import attention_reference, benchmark, make_attention_inputs


def main(batch: int = 1, seq_len: int = 32, dim: int = 64, runs: int = 20) -> None:
    np.random.seed(0)
    q, k, v = make_attention_inputs(batch, seq_len, dim)
    out, elapsed_ms = benchmark(attention_reference, q, k, v, runs=runs)

    print("Chapter 7: Attention Reference")
    print("=" * 72)
    print(f"Shape: B={batch}, S={seq_len}, D={dim}")
    print(f"Latency: {elapsed_ms:.4f} ms per run")
    print(f"Output norm: {np.linalg.norm(out):.4f}")
    print()
    print("Interpretation:")
    print("- This version materializes the full scores/probs path.")
    print("- It is the correctness baseline for FlashAttention-style rewrites.")
    print()

    record = set_benchmark_record(
        scenario="chapter 7 attention reference",
        operator="Attention",
        platform="CPU",
        target="numpy",
        dtype="float32",
        shape=f"B={batch},S={seq_len},D={dim}",
        baseline="standard attention",
        optimization="none",
    )
    print("Benchmark record skeleton:")
    for key in ["scenario", "operator", "platform", "target", "dtype", "shape", "baseline", "optimization"]:
        print(f"  {key}: {record[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 7 attention reference demo")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.batch, args.seq_len, args.dim, args.runs)
