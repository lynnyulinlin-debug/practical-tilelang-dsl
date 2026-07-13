"""Chapter 6 focused demo: naive GEMM only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_record_template import set_benchmark_record
from examples.ch06_basic_ops_core import benchmark, gemm_naive, make_gemm_inputs


def main(m: int = 64, n: int = 64, k: int = 64, runs: int = 1) -> None:
    np.random.seed(0)
    a, b = make_gemm_inputs(m, n, k)
    ref = a @ b
    out, elapsed_ms = benchmark(gemm_naive, a, b, runs=runs)

    print("Chapter 6: Naive GEMM")
    print("=" * 72)
    print(f"Shape: M={m}, N={n}, K={k}")
    print(f"Latency: {elapsed_ms:.4f} ms per run")
    print(f"Max error: {np.max(np.abs(out - ref)):.2e}")
    print()
    print("Interpretation:")
    print("- This version exposes repeated global-memory traffic.")
    print("- It is a correctness baseline before tile reuse.")
    print()

    record = set_benchmark_record(
        scenario="chapter 6 naive gemm",
        operator="GEMM",
        platform="CPU",
        target="numpy",
        dtype="float32",
        shape=f"M={m},N={n},K={k}",
        baseline="naive GEMM",
        optimization="none",
    )
    print("Benchmark record skeleton:")
    for key in ["scenario", "operator", "platform", "target", "dtype", "shape", "baseline", "optimization"]:
        print(f"  {key}: {record[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 6 naive GEMM demo")
    parser.add_argument("--m", type=int, default=64, help="GEMM M dimension")
    parser.add_argument("--n", type=int, default=64, help="GEMM N dimension")
    parser.add_argument("--k", type=int, default=64, help="GEMM K dimension")
    parser.add_argument("--runs", type=int, default=1, help="Benchmark runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.m, args.n, args.k, args.runs)
