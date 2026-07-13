"""Chapter 6 focused demo: blocked GEMM only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_record_template import set_benchmark_record
from examples.ch06_basic_ops_core import benchmark, gemm_blocked, make_gemm_inputs


def main(m: int = 64, n: int = 64, k: int = 64, bm: int = 32, bn: int = 32, bk: int = 16, runs: int = 1) -> None:
    np.random.seed(0)
    a, b = make_gemm_inputs(m, n, k)
    ref = a @ b
    out, elapsed_ms = benchmark(gemm_blocked, a, b, bm, bn, bk, runs=runs)

    print("Chapter 6: Blocked GEMM")
    print("=" * 72)
    print(f"Shape: M={m}, N={n}, K={k}")
    print(f"Tile: bm={bm}, bn={bn}, bk={bk}")
    print(f"Latency: {elapsed_ms:.4f} ms per run")
    print(f"Max error: {np.max(np.abs(out - ref)):.2e}")
    print()
    print("Interpretation:")
    print("- This version shows how tile reuse changes the cost structure.")
    print("- It is the bridge from correctness to reuse-aware optimization.")
    print()

    record = set_benchmark_record(
        scenario="chapter 6 blocked gemm",
        operator="GEMM",
        platform="CPU",
        target="numpy",
        dtype="float32",
        shape=f"M={m},N={n},K={k}",
        baseline="naive GEMM",
        optimization=f"blocked GEMM (bm={bm},bn={bn},bk={bk})",
    )
    print("Benchmark record skeleton:")
    for key in ["scenario", "operator", "platform", "target", "dtype", "shape", "baseline", "optimization"]:
        print(f"  {key}: {record[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 6 blocked GEMM demo")
    parser.add_argument("--m", type=int, default=64, help="GEMM M dimension")
    parser.add_argument("--n", type=int, default=64, help="GEMM N dimension")
    parser.add_argument("--k", type=int, default=64, help="GEMM K dimension")
    parser.add_argument("--bm", type=int, default=32, help="Block size for M")
    parser.add_argument("--bn", type=int, default=32, help="Block size for N")
    parser.add_argument("--bk", type=int, default=16, help="Block size for K")
    parser.add_argument("--runs", type=int, default=1, help="Benchmark runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.runs)
