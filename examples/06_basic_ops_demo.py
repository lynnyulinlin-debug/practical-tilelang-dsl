"""Chapter 6 demo: vector add and GEMM evolution.

This script is CPU-only and focuses on the educational progression:
vector add -> naive GEMM -> blocked GEMM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_record_template import set_benchmark_record
from examples.ch06_basic_ops_core import (
    benchmark,
    gemm_blocked,
    gemm_naive,
    make_gemm_inputs,
    make_vector_add_inputs,
    vector_add,
)


def demo(
    mode: str = "all",
    vec_n: int = 1024,
    m: int = 64,
    n: int = 64,
    k: int = 64,
    runs: int = 1000,
) -> None:
    np.random.seed(0)

    print("Chapter 6: Basic Ops Demo")
    print("=" * 72)
    ran_any = False
    if mode in {"vector-add", "all"}:
        a, b = make_vector_add_inputs(vec_n)
        add_out, add_ms = benchmark(vector_add, a, b, runs=runs)
        print(f"Vector Add: {add_ms:.4f} ms per run, max error {np.max(np.abs(add_out - (a + b))):.2e}")
        ran_any = True

    if mode in {"naive", "all"}:
        a2, b2 = make_gemm_inputs(m, n, k)
        ref = a2 @ b2
        naive_out, naive_ms = benchmark(gemm_naive, a2, b2, runs=1)
        print(f"Naive GEMM: {naive_ms:.4f} ms per run, max error {np.max(np.abs(naive_out - ref)):.2e}")
        ran_any = True

    if mode in {"blocked", "all"}:
        a2, b2 = make_gemm_inputs(m, n, k)
        ref = a2 @ b2
        blocked_out, blocked_ms = benchmark(gemm_blocked, a2, b2, runs=max(1, runs // 100))
        print(f"Blocked GEMM: {blocked_ms:.4f} ms per run, max error {np.max(np.abs(blocked_out - ref)):.2e}")
        ran_any = True

    if ran_any:
        print()
        print("Optimization interpretation:")
        if mode in {"vector-add", "all"}:
            print("- Vector add establishes the correctness baseline.")
        if mode in {"naive", "all"}:
            print("- Naive GEMM exposes repeated global-memory traffic.")
        if mode in {"blocked", "all"}:
            print("- Blocked GEMM shows how tile reuse changes the cost structure.")

        record = set_benchmark_record(
            scenario=f"chapter 6 {mode} demo",
            operator="GEMM" if mode != "vector-add" else "Vector Add",
            platform="CPU",
            target="numpy",
            dtype="float32",
            shape="M=64,N=64,K=64" if mode != "vector-add" else f"N={vec_n}",
            baseline="naive GEMM" if mode != "vector-add" else "vector add baseline",
            optimization="blocked GEMM" if mode != "vector-add" else "vector add correctness",
        )
        print()
        print("Benchmark record skeleton:")
        for key in ["scenario", "operator", "platform", "target", "dtype", "shape", "baseline", "optimization"]:
            print(f"  {key}: {record[key]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Chapter 6 basic ops demo")
    parser.add_argument(
        "--mode",
        choices=["vector-add", "naive", "blocked", "all"],
        default="all",
        help="Which part of the demo to run",
    )
    parser.add_argument("--vec-n", type=int, default=1024, help="Vector length")
    parser.add_argument("--m", type=int, default=64, help="GEMM M dimension")
    parser.add_argument("--n", type=int, default=64, help="GEMM N dimension")
    parser.add_argument("--k", type=int, default=64, help="GEMM K dimension")
    parser.add_argument("--runs", type=int, default=1000, help="Benchmark runs for vector add")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(args.mode, args.vec_n, args.m, args.n, args.k, args.runs)
