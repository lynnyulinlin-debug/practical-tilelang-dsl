"""Chapter 3 focused demo: environment, first kernel, and verification path."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.ch06_basic_ops_core import benchmark, make_vector_add_inputs, vector_add


def tvm_status() -> str:
    try:
        tvm = importlib.import_module("tvm")
        cuda_ok = getattr(tvm, "cuda", lambda: None)()
        cuda_exist = bool(getattr(cuda_ok, "exist", False))
        return f"TVM {getattr(tvm, '__version__', 'unknown')} (cuda={cuda_exist})"
    except Exception as exc:  # pragma: no cover - demo output only
        return f"TVM unavailable: {exc.__class__.__name__}"


def main(vec_n: int = 1024, runs: int = 1000) -> None:
    a, b = make_vector_add_inputs(vec_n)
    out, elapsed_ms = benchmark(vector_add, a, b, runs=runs)

    print("Chapter 3: Quick start")
    print("=" * 72)
    print(f"Environment: {tvm_status()}")
    print(f"Vector Add: {elapsed_ms:.4f} ms per run")
    print(f"Max error: {abs(out - (a + b)).max():.2e}")
    print()
    print("Interpretation:")
    print("- This script is the smallest runnable loop for environment -> kernel -> verify.")
    print("- Correctness checks should happen before performance tuning.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 3 quick start demo")
    parser.add_argument("--vec-n", type=int, default=1024, help="Vector length")
    parser.add_argument("--runs", type=int, default=1000, help="Benchmark runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.vec_n, args.runs)
