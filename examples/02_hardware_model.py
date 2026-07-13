"""Chapter 2 focused demo: hardware hierarchy and tile decisions."""

from __future__ import annotations

import argparse


def roofline_classify(flops: float, bytes_accessed: float, peak_compute_tflops: float, peak_memory_gbs: float) -> str:
    ai = flops / bytes_accessed
    roofline_tflops = ai * peak_memory_gbs / 1e3
    return "compute-bound" if roofline_tflops >= peak_compute_tflops else "memory-bound"


def main() -> None:
    print("Chapter 2: Hardware hierarchy")
    print("=" * 72)
    print("Execution: Grid -> Block -> Warp -> Thread")
    print("Memory:    Register -> Shared -> L2 -> Global")
    print("Compute:   CUDA Core / Tensor Core")
    print()

    cases = [
        ("Vector Add", 2 * 1024, 3 * 1024 * 4),
        ("Naive GEMM", 2 * 1024**3, (3 * 1024**2) * 4),
        ("Shared GEMM", 2 * 1024**3, 512 * 1024 * 4),
    ]

    for name, flops, bytes_accessed in cases:
        kind = roofline_classify(flops, bytes_accessed, peak_compute_tflops=312.0, peak_memory_gbs=2000.0)
        ai = flops / bytes_accessed
        print(f"{name:<12} AI={ai:8.2f} FLOP/Byte -> {kind}")

    print()
    print("Interpretation:")
    print("- Tile is the boundary that connects data reuse, parallel granularity, and memory locality.")
    print("- Tensor Core is not always faster; layout and tile shape must match.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 2 hardware model demo")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    main()
