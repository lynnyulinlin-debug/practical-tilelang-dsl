"""Chapter 5 demo: schedule optimization and roofline analysis.

This script is intentionally lightweight and CPU-only. It illustrates how
to estimate arithmetic intensity, classify bottlenecks, and decide which
optimization should come next.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_record_template import set_benchmark_record


@dataclass(frozen=True)
class KernelProfile:
    name: str
    flops: float
    bytes_accessed: float

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes_accessed


def classify(profile: KernelProfile, peak_compute_tflops: float, peak_memory_gbs: float) -> str:
    """Return a simple bottleneck classification."""

    roofline_tflops = profile.arithmetic_intensity * peak_memory_gbs / 1e3
    if roofline_tflops >= peak_compute_tflops:
        return "compute-bound"
    return "memory-bound"


def next_step(profile: KernelProfile, peak_compute_tflops: float, peak_memory_gbs: float) -> str:
    bottleneck = classify(profile, peak_compute_tflops, peak_memory_gbs)
    if bottleneck == "memory-bound":
        return "Focus on data reuse, layout, and shared memory."
    return "Focus on compute efficiency, pipeline depth, and tensor core usage."


def demo(peak_compute_tflops: float = 312.0, peak_memory_gbs: float = 2000.0) -> None:

    profiles = [
        KernelProfile("Vector Add", flops=2 * 1024, bytes_accessed=3 * 1024 * 4),
        KernelProfile("Naive GEMM", flops=2 * 1024**3, bytes_accessed=(3 * 1024**2) * 4),
        KernelProfile("Shared GEMM", flops=2 * 1024**3, bytes_accessed=512 * 1024 * 4),
        KernelProfile("Attention baseline", flops=2 * 1 * 32 * 32 * 64, bytes_accessed=3 * 1 * 32 * 64 * 4),
    ]

    print("Chapter 5: Schedule Optimization Demo")
    print("=" * 72)
    print(f"Peak compute: {peak_compute_tflops:.1f} TFLOPS")
    print(f"Peak memory:  {peak_memory_gbs:.1f} GB/s")
    print()

    for profile in profiles:
        bottleneck = classify(profile, peak_compute_tflops, peak_memory_gbs)
        print(f"{profile.name}")
        print(f"  FLOPs: {profile.flops:.3e}")
        print(f"  Bytes: {profile.bytes_accessed:.3e}")
        print(f"  AI:    {profile.arithmetic_intensity:.2f} FLOP/Byte")
        print(f"  Class: {bottleneck}")
        print(f"  Next:  {next_step(profile, peak_compute_tflops, peak_memory_gbs)}")
        print()

    record = set_benchmark_record(
        scenario="chapter 5 schedule optimization analysis",
        operator="analysis",
        platform="NVIDIA",
        target="cuda",
        dtype="float32",
        shape="analysis only",
        baseline="roofline estimate",
        optimization="shared memory / pipeline / layout",
    )
    print("Benchmark record skeleton:")
    for key, value in record.items():
        if key in {"scenario", "operator", "platform", "target", "dtype", "shape", "baseline", "optimization"}:
            print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Chapter 5 schedule optimization demo")
    parser.add_argument("--peak-compute", type=float, default=312.0, help="Peak compute throughput in TFLOPS")
    parser.add_argument("--peak-memory", type=float, default=2000.0, help="Peak memory bandwidth in GB/s")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(args.peak_compute, args.peak_memory)
