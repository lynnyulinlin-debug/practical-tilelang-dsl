"""Chapter 5 focused demo: optimization next-step hints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_record_template import set_benchmark_record
from examples.ch05_schedule_optimize_core import get_profiles, next_step


def main(peak_compute_tflops: float = 312.0, peak_memory_gbs: float = 2000.0) -> None:
    print("Chapter 5: Optimization Next-Step Hints")
    print("=" * 72)
    for profile in get_profiles():
        print(f"{profile.name}")
        print(f"  Next: {next_step(profile, peak_compute_tflops, peak_memory_gbs)}")
        print()

    record = set_benchmark_record(
        scenario="chapter 5 autotuning hint demo",
        operator="analysis",
        platform="NVIDIA",
        target="cuda",
        dtype="float32",
        shape="analysis only",
        baseline="roofline estimate",
        optimization="autotuning hint",
    )
    print("Benchmark record skeleton:")
    for key in ["scenario", "operator", "platform", "target", "dtype", "shape", "baseline", "optimization"]:
        print(f"  {key}: {record[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 5 autotuning hint demo")
    parser.add_argument("--peak-compute", type=float, default=312.0, help="Peak compute throughput in TFLOPS")
    parser.add_argument("--peak-memory", type=float, default=2000.0, help="Peak memory bandwidth in GB/s")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.peak_compute, args.peak_memory)
