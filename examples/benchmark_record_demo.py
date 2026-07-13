"""Demonstrate how to fill benchmark records for different chapters."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_record_template import set_benchmark_record


def main() -> None:
    records = [
        set_benchmark_record(
            scenario="chapter 6 baseline GEMM",
            operator="GEMM",
            platform="NVIDIA",
            target="cuda",
            dtype="float16",
            shape="M=1024,N=1024,K=1024",
            baseline="naive GEMM",
            optimization="shared memory + layout",
        ),
        set_benchmark_record(
            scenario="chapter 7 attention benchmark",
            operator="Attention",
            platform="NVIDIA",
            target="cuda",
            dtype="float16",
            shape="B=1,S=4096,D=128",
            baseline="standard attention",
            optimization="FlashAttention + fusion",
        ),
        set_benchmark_record(
            scenario="chapter 10 end-to-end migration",
            operator="subgraph",
            platform="multi-backend",
            target="cuda",
            dtype="float16",
            shape="subgraph specific",
            baseline="original implementation",
            optimization="iterative migration + backend tuning",
        ),
    ]

    for idx, record in enumerate(records, 1):
        print(f"=== record {idx} ===")
        for key, value in record.items():
            print(f"{key}: {value}")
        print()


if __name__ == "__main__":
    main()
