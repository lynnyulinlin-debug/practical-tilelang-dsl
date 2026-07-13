"""Shared benchmark record template for tutorial chapters.

This file is intentionally simple: it defines a canonical record shape
that chapters can reuse when collecting or reporting benchmark data.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


BENCHMARK_RECORD_TEMPLATE: Dict[str, Any] = {
    "scenario": "",
    "operator": "",
    "platform": "",
    "target": "",
    "version": "",
    "driver_version": "",
    "runtime_version": "",
    "dtype": "",
    "shape": "",
    "batch": 1,
    "warmup": 10,
    "runs": 100,
    "latency_ms": None,
    "throughput": None,
    "tflops": None,
    "gbps": None,
    "peak_mem_mb": None,
    "avg_mem_mb": None,
    "baseline": "",
    "optimization": "",
    "notes": "",
}


def new_benchmark_record() -> Dict[str, Any]:
    """Return a fresh benchmark record with canonical keys."""

    return deepcopy(BENCHMARK_RECORD_TEMPLATE)


def set_benchmark_record(**kwargs: Any) -> Dict[str, Any]:
    """Create a benchmark record and fill selected fields.

    Unknown keys are ignored so the template stays stable across chapters.
    """

    record = new_benchmark_record()
    for key, value in kwargs.items():
        if key in record:
            record[key] = value
    return record


def benchmark_record_keys() -> tuple[str, ...]:
    """Expose the canonical key order for documentation and printing."""

    return tuple(BENCHMARK_RECORD_TEMPLATE.keys())


if __name__ == "__main__":
    record = set_benchmark_record(
        scenario="example",
        operator="GEMM",
        platform="NVIDIA",
        target="cuda",
        dtype="float16",
        shape="M=1024,N=1024,K=1024",
    )
    for key in benchmark_record_keys():
        print(f"{key}: {record[key]}")
