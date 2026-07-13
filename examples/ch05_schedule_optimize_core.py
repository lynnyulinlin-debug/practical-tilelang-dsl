"""Shared helpers for chapter 5 schedule-optimization demos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelProfile:
    name: str
    flops: float
    bytes_accessed: float

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes_accessed


def classify(profile: KernelProfile, peak_compute_tflops: float, peak_memory_gbs: float) -> str:
    roofline_tflops = profile.arithmetic_intensity * peak_memory_gbs / 1e3
    if roofline_tflops >= peak_compute_tflops:
        return "compute-bound"
    return "memory-bound"


def next_step(profile: KernelProfile, peak_compute_tflops: float, peak_memory_gbs: float) -> str:
    bottleneck = classify(profile, peak_compute_tflops, peak_memory_gbs)
    if bottleneck == "memory-bound":
        return "Focus on data reuse, layout, and shared memory."
    return "Focus on compute efficiency, pipeline depth, and tensor core usage."


def get_profiles() -> list[KernelProfile]:
    return [
        KernelProfile("Vector Add", flops=2 * 1024, bytes_accessed=3 * 1024 * 4),
        KernelProfile("Naive GEMM", flops=2 * 1024**3, bytes_accessed=(3 * 1024**2) * 4),
        KernelProfile("Shared GEMM", flops=2 * 1024**3, bytes_accessed=512 * 1024 * 4),
        KernelProfile("Attention baseline", flops=2 * 1 * 32 * 32 * 64, bytes_accessed=3 * 1 * 32 * 64 * 4),
    ]

