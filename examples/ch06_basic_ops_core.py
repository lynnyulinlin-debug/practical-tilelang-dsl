"""Shared helpers for chapter 6 basic-ops demos."""

from __future__ import annotations

import time

import numpy as np


def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def gemm_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("shape mismatch")
    c = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc += float(a[i, kk]) * float(b[kk, j])
            c[i, j] = acc
    return c


def gemm_blocked(
    a: np.ndarray,
    b: np.ndarray,
    bm: int = 32,
    bn: int = 32,
    bk: int = 16,
) -> np.ndarray:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("shape mismatch")
    c = np.zeros((m, n), dtype=np.float32)
    for ii in range(0, m, bm):
        i_end = min(ii + bm, m)
        for jj in range(0, n, bn):
            j_end = min(jj + bn, n)
            c_block = np.zeros((i_end - ii, j_end - jj), dtype=np.float32)
            for kk in range(0, k, bk):
                k_end = min(kk + bk, k)
                a_block = a[ii:i_end, kk:k_end]
                b_block = b[kk:k_end, jj:j_end]
                c_block += a_block @ b_block
            c[ii:i_end, jj:j_end] = c_block
    return c


def benchmark(fn, *args, runs: int = 10) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = None
    for _ in range(runs):
        out = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000 / runs
    return out, elapsed_ms


def make_vector_add_inputs(vec_n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.random(vec_n).astype(np.float32)
    b = rng.random(vec_n).astype(np.float32)
    return a, b


def make_gemm_inputs(m: int, n: int, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.random((m, k)).astype(np.float32)
    b = rng.random((k, n)).astype(np.float32)
    return a, b

