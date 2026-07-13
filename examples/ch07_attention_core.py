"""Shared helpers for chapter 7 attention demos."""

from __future__ import annotations

import math
import time

import numpy as np


def make_attention_inputs(batch: int, seq_len: int, dim: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((batch, seq_len, dim)).astype(np.float32)
    k = rng.standard_normal((batch, seq_len, dim)).astype(np.float32)
    v = rng.standard_normal((batch, seq_len, dim)).astype(np.float32)
    return q, k, v


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(0, 2, 1) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs @ v


def attention_blockwise(q: np.ndarray, k: np.ndarray, v: np.ndarray, block_size: int = 8) -> np.ndarray:
    batch, seq_len, dim = q.shape
    out = np.zeros((batch, seq_len, dim), dtype=np.float32)
    scale = 1.0 / math.sqrt(dim)

    for b in range(batch):
        for q_start in range(0, seq_len, block_size):
            q_end = min(q_start + block_size, seq_len)
            q_block = q[b, q_start:q_end]

            running_max = np.full((q_end - q_start, 1), -np.inf, dtype=np.float32)
            running_sum = np.zeros((q_end - q_start, 1), dtype=np.float32)
            running_out = np.zeros((q_end - q_start, dim), dtype=np.float32)

            for kv_start in range(0, seq_len, block_size):
                kv_end = min(kv_start + block_size, seq_len)
                k_block = k[b, kv_start:kv_end]
                v_block = v[b, kv_start:kv_end]

                scores = (q_block @ k_block.T) * scale
                block_max = scores.max(axis=-1, keepdims=True)
                new_max = np.maximum(running_max, block_max)

                exp_old = np.exp(running_max - new_max) * running_sum
                exp_new = np.exp(scores - new_max)
                running_sum = exp_old + exp_new.sum(axis=-1, keepdims=True)
                running_out = running_out * np.exp(running_max - new_max) + exp_new @ v_block
                running_max = new_max

            out[b, q_start:q_end] = running_out / running_sum

    return out


def benchmark(fn, *args, runs: int = 10, **kwargs) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = None
    for _ in range(runs):
        out = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000 / runs
    return out, elapsed_ms
