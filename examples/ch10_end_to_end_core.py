"""Shared helpers for chapter 10 end-to-end migration demos."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_record_template import set_benchmark_record


@dataclass(frozen=True)
class ProjectConfig:
    batch: int = 1
    seq_len: int = 32
    dim: int = 64
    block_size: int = 8
    runs: int = 20

    @property
    def qkv_dim(self) -> int:
        return self.dim * 3


TARGET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cpu": {
        "tile": "fallback",
        "layout": "numpy-reference",
        "note": "CPU fallback path for documentation and verification",
    },
    "cuda": {
        "tile": "64x64",
        "layout": "coalesced",
        "note": "favor shared-memory reuse and Tensor Core-friendly shapes",
    },
    "rocm": {
        "tile": "64x64",
        "layout": "wavefront-friendly",
        "note": "favor local memory reuse and vectorization-aware layouts",
    },
    "ascend": {
        "tile": "32x64",
        "layout": "buffer-friendly",
        "note": "favor on-chip buffer locality and matrix-path alignment",
    },
}

CASE_STYLES: Dict[str, Dict[str, str]] = {
    "generic": {
        "label": "某大模型相关子图",
        "scope": "chapter-level overview and migration workflow",
        "note": "default neutral phrasing for the mainline project",
    },
    "operator": {
        "label": "某推理场景中的部分算子",
        "scope": "operator-level migration and benchmark reporting",
        "note": "use when the write-up focuses on local operators and subgraph slices",
    },
    "public": {
        "label": "DeepSeek 相关子图（在可公开范围内）",
        "scope": "public-verifiable case layer only",
        "note": "use only when public evidence is sufficient and the wording stays process-focused",
    },
}

PROJECT_LAYERS: Dict[str, Dict[str, str]] = {
    "generic": {
        "name": "总述层项目",
        "focus": "chapter-level migration workflow",
    },
    "operator": {
        "name": "算子层项目",
        "focus": "subgraph/operator migration and reporting",
    },
    "public": {
        "name": "公开案例层项目",
        "focus": "public-verifiable case packaging",
    },
}


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    x_norm = x / np.sqrt(variance + eps)
    return x_norm * weight


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    y = x @ weight
    if bias is not None:
        y = y + bias
    return y


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    return probs / probs.sum(axis=-1, keepdims=True)


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(0, 2, 1) * scale
    probs = softmax(scores)
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


def decoder_baseline(x: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    x_norm = rmsnorm(x, params["rms_weight"])
    qkv = linear(x_norm, params["w_qkv"], params["b_qkv"])
    q, k, v = np.split(qkv, 3, axis=-1)
    attn = attention_reference(q, k, v)
    out = linear(attn, params["w_out"], params["b_out"])
    return x + out


def decoder_migrated(x: np.ndarray, params: Dict[str, np.ndarray], block_size: int) -> np.ndarray:
    x_norm = rmsnorm(x, params["rms_weight"])
    qkv = linear(x_norm, params["w_qkv"], params["b_qkv"])
    q, k, v = np.split(qkv, 3, axis=-1)
    attn = attention_blockwise(q, k, v, block_size=block_size)
    out = linear(attn, params["w_out"], params["b_out"])
    return x + out


def benchmark(fn, *args, runs: int = 10) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = None
    for _ in range(runs):
        out = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000 / runs
    return out, elapsed_ms


def tensor_bytes(shape: Iterable[int], dtype=np.float32) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * np.dtype(dtype).itemsize


def estimate_peak_bytes(batch: int, seq_len: int, dim: int, kind: str, block_size: int) -> int:
    """Estimate peak temporary memory for the two stages."""

    x = tensor_bytes((batch, seq_len, dim))
    qkv = tensor_bytes((batch, seq_len, dim * 3))
    qkv_split = tensor_bytes((batch, seq_len, dim))
    output = tensor_bytes((batch, seq_len, dim))
    if kind == "baseline":
        scores = tensor_bytes((batch, seq_len, seq_len))
        probs = tensor_bytes((batch, seq_len, seq_len))
        return x + qkv + scores + probs + qkv_split + output
    q_block = tensor_bytes((batch, min(seq_len, block_size), dim))
    scores_block = tensor_bytes((batch, min(seq_len, block_size), min(seq_len, block_size)))
    probs_block = scores_block
    return x + qkv + q_block + scores_block + probs_block + qkv_split + output


def make_params(dim: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "rms_weight": rng.normal(size=(dim,)).astype(np.float32),
        "w_qkv": rng.normal(size=(dim, dim * 3)).astype(np.float32) / math.sqrt(dim),
        "b_qkv": rng.normal(size=(dim * 3,)).astype(np.float32) * 0.01,
        "w_out": rng.normal(size=(dim, dim)).astype(np.float32) / math.sqrt(dim),
        "b_out": rng.normal(size=(dim,)).astype(np.float32) * 0.01,
    }


def make_inputs(config: ProjectConfig, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(config.batch, config.seq_len, config.dim)).astype(np.float32)


def compute_metrics(
    baseline_out: np.ndarray,
    optimized_out: np.ndarray,
    baseline_ms: float,
    optimized_ms: float,
    batch: int,
    seq_len: int,
    dim: int,
) -> Dict[str, float]:
    max_err = float(np.max(np.abs(baseline_out - optimized_out)))
    flops = float(2 * batch * seq_len * dim * dim)
    baseline_tflops = flops / (baseline_ms * 1e-3) / 1e12 if baseline_ms > 0 else 0.0
    optimized_tflops = flops / (optimized_ms * 1e-3) / 1e12 if optimized_ms > 0 else 0.0
    return {
        "max_error": max_err,
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup": baseline_ms / optimized_ms if optimized_ms > 0 else float("inf"),
        "baseline_tflops": baseline_tflops,
        "optimized_tflops": optimized_tflops,
    }


def run_project(
    config: ProjectConfig,
    targets: list[str],
    case_style: str = "generic",
    subgraph_label: str = "toy decoder subgraph",
    sweep: bool = False,
    write_report: bool = False,
    report_path: str | None = None,
) -> None:
    params = make_params(config.dim)
    case_profile = CASE_STYLES.get(case_style, CASE_STYLES["generic"])
    layer_profile = PROJECT_LAYERS.get(case_style, PROJECT_LAYERS["generic"])

    print("Chapter 10: End-to-End Migration Project")
    print("=" * 80)
    print(f"Project layer: {layer_profile['name']}")
    print(f"Layer focus: {layer_profile['focus']}")
    print(f"Case style: {case_style}")
    print(f"Case label: {case_profile['label']}")
    print(f"Case scope: {case_profile['scope']}")
    print(f"Subgraph label: {subgraph_label}")
    print(f"Scenario: {subgraph_label} migration")
    print(f"Shape: B={config.batch}, S={config.seq_len}, D={config.dim}")
    print(f"Block size: {config.block_size}")
    print()

    print("Operator priority")
    for idx, name in enumerate(["RMSNorm", "QKV projection", "Attention", "Output projection", "Residual"], start=1):
        print(f"  {idx}. {name}")
    print()

    print("Migration plan")
    print("  1. baseline: separate operators + full attention")
    print("  2. migrated: blockwise attention")
    print("  3. multi-backend: record target-specific configs")
    print("  4. benchmark: operator / subgraph / end-to-end")
    print()

    def run_once(local_config: ProjectConfig) -> Dict[str, Any]:
        local_x = make_inputs(local_config)
        baseline_out, baseline_ms = benchmark(decoder_baseline, local_x, params, runs=local_config.runs)
        migrated_out, migrated_ms = benchmark(
            decoder_migrated,
            local_x,
            params,
            local_config.block_size,
            runs=local_config.runs,
        )
        local_metrics = compute_metrics(
            baseline_out,
            migrated_out,
            baseline_ms,
            migrated_ms,
            local_config.batch,
            local_config.seq_len,
            local_config.dim,
        )
        baseline_peak = estimate_peak_bytes(
            local_config.batch,
            local_config.seq_len,
            local_config.dim,
            "baseline",
            local_config.block_size,
        )
        migrated_peak = estimate_peak_bytes(
            local_config.batch,
            local_config.seq_len,
            local_config.dim,
            "migrated",
            local_config.block_size,
        )
        return {
            "config": local_config.__dict__,
            "metrics": local_metrics,
            "baseline_peak_mb": baseline_peak / (1024**2),
            "migrated_peak_mb": migrated_peak / (1024**2),
        }

    sweep_seq_lens = [config.seq_len]
    if sweep:
        if config.seq_len >= 16:
            half = max(8, config.seq_len // 2)
            sweep_seq_lens = sorted({half, config.seq_len})
        elif config.seq_len > 8:
            sweep_seq_lens = sorted({8, config.seq_len})

    sweep_results: list[Dict[str, Any]] = []
    for seq_len in sweep_seq_lens:
        local_config = ProjectConfig(
            batch=config.batch,
            seq_len=seq_len,
            dim=config.dim,
            block_size=min(config.block_size, seq_len),
            runs=config.runs,
        )
        sweep_results.append(run_once(local_config))

    primary_result = sweep_results[-1]
    metrics = primary_result["metrics"]
    print("Stage comparison")
    print(
        f"  baseline: {metrics['baseline_ms']:.4f} ms/run, "
        f"est_peak={primary_result['baseline_peak_mb']:.2f} MiB"
    )
    print(
        f"  migrated: {metrics['optimized_ms']:.4f} ms/run, "
        f"est_peak={primary_result['migrated_peak_mb']:.2f} MiB"
    )
    print(f"  max error: {metrics['max_error']:.2e}")
    print(f"  speedup: {metrics['speedup']:.2f}x")
    print()

    if len(sweep_results) > 1:
        print("Sequence-length sweep")
        for item in sweep_results:
            item_cfg = item["config"]
            item_metrics = item["metrics"]
            print(
                f"  S={item_cfg['seq_len']}: "
                f"baseline={item_metrics['baseline_ms']:.4f} ms/run, "
                f"migrated={item_metrics['optimized_ms']:.4f} ms/run, "
                f"speedup={item_metrics['speedup']:.2f}x"
            )
        print()

    print("Target migration notes")
    for target in targets:
        cfg = TARGET_CONFIGS.get(target, {"tile": "unknown", "layout": "unknown", "note": "no plan"})
        print(f"  [{target}] tile={cfg['tile']} layout={cfg['layout']} | {cfg['note']}")
    print()

    record = set_benchmark_record(
        scenario=f"chapter 10 end-to-end migration ({case_profile['label']})",
        operator=subgraph_label,
        platform="multi-backend",
        target="multi",
        dtype="float32",
        shape=f"B={config.batch},S={config.seq_len},D={config.dim}",
        baseline="baseline decoder subgraph",
        optimization="blockwise attention + iterative migration",
        notes="CPU/Numpy reference project for chapter 10",
    )
    record.update(
        {
            "latency_ms": metrics["optimized_ms"],
            "tflops": metrics["optimized_tflops"],
            "peak_mem_mb": primary_result["migrated_peak_mb"],
            "avg_mem_mb": primary_result["migrated_peak_mb"] * 0.75,
        }
    )

    print("Benchmark record")
    for key in ["scenario", "operator", "platform", "target", "dtype", "shape", "baseline", "optimization", "latency_ms", "tflops", "peak_mem_mb"]:
        print(f"  {key}: {record[key]}")
    print()

    summary = {
        "case_style": case_style,
        "case_profile": case_profile,
        "project_layer": layer_profile,
        "subgraph_label": subgraph_label,
        "sweep_enabled": sweep,
        "config": config.__dict__,
        "targets": targets,
        "metrics": metrics,
        "baseline_peak_mb": primary_result["baseline_peak_mb"],
        "migrated_peak_mb": primary_result["migrated_peak_mb"],
        "sweep_results": sweep_results,
        "benchmark_record": record,
    }

    if write_report:
        out_path = Path(report_path or "artifacts/ch10_end_to_end_migration_report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved report to {out_path}")

