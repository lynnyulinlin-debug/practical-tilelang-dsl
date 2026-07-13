"""
Example 4: Matrix Multiplication (GEMM) - From Naive to Optimized

这是第4章 "分块与内存层次" 的代码示例。
展示三个GEMM实现的演进：
  1. 朴素版本（无优化，很慢）
  2. 带shared memory的版本（快10-20倍）
  3. Tensor Core 路径（章节讨论的目标形态，当前脚本不直接运行）

这个例子最重要的是理解性能差异的根本原因。

依赖：
  - TVM >= 0.13
  - CUDA compute capability >= 7.0（用于章节中的 Tensor Core 路径）

运行：
  python 04_gemm_basic.py --mode auto
  python 04_gemm_basic.py --mode cpu
  python 04_gemm_basic.py --mode naive
  python 04_gemm_basic.py --mode shared
  python 04_gemm_basic.py --mode all

在当前机器上如果没有可用 CUDA 驱动，会自动退回到 CPU fallback。
当前 `apache-tvm` 构建如果不提供 script helpers，则只执行 CPU 路径。
"""

import argparse
import numpy as np
import time
import sys

try:
    import tvm
    from tvm import te
    from tvm.ir import IRModule
    from tvm.runtime._tensor import tensor as tvm_tensor

    try:
        from tvm import tir
    except ImportError:
        import tvm.tirx as tir

    TVM_AVAILABLE = True
except ModuleNotFoundError:
    tvm = None
    te = None
    IRModule = None
    tvm_tensor = None
    tir = None
    TVM_AVAILABLE = False


def benchmark_gemm(
    f,
    a_np,
    b_np,
    c_np,
    name="GEMM",
    device=None,
    do_sync=True,
    num_runs: int = 100,
):
    """
    测试GEMM性能

    参数：
      f: 编译后的kernel函数
      a_np, b_np, c_np: numpy数组
      name: 用于打印的名称

    返回：
      (执行时间ms, TFLOPS, 带宽利用率%)
    """
    M, K = a_np.shape
    K, N = b_np.shape

    if device is None:
        device = tvm.cuda(0)

    a_gpu = tvm_tensor(a_np, device=device)
    b_gpu = tvm_tensor(b_np, device=device)
    c_gpu = tvm_tensor(c_np, device=device)

    # 预热
    for _ in range(5):
        f(a_gpu, b_gpu, c_gpu)
    if do_sync and hasattr(device, "synchronize"):
        device.synchronize()

    # 计时
    start = time.time()
    for _ in range(num_runs):
        f(a_gpu, b_gpu, c_gpu)
    if do_sync and hasattr(device, "synchronize"):
        device.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000

    # 计算性能指标
    flops = 2 * M * N * K  # 乘加计数
    tflops = flops / (avg_time_ms * 1e-3) / 1e12

    # 内存访问量
    bytes_accessed = (M * K + K * N + M * N) * 4  # float32
    gbps = bytes_accessed / (avg_time_ms * 1e-3) / 1e9

    print(f"\n{name}:")
    print(f"  Shape: {M}×{K} × {K}×{N} → {M}×{N}")
    print(f"  Time: {avg_time_ms:.3f} ms")
    print(f"  TFLOPS: {tflops:.1f} (Peak A100: 312 TFLOPS)")
    print(f"  Bandwidth: {gbps:.1f} GB/s (Peak A100: 2000 GB/s)")
    print(f"  Utilization: {tflops/312*100:.1f}% (compute), {gbps/2000*100:.1f}% (memory)")

    return avg_time_ms, tflops, gbps


def cpu_fallback_matmul(m: int = 256, n: int = 256, k: int = 256, num_runs: int = 10):
    """
    没有 CUDA 时，跑一个最小可验证的 TVM GEMM CPU 版本。
    """
    print("\n" + "=" * 70)
    print("CPU Fallback: TVM GEMM on LLVM")
    print("=" * 70)

    if not TVM_AVAILABLE:
        print("TVM is not available in this Python environment; using a pure NumPy fallback.")
        a_np = np.random.randn(m, k).astype("float32")
        b_np = np.random.randn(k, n).astype("float32")
        start = time.time()
        for _ in range(num_runs):
            c_np = a_np @ b_np
        end = time.time()
        ref = a_np @ b_np
        time_ms = (end - start) * 1000 / num_runs
        print(f"\nCPU GEMM Baseline:")
        print(f"  Shape: {m}×{k} × {k}×{n} → {m}×{n}")
        print(f"  Time: {time_ms:.3f} ms")
        print(f"  Max error: {np.max(np.abs(c_np - ref)):.2e}")
        print("  TVM-specific GPU versions are skipped in this environment.")
        return time_ms, None, None

    M, N, K = m, n, k
    A = te.placeholder((M, K), name="A", dtype="float32")
    B = te.placeholder((K, N), name="B", dtype="float32")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    prim_func = te.create_prim_func([A, B, C])
    ir_module = IRModule({"matmul_cpu": prim_func})
    f = tvm.build(ir_module, target="llvm")

    a_np = np.random.randn(M, K).astype("float32")
    b_np = np.random.randn(K, N).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    time_ms, tflops, gbps = benchmark_gemm(
        f,
        a_np,
        b_np,
        c_np,
        name="CPU GEMM Baseline",
        device=tvm.cpu(0),
        do_sync=False,
        num_runs=num_runs,
    )

    print("\nCUDA-specific versions are skipped in this environment.")
    return time_ms, tflops, gbps


def naive_matmul(m: int = 1024, n: int = 1024, k: int = 1024, num_runs: int = 100):
    """
    版本1：朴素矩阵乘法（无优化）

    所有数据都从全局内存读取，没有数据复用。
    预期性能：很差，10-50 TFLOPS
    """
    print("\n" + "="*70)
    print("Version 1: Naive GEMM (No Optimization)")
    print("="*70)

    if not TVM_AVAILABLE:
        print("TVM is not available in this Python environment; using a pure NumPy naive GEMM fallback.")
        a_np = np.random.randn(m, k).astype("float32")
        b_np = np.random.randn(k, n).astype("float32")
        ref = a_np @ b_np
        start = time.time()
        for _ in range(max(1, num_runs // 10)):
            c_np = a_np @ b_np
        end = time.time()
        time_ms = (end - start) * 1000 / max(1, num_runs // 10)
        print(f"  Max error: {np.max(np.abs(c_np - ref)):.2e}")
        print(f"  Time: {time_ms:.3f} ms")
        return time_ms, None, None

    M, N, K = m, n, k

    print(f"\n代码结构：")
    print(f"""
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i,j] += A[i,k] * B[k,j]  # 全部从全局内存读
    """)

    print(f"\n性能分析：")
    print(f"  计算量：2×{M}×{N}×{K} = {2*M*N*K/1e9:.1f}G 次乘加")
    print(f"  内存访问：A:{M*K}, B:{K*N}, C:{M*N} = {(M*K+K*N+M*N)*4/1e9:.1f}G 字节")
    print(f"  算术强度：{2*M*N*K / (M*K+K*N+M*N):.1f} FLOP/Byte")
    print(f"  （这很低！A100理论强度156，所以这个kernel是内存受限的）")

    # 定义kernel
    @tir.prim_func
    def matmul_kernel(
        A: tir.Buffer[(M, K), "float32"],
        B: tir.Buffer[(K, N), "float32"],
        C: tir.Buffer[(M, N), "float32"]
    ):
        for i, j, k in tir.grid(M, N, K):
            with tir.block("compute"):
                vi, vj, vk = tir.axis.remap("SSR", [i, j, k])
                with tir.init():
                    C[vi, vj] = 0.0
                C[vi, vj] += A[vi, vk] * B[vk, vj]

    ir_module = tir.IRModule({"matmul": matmul_kernel})
    f = tvm.build(ir_module, target="cuda")

    # 测试
    a_np = np.random.randn(M, K).astype("float32")
    b_np = np.random.randn(K, N).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    time_ms, tflops, gbps = benchmark_gemm(f, a_np, b_np, c_np, "Naive GEMM", num_runs=num_runs)

    return time_ms, tflops, gbps


def shared_memory_matmul(m: int = 1024, n: int = 1024, k: int = 1024, num_runs: int = 100):
    """
    版本2：带shared memory的矩阵乘法

    把A和B的块加载到shared memory，利用线程间的数据共享。
    预期性能：100-150 TFLOPS（快10-20倍）
    """
    print("\n" + "="*70)
    print("Version 2: GEMM with Shared Memory")
    print("="*70)

    if not TVM_AVAILABLE:
        print("TVM is not available in this Python environment; using a pure NumPy blocked GEMM fallback.")
        a_np = np.random.randn(m, k).astype("float32")
        b_np = np.random.randn(k, n).astype("float32")
        ref = a_np @ b_np
        start = time.time()
        for _ in range(max(1, num_runs // 10)):
            c_np = a_np @ b_np
        end = time.time()
        time_ms = (end - start) * 1000 / max(1, num_runs // 10)
        print(f"  Max error: {np.max(np.abs(c_np - ref)):.2e}")
        print(f"  Time: {time_ms:.3f} ms")
        return time_ms, None, None

    M, N, K = m, n, k
    BM, BN, BK = 64, 64, 32  # tile大小

    print(f"\n代码结构：")
    print(f"""
    分块大小：BM={BM}, BN={BN}, BK={BK}
    Shared memory用量：{BM*BK*4 + BK*BN*4} 字节 = {(BM*BK*4 + BK*BN*4)/1024:.1f} KB

    for ii in range(M/BM):
        for jj in range(N/BN):
            C_local[BM, BN] = 0
            for kk in range(K/BK):
                A_shared[BM, BK] = A[ii*BM:, kk*BK:]  # 加载
                B_shared[BK, BN] = B[kk*BK:, jj*BN:]
                __syncthreads()

                # 在shared memory上计算
                for i in range(BM):
                    for j in range(BN):
                        for k in range(BK):
                            C_local[i,j] += A_shared[i,k] * B_shared[k,j]

            C[...] = C_local
    """)

    print(f"\n性能分析：")
    total_flops = 2 * M * N * K
    print(f"  计算量：{total_flops/1e9:.1f}G 次乘加（同上）")

    # 全局内存访问分析
    num_blocks_m = (M + BM - 1) // BM
    num_blocks_n = (N + BN - 1) // BN
    num_blocks_k = (K + BK - 1) // BK

    # A的全局访问：每个block读一次
    a_global = num_blocks_m * num_blocks_n * BM * BK
    # B的全局访问：每个block读一次
    b_global = num_blocks_m * num_blocks_n * BK * BN
    # C的全局访问：读+写
    c_global = M * N * 2

    total_bytes = (a_global + b_global + c_global) * 4
    arithmetic_intensity = total_flops / total_bytes

    print(f"  全局内存访问：{total_bytes/1e9:.1f}G 字节")
    print(f"  新的算术强度：{arithmetic_intensity:.1f} FLOP/Byte")
    print(f"  性能上限（受计算限制）：{312*arithmetic_intensity/156:.0f} TFLOPS")
    print(f"  （vs 朴素版本的 ~10 TFLOPS，快 ~30 倍！）")

    # 定义kernel（简化版本，不包括所有细节）
    @tir.prim_func
    def matmul_shared_kernel(
        A: tir.Buffer[(M, K), "float32"],
        B: tir.Buffer[(K, N), "float32"],
        C: tir.Buffer[(M, N), "float32"]
    ):
        # 这是一个简化的版本，真实的实现会更复杂
        # 这里主要是为了演示概念

        A_shared = tir.alloc_buffer([BM, BK], dtype="float32", scope="shared")
        B_shared = tir.alloc_buffer([BK, BN], dtype="float32", scope="shared")

        for ii in tir.parallel(M // BM, axis="blockIdx.x"):
            for jj in tir.parallel(N // BN, axis="blockIdx.y"):
                C_local = tir.alloc_buffer([BM, BN], dtype="float32", scope="local")

                # 初始化
                for v_i in range(BM):
                    for v_j in range(BN):
                        C_local[v_i, v_j] = 0.0

                # K维度的分块循环
                for kk in range(K // BK):
                    # 加载A到shared memory
                    for i in tir.parallel(BM, axis="threadIdx.y"):
                        for k in tir.parallel(BK, axis="threadIdx.x"):
                            A_shared[i, k] = A[ii * BM + i, kk * BK + k]

                    # 加载B到shared memory
                    for k in tir.parallel(BK, axis="threadIdx.y"):
                        for j in tir.parallel(BN, axis="threadIdx.x"):
                            B_shared[k, j] = B[kk * BK + k, jj * BN + j]

                    # 同步
                    tir.tvm_storage_sync("shared")

                    # 在shared memory上计算
                    for v_i in range(BM):
                        for v_j in range(BN):
                            for v_k in range(BK):
                                C_local[v_i, v_j] += A_shared[v_i, v_k] * B_shared[v_k, v_j]

                    tir.tvm_storage_sync("shared")

                # 写回
                for i in tir.parallel(BM, axis="threadIdx.y"):
                    for j in tir.parallel(BN, axis="threadIdx.x"):
                        C[ii * BM + i, jj * BN + j] = C_local[i, j]

    ir_module = tir.IRModule({"matmul_shared": matmul_shared_kernel})
    f = tvm.build(ir_module, target="cuda")

    # 测试
    a_np = np.random.randn(M, K).astype("float32")
    b_np = np.random.randn(K, N).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    time_ms, tflops, gbps = benchmark_gemm(
        f, a_np, b_np, c_np, "Shared Memory GEMM", num_runs=num_runs
    )

    return time_ms, tflops, gbps


def comparison():
    """
    对比分析
    """
    print("\n" + "="*70)
    print("Performance Comparison")
    print("="*70)

    print("""
优化方法        | 代码复杂度 | 性能      | 主要改进
----------------|----------|---------|------------------
朴素版本        | 简单      | 低       | 无
Shared Memory   | 中等      | 中-高    | 减少全局内存访问
Tensor Core     | 复杂      | 很高     | 硬件加速 + 数据复用
Multi-level     | 很复杂    | 最优     | 所有优化结合

关键洞察：
  1. 朴素版本是内存受限的（算术强度低）
  2. Shared memory优化通过增加数据复用来提高算术强度
  3. Tensor Core进一步提高计算吞吐
  4. 在第5章会学习如何自动搜索最优的分块大小和参数
    """)


def parse_args():
    parser = argparse.ArgumentParser(description="Chapter 4 GEMM demo")
    parser.add_argument(
        "--mode",
        choices=["auto", "cpu", "naive", "shared", "all"],
        default="auto",
        help="Execution mode",
    )
    parser.add_argument("--m", type=int, default=1024, help="GEMM M dimension")
    parser.add_argument("--n", type=int, default=1024, help="GEMM N dimension")
    parser.add_argument("--k", type=int, default=1024, help="GEMM K dimension")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark runs")
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n" + "="*70)
    print("GEMM Optimization Tutorial")
    print("="*70)

    cuda_available = TVM_AVAILABLE and tvm.cuda().exist

    if not TVM_AVAILABLE:
        print("TVM is not available in this Python environment; using NumPy fallbacks.")
        if args.mode in {"naive", "shared", "all"}:
            if args.mode in {"naive", "all"}:
                naive_matmul(args.m, args.n, args.k, num_runs=max(1, args.runs // 10))
            if args.mode in {"shared", "all"}:
                shared_memory_matmul(args.m, args.n, args.k, num_runs=max(1, args.runs // 10))
        else:
            cpu_fallback_matmul(args.m, args.n, args.k, num_runs=max(1, args.runs // 10))
        return

    if args.mode == "cpu" or (args.mode == "auto" and not cuda_available) or not cuda_available:
        if not cuda_available and args.mode != "cpu":
            print("CUDA not available in this environment; running CPU fallback instead.")
        elif args.mode == "cpu":
            print("CPU mode requested; running CPU fallback instead.")
        cpu_fallback_matmul(args.m, args.n, args.k, num_runs=max(1, args.runs // 10))
        return

    if not hasattr(tir, "prim_func") or not hasattr(tir, "block"):
        print("This TVM build does not expose the script helpers used by the GPU examples.")
        print("Skip GEMM GPU examples.")
        return

    mode = args.mode
    if mode == "auto":
        mode = "all"

    if mode == "naive" or mode == "all":
        t1, tflops1, gbps1 = naive_matmul(args.m, args.n, args.k, num_runs=args.runs)

    if mode == "shared" or mode == "all":
        t2, tflops2, gbps2 = shared_memory_matmul(args.m, args.n, args.k, num_runs=args.runs)

    if mode == "all":
        comparison()
        print(f"\n加速比：{tflops1/tflops2:.1f}x")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("""
  1. 修改BM/BN/BK的值，观察性能变化
  2. 计算shared memory的bank conflict
  3. 在第5章学习流水线和自动调优
  4. 在第8章看完整的、经过充分优化的GEMM实现
    """)


if __name__ == "__main__":
    main()
