"""
Example 3: Vector Add - First TileLang Program

这是第3章 "TileLang快速入门" 的完整代码示例。
展示从高层TE API到底层TIR API的两种写法。

依赖：
  - TVM >= 0.13
  - CUDA 环境（或用 target="llvm" 测试）

运行：
  python 03_vector_add.py --mode auto
  python 03_vector_add.py --mode cpu
  python 03_vector_add.py --mode cuda
  python 03_vector_add.py --mode all
"""

import numpy as np
import time
import argparse
import sys
from pathlib import Path

try:
    import tvm
    from tvm import te
    from tvm.ir import IRModule
    from tvm.runtime._tensor import tensor as tvm_tensor

    TVM_AVAILABLE = True
except ModuleNotFoundError:
    tvm = None
    te = None
    IRModule = None
    tvm_tensor = None
    TVM_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def example_te_api(n: int = 1024, num_runs: int = 100):
    """
    高层 TE API 方式
    优点：简洁，编译器自动处理细节
    缺点：灵活性有限，难以做深度优化
    """
    print("=" * 60)
    print("Example 1: Vector Add using high-level TE API")
    print("=" * 60)

    if not TVM_AVAILABLE:
        print("TVM is not available in this Python environment; skipping the TE/TIR demo.")
        return None

    # 问题参数
    # 第1步：定义计算（TE表达）
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.placeholder((n,), name="B", dtype="float32")
    C = te.compute((n,), lambda i: A[i] + B[i], name="C")

    # 第2步：把 TE 计算 lowered 成 PrimFunc，再交给 TVM build
    prim_func = te.create_prim_func([A, B, C])
    ir_module = IRModule({"vector_add_te": prim_func})

    # 第3步：编译到 GPU
    print(f"\nCompiling to CUDA target...")
    f = tvm.build(ir_module, target="cuda")

    # 第4步：准备数据
    a_np = np.random.uniform(0, 1, (n,)).astype("float32")
    b_np = np.random.uniform(0, 1, (n,)).astype("float32")

    # 第5步：创建GPU数组
    a_gpu = tvm_tensor(a_np, device=tvm.cuda(0))
    b_gpu = tvm_tensor(b_np, device=tvm.cuda(0))
    c_gpu = tvm_tensor(np.zeros((n,), dtype="float32"), device=tvm.cuda(0))

    # 第6步：执行
    print(f"Executing kernel...")
    f(a_gpu, b_gpu, c_gpu)

    # 第7步：验证结果
    c_np = c_gpu.numpy()
    c_expected = a_np + b_np
    error = np.max(np.abs(c_np - c_expected))

    print(f"\nResults:")
    print(f"  Max error: {error:.2e}")
    print(f"  Test: {'PASSED ✓' if error < 1e-5 else 'FAILED ✗'}")

    # 性能测试
    print(f"\nPerformance:")
    tvm.cuda().synchronize()  # 确保GPU空闲

    start = time.time()
    for _ in range(num_runs):
        f(a_gpu, b_gpu, c_gpu)
    tvm.cuda().synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000
    bytes_per_op = n * 3 * 4  # 3个float32数组
    gbps = bytes_per_op / avg_time_ms / 1e6

    print(f"  Time per run: {avg_time_ms:.3f} ms")
    print(f"  Bandwidth: {gbps:.1f} GB/s")
    print(f"  Peak (A100): 2000 GB/s, Utilization: {gbps/2000*100:.1f}%")

    return f


def example_tir_api(n: int = 1024, num_runs: int = 100):
    """
    底层 TIR API 方式（更接近 TileLang）
    优点：完全可控，可以做低层优化
    缺点：代码更复杂
    """
    print("\n" + "=" * 60)
    print("Example 2: Vector Add using low-level TIR API")
    print("=" * 60)

    if not TVM_AVAILABLE:
        print("TVM is not available in this Python environment; skipping the TE/TIR demo.")
        return None

    # 新版 TVM 中，TE 也可以直接 lowered 成 PrimFunc，作为“底层表示”示例
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.placeholder((n,), name="B", dtype="float32")
    C = te.compute((n,), lambda i: A[i] + B[i], name="C")
    prim_func = te.create_prim_func([A, B, C])
    ir_module = IRModule({"vector_add": prim_func})

    # 编译
    print(f"\nCompiling to CUDA target...")
    f = tvm.build(ir_module, target="cuda")

    # 执行（和前面一样）
    a_np = np.random.uniform(0, 1, (n,)).astype("float32")
    b_np = np.random.uniform(0, 1, (n,)).astype("float32")

    a_gpu = tvm_tensor(a_np, device=tvm.cuda(0))
    b_gpu = tvm_tensor(b_np, device=tvm.cuda(0))
    c_gpu = tvm_tensor(np.zeros((n,), dtype="float32"), device=tvm.cuda(0))

    print(f"Executing kernel...")
    f(a_gpu, b_gpu, c_gpu)

    # 验证
    c_np = c_gpu.numpy()
    c_expected = a_np + b_np
    error = np.max(np.abs(c_np - c_expected))

    print(f"\nResults:")
    print(f"  Max error: {error:.2e}")
    print(f"  Test: {'PASSED ✓' if error < 1e-5 else 'FAILED ✗'}")

    # 性能测试
    print(f"\nPerformance:")
    tvm.cuda().synchronize()

    start = time.time()
    for _ in range(num_runs):
        f(a_gpu, b_gpu, c_gpu)
    tvm.cuda().synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000
    bytes_per_op = n * 3 * 4
    gbps = bytes_per_op / avg_time_ms / 1e6

    print(f"  Time per run: {avg_time_ms:.3f} ms")
    print(f"  Bandwidth: {gbps:.1f} GB/s")
    print(f"  Peak (A100): 2000 GB/s, Utilization: {gbps/2000*100:.1f}%")

    # 查看生成的 PTX 代码
    print(f"\nGenerated PTX (first 20 lines):")
    ptx_code = f.imported_modules[0].get_source()
    lines = ptx_code.split('\n')[:20]
    for line in lines:
        print(f"  {line}")

    return f


def example_cpu_fallback(n: int = 1024, num_runs: int = 10):
    """
    如果没有 CUDA 环境，用 CPU 测试
    """
    print("\n" + "=" * 60)
    print("Example 3: Vector Add on CPU (fallback if no CUDA)")
    print("=" * 60)

    if not TVM_AVAILABLE:
        print("TVM is not available in this Python environment; using a pure NumPy fallback.")
        a_np = np.random.uniform(0, 1, (n,)).astype("float32")
        b_np = np.random.uniform(0, 1, (n,)).astype("float32")
        start = time.time()
        for _ in range(num_runs):
            c_np = a_np + b_np
        end = time.time()
        error = np.max(np.abs(c_np - (a_np + b_np)))
        avg_time_ms = (end - start) / num_runs * 1000
        print(f"  Max error: {error:.2e}")
        print(f"  Test: {'PASSED ✓' if error < 1e-5 else 'FAILED ✗'}")
        print(f"  Time per run: {avg_time_ms:.3f} ms")
        return None

    # 定义计算
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.placeholder((n,), name="B", dtype="float32")
    C = te.compute((n,), lambda i: A[i] + B[i], name="C")

    prim_func = te.create_prim_func([A, B, C])
    ir_module = IRModule({"vector_add_cpu": prim_func})

    # 编译到 LLVM（CPU）
    print(f"\nCompiling to LLVM target...")
    f = tvm.build(ir_module, target="llvm")

    # 准备数据（CPU 数组）
    a_np = np.random.uniform(0, 1, (n,)).astype("float32")
    b_np = np.random.uniform(0, 1, (n,)).astype("float32")

    a_cpu = tvm_tensor(a_np, device=tvm.cpu(0))
    b_cpu = tvm_tensor(b_np, device=tvm.cpu(0))
    c_cpu = tvm_tensor(np.zeros((n,), dtype="float32"), device=tvm.cpu(0))

    print(f"Executing kernel...")
    f(a_cpu, b_cpu, c_cpu)

    # 验证
    c_np = c_cpu.numpy()
    c_expected = a_np + b_np
    error = np.max(np.abs(c_np - c_expected))

    print(f"\nResults:")
    print(f"  Max error: {error:.2e}")
    print(f"  Test: {'PASSED ✓' if error < 1e-5 else 'FAILED ✗'}")

    # 性能测试
    print(f"\nPerformance (CPU):")
    start = time.time()
    for _ in range(num_runs):
        f(a_cpu, b_cpu, c_cpu)
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000
    print(f"  Time per run: {avg_time_ms:.3f} ms")
    print(f"  (CPU performance is much slower, this is expected)")

    return f


def troubleshooting_guide():
    """
    常见错误和解决方案
    """
    print("\n" + "=" * 60)
    print("Troubleshooting Guide")
    print("=" * 60)

    print("""
如果遇到以下错误，按如下处理：

1. ImportError: No module named 'tvm'
   → 解决：CPU 环境可用 `pip install apache-tvm`
   → CUDA 环境可按版本安装相应的 `apache-tvm-cuXXX`

2. RuntimeError: CUDA is not available
   → 原因：没有CUDA环境或TVM没有CUDA支持
   → 解决：用 target="llvm" 测试CPU版本
   → 或安装支持 CUDA 的 TVM 包（例如 apache-tvm-cu121）

3. RuntimeError: Check failed: entries.size() > 0
   → 原因：GPU kernel 编译失败，通常是因为不支持的操作
   → 解决：检查代码中的操作是否被TVM支持

4. 结果不对（error很大）
   → 原因：通常是数据类型不一致或边界问题
   → 解决：检查所有 dtype 是否都是 float32

5. 性能很低（GB/s 很小）
   → 这是正常的！这个例子没有任何优化
   → 第4章会教怎样优化到 50%+ 的带宽利用率
    """)


def parse_args():
    parser = argparse.ArgumentParser(description="Chapter 3 vector add demo")
    parser.add_argument(
        "--mode",
        choices=["auto", "cpu", "cuda", "all"],
        default="auto",
        help="Execution mode",
    )
    parser.add_argument("--n", type=int, default=1024, help="Vector length")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark runs for GPU path")
    return parser.parse_args()


def main():
    """主程序"""
    args = parse_args()
    print("\n" + "=" * 60)
    print("TileLang Quick Start: Vector Add Examples")
    print("=" * 60)

    # 检查CUDA可用性
    cuda_available = TVM_AVAILABLE and tvm.cuda().exist
    print(f"\nCUDA available: {cuda_available}")

    run_cuda = cuda_available and args.mode in {"auto", "cuda", "all"}
    run_cpu = args.mode in {"cpu", "all"} or not cuda_available

    if run_cuda:
        # 运行CUDA示例
        example_te_api(n=args.n, num_runs=args.runs)
        example_tir_api(n=args.n, num_runs=args.runs)

    if run_cpu:
        if not cuda_available:
            print("\nCUDA not available, using CPU fallback")
        elif args.mode == "cpu":
            print("\nCPU mode requested, using CPU fallback")
        example_cpu_fallback(n=args.n, num_runs=max(10, args.runs // 10))

    # 显示故障排查指南
    troubleshooting_guide()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("""
Next steps:
  1. 修改代码参数（n的大小），观察性能变化
  2. 查看生成的 PTX 代码，理解编译结果
  3. 在第4章中学习如何优化性能（shared memory, tiling）
  4. 在第8章实现矩阵乘法，看到真正的性能提升
    """)


if __name__ == "__main__":
    main()
