# TileLang 教程快速参考指南

## 📚 文档导航

### 核心学习路径

```
START
  ↓
README.md (项目概览)
  ↓
[选择你的起点]
  ├─→ 完全新手 → 第1章 → 第2章 → 第3章
  ├─→ 有CUDA经验 → 第4章 开始
  └─→ 想快速上手 → 第3章 (跳过第1-2章)
  ↓
第4-5章 (核心概念)
  ↓
examples/ (代码示例)
  ↓
第6-10章 (进阶/实战)
  ↓
END
```

---

## 🗂️ 文件结构

```
tilelang-in-action/
│
├─ 📖 文档
│  ├─ README.md                   ← 开始这里
│  ├─ MAINTENANCE.md              ← 维护规范
│  └─ QUICKREF.md                 ← 本文件
│
├─ 📚 章节内容 (chapters/)
│  ├─ 01-why-dsl/                ← ✓ Kernel 与 DSL：AI 编译器的动机
│  ├─ 02-hardware-model/         ← ✓ GPU 硬件模型与执行层次
│  ├─ 03-quick-start/            ← ✓ TileLang 快速入门与第一个 Kernel
│  ├─ 04-tiling-memory/          ← ✓ 分块、内存层次与数据复用
│  ├─ 05-schedule-optimize/      ← ✓ 调度优化与性能分析
│  ├─ 06-basic-ops/              ← ◐ 内容成型：基础算子实战，待 benchmark
│  ├─ 07-advanced-ops/           ← ◐ 内容成型：高级算子实战，待 benchmark
│  ├─ 08-multi-hardware/         ← ◐ 内容成型：多后端适配与迁移记录
│  ├─ 09-triton-comparison/      ← ◐ 内容成型：TileLang 与 Triton：设计与取舍
│  └─ 10-end-to-end/             ← ◐ 内容成型：端到端工程实战
│
├─ 💻 代码示例 (examples/)
│  ├─ 03_vector_add.py           ← ✓ Vector Add
│  ├─ 03_vector_add.ipynb        ← ✓ Vector Add 教学 notebook
│  ├─ 04_gemm_basic.py           ← ✓ GEMM 基础版
│  ├─ 04_gemm_basic.ipynb         ← ✓ GEMM 教学 notebook
│  ├─ 01_why_dsl.ipynb           ← ✓ 第 1 章教学 notebook
│  ├─ 02_hardware_model.ipynb    ← ✓ 第 2 章教学 notebook
│  ├─ 01_why_dsl.py              ← ✓ 第 1 章判断脚本
│  ├─ 02_hardware_model.py       ← ✓ 第 2 章判断脚本
│  ├─ 03_quick_start.py          ← ✓ 第 3 章判断脚本
│  ├─ 04_tiling_memory.py        ← ✓ 第 4 章判断脚本
│  ├─ 05_schedule_optimize_demo.py ← ✓ 调度优化总览脚本
│  ├─ 05_roofline_analysis.py    ← ✓ Roofline 拆分脚本
│  ├─ 05_autotuning_demo.py      ← ✓ 调优建议拆分脚本
│  ├─ 06_basic_ops_overview.ipynb ← ✓ 基础算子教学 notebook
│  ├─ 06_basic_ops_demo.py       ← ✓ 基础算子总览脚本
│  ├─ 06_gemm_naive.py           ← ✓ Naive GEMM 拆分脚本
│  ├─ 06_gemm_blocked.py         ← ✓ Blocked GEMM 拆分脚本
│  ├─ 07_attention_baseline.ipynb ← ✓ Attention 教学 notebook
│  ├─ 07_attention_baseline.py   ← ✓ Attention 总览脚本
│  ├─ 07_attention_reference.py  ← ✓ Attention reference 拆分脚本
│  ├─ 07_attention_blockwise.py  ← ✓ Attention blockwise 拆分脚本
│  ├─ 08_gemm_porting.ipynb      ← ✓ GEMM 跨平台迁移教学 notebook
│  ├─ 08_gemm_porting_overview.ipynb ← ✓ 第 8 章总览层教学 notebook
│  ├─ 08_gemm_porting_platform.ipynb ← ✓ 第 8 章平台层教学 notebook
│  ├─ 09_triton_comparison.ipynb ← ✓ Triton 对比教学 notebook
│  ├─ 09_triton_comparison_overview.ipynb ← ✓ 第 9 章总览层教学 notebook
│  ├─ 09_triton_selection.ipynb  ← ✓ 第 9 章选型层教学 notebook
│  ├─ 10_end_to_end_migration.ipynb ← ✓ 端到端迁移教学 notebook
│  ├─ 10_end_to_end_generic.ipynb ← ✓ 第 10 章总述层教学 notebook
│  ├─ 10_end_to_end_operator.ipynb ← ✓ 第 10 章算子层教学 notebook
│  ├─ 10_end_to_end_sweep.ipynb  ← ✓ 第 10 章趋势扫描教学 notebook
│  ├─ 10_end_to_end_migration.py ← ✓ 端到端迁移总览脚本（兼容入口）
│  ├─ 10_end_to_end_generic.py   ← ✓ 总述层项目脚本
│  ├─ 10_end_to_end_operator.py  ← ✓ 算子层项目脚本
│  ├─ 10_end_to_end_sweep.py     ← ✓ 趋势扫描脚本
│  ├─ 10-end-to-end/07-project-guide.md ← ✓ 端到端项目导读
│  ├─ 10-end-to-end/08-project-report.md ← ✓ 端到端项目报告
│  ├─ benchmark_record_template.py ← ✓ 统一 benchmark 记录模板
│  ├─ benchmark_record_demo.py    ← ✓ benchmark 记录示例
│  └─ 其余示例待补充
│
└─ 📋 历史文档
   └─ 无（已清理）
```

---

## ⏱️ 学习时间估计

### 按章节

| 章节 | 内容 | 预计时间 | 难度 |
|------|------|---------|------|
| 第1章 | 为什么需要 DSL | 2 小时 | ⭐ 简单 |
| 第2章 | 硬件基础 | 3 小时 | ⭐⭐ 中等 |
| 第3章 | 快速入门 | 2 小时 | ⭐ 简单 |
| 第4章 | 分块与内存 | 4 小时 | ⭐⭐ 中等 |
| 第5章 | 调度与优化 | 3 小时 | ⭐⭐ 中等 |
| 第6章 | 基础算子实战 | 3 小时 | ⭐⭐ 中等 |
| 第7章 | 高级算子实战 | 8 小时 | ⭐⭐⭐ 难 |
| 第8章 | 多后端适配 | 3 小时 | ⭐⭐ 中等 |
| 第9章 | Triton 对比 | 2 小时 | ⭐⭐ 中等 |
| 第10章 | 端到端实战 | 20-40 小时 | ⭐⭐⭐⭐ 很难 |

### 学习路径

| 路径 | 总时间 | 覆盖范围 |
|------|--------|---------|
| 快速入门 | 1 周 | 第1-5章 + 代码示例 |
| 进阶应用 | 2-3 周 | 第1-9章 + 完整案例 |
| 完全学习 | 4-6 周 | 所有章节 + 性能数据 + 项目实战 |

---

## 🎯 快速查找

### 我想学...

**理论基础**
- 为什么用 DSL？ → 第 1 章
- GPU 怎样执行代码？ → 第 2 章
- 为什么分块很重要？ → 第 2.4 + 第 4 章
- 查看第 1 章判断脚本 → `examples/01_why_dsl.py`
- 查看第 2 章判断脚本 → `examples/02_hardware_model.py`
- 查看第 3 章判断脚本 → `examples/03_quick_start.py`
- 查看第 4 章判断脚本 → `examples/04_tiling_memory.py`
- 查看第 5 章总览脚本 → `examples/05_schedule_optimize_demo.py`
- 查看第 5 章 Roofline 分析 → `examples/05_roofline_analysis.py`
- 查看第 5 章调优建议 → `examples/05_autotuning_demo.py`

**实践技能**
- 写第一个程序 → 第 3 章 + examples/03_vector_add.ipynb
- 理解 DSL 为什么重要 → 第 1 章 + examples/01_why_dsl.ipynb
- 理解 GPU 硬件模型 → 第 2 章 + examples/02_hardware_model.ipynb
- 优化矩阵乘法 → 第 4-5 章 + examples/04_gemm_basic.ipynb
- 分析调度优化 → 第 5 章 + examples/05_schedule_optimize_demo.py
- 基础算子总览 → 第 6 章 + examples/06_basic_ops_overview.ipynb
- 查看基础算子总览 → 第 6 章 + examples/06_basic_ops_demo.py
- 查看 Naive GEMM 拆分脚本 → 第 6 章 + examples/06_gemm_naive.py
- 查看 Blocked GEMM 拆分脚本 → 第 6 章 + examples/06_gemm_blocked.py
- 理解 Attention / FlashAttention → 第 7 章 + examples/07_attention_baseline.ipynb
- 查看 Attention 总览 → 第 7 章 + examples/07_attention_baseline.py
- 查看 Attention reference 拆分脚本 → 第 7 章 + examples/07_attention_reference.py
- 查看 Attention blockwise 拆分脚本 → 第 7 章 + examples/07_attention_blockwise.py
- 理解跨平台 GEMM 迁移 → 第 8 章 + examples/08_gemm_porting.ipynb
- 理解跨平台 GEMM 总览层 → 第 8 章 + examples/08_gemm_porting_overview.ipynb
- 理解跨平台 GEMM 平台层 → 第 8 章 + examples/08_gemm_porting_platform.ipynb
- 理解 TileLang vs Triton → 第 9 章 + examples/09_triton_comparison.ipynb
- 理解 TileLang vs Triton 总览层 → 第 9 章 + examples/09_triton_comparison_overview.ipynb
- 理解 TileLang vs Triton 选型层 → 第 9 章 + examples/09_triton_selection.ipynb
- 理解端到端迁移流程 → 第 10 章 + examples/10_end_to_end_migration.ipynb
- 理解总述层项目 → 第 10 章 + examples/10_end_to_end_generic.ipynb
- 理解算子层项目 → 第 10 章 + examples/10_end_to_end_operator.ipynb
- 理解趋势扫描 → 第 10 章 + examples/10_end_to_end_sweep.ipynb
- 查看端到端迁移总览脚本 → 第 10 章 + examples/10_end_to_end_migration.py
- 查看总述层项目脚本 → 第 10 章 + examples/10_end_to_end_generic.py
- 查看算子层项目脚本 → 第 10 章 + examples/10_end_to_end_operator.py
- 查看趋势扫描脚本 → 第 10 章 + examples/10_end_to_end_sweep.py
- 切换第 10 章案例表述层 → `examples/10_end_to_end_migration.py --case-style {generic,operator,public}`
- 设置第 10 章子图标签 → `examples/10_end_to_end_migration.py --subgraph-label "..."`
- 做第 10 章序列长度扫描 → `examples/10_end_to_end_migration.py --sweep`
- 了解第 10 章正文与脚本分工 → 第 10 章 + chapters/10-end-to-end/07-project-guide.md
- 查看第 10 章项目报告 → 第 10 章 + chapters/10-end-to-end/08-project-report.md
- 记录 benchmark 模板 → examples/benchmark_record_template.py

**性能优化**
- 怎样分析瓶颈？ → 第 5 章 5.4 (Roofline 模型)
- 怎样用 autotuning？ → 第 5 章 5.5
- 怎样测性能？ → 第 5 章 5.6 + examples/

**多硬件**
- 如何支持 AMD GPU？ → 第 8 章
- 如何支持昇腾 NPU？ → 第 8 章
- 同一代码多硬件运行 → 第 8 章

**对比和决策**
- TileLang vs Triton？ → 第 9 章
- 什么时候选 TileLang？ → 第 1 章 1.6
- 什么时候选手写 CUDA？ → 第 1 章 1.6

**遇到问题**
- 代码编译失败 → 第 3 章 3.5
- 性能达不到预期 → 第 5 章
- 环境配置问题 → 第 3 章 3.1

---

## 🔑 核心概念速查

### 内存相关
```
延迟排序（快到慢）：
  寄存器 (1ns) → Shared Memory (5ns) → L2 Cache (30ns) → HBM (200ns)

对应 TileLang：
  T.alloc_fragment → T.alloc_shared → (自动) → T.alloc (全局)

访问合并：
  ✓ 线程 i 访问 address[i]     (连续，合并)
  ✗ 线程 i 访问 address[i*stride] (步进，可能非合并)
  ✗ 线程 i 访问 random_address[i] (随机，严重非合并)
```

### 并行相关
```
执行层次：
  Grid (多 block) → Block (多 warp) → Warp (32 线程) → Thread

对应 TileLang：
  T.parallel(..., axis="blockIdx.x") → blockIdx.y → threadIdx.x
  (具体映射取决于代码结构)

分块三层：
  Level 1: Block 间分块 (Grid)
  Level 2: Block 内分块 (Shared Memory)
  Level 3: 线程内分块 (寄存器)
```

### 性能相关
```
Roofline 模型：
  AI = FLOP / Byte (算术强度)
  如果 AI × 带宽 > 计算峰值 → 计算受限
  否则 → 内存受限

优化方向：
  内存受限 → 增加数据复用 (shared memory, tiling)
  计算受限 → 增加并行度或用 Tensor Core
```

---

## 📋 常用代码模板

### Vector Add (最简单)
```python
from tvm import te
from tvm.ir import IRModule

A = te.placeholder((n,), name="A", dtype="float32")
B = te.placeholder((n,), name="B", dtype="float32")
C = te.compute((n,), lambda i: A[i] + B[i], name="C")
prim_func = te.create_prim_func([A, B, C])
mod = IRModule({"vector_add": prim_func})
```

### 分块 GEMM 框架
```python
from tvm import te

# 先用 TE 定义计算图，再 lowered 成 PrimFunc
A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
prim_func = te.create_prim_func([A, B, C])
```

> 说明：不同 TVM 版本的底层 TIR / `tirx` API 不完全一致。这里优先保留能跨版本跑通的 TE -> PrimFunc 写法。

### 性能基准
```python
def benchmark(kernel, a, b, c):
    # 预热
    for _ in range(10):
        kernel(a, b, c)
    tvm.cuda().synchronize()
    
    # 测试
    start = time.time()
    for _ in range(100):
        kernel(a, b, c)
    tvm.cuda().synchronize()
    end = time.time()
    
    # 计算指标
    avg_time = (end - start) / 100
    tflops = flops / avg_time / 1e12
    return tflops
```

### Benchmark 模板

```md
## Benchmark 记录模板

### 基本信息
- 场景名称：
- 算子名称：
- 平台名称：
- target：
- 版本信息：
- 驱动 / 运行时版本：

### 测试参数
- dtype：
- shape / M N K：
- batch：
- warmup 次数：
- 统计次数：

### 性能指标
- latency：
- throughput：
- TFLOPS：
- GB/s：
- 峰值显存：
- 平均显存：

### 备注
- baseline：
- 优化点：
- 风险 / 限制：
```

```python
benchmark_record = {
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
```

> 说明：第 8 章的跨平台迁移表、第 7 章的 Attention benchmark 表、第 10 章的端到端 benchmark 表，都应该尽量复用这套字段，只按场景增删，不要各自发明格式。

### 统一 benchmark 代码模板

```python
from examples.benchmark_record_template import new_benchmark_record, set_benchmark_record

record = set_benchmark_record(
    scenario="attention benchmark",
    operator="Attention",
    platform="NVIDIA",
    target="cuda",
    dtype="float16",
    shape="B=1,S=4096,D=128",
)
```

> 说明：如果后续要在章节里记录 benchmark，优先从这个模板生成，再按章节补字段。

---

## 🛠️ 常见任务

### 任务 1：运行 Vector Add 示例

```bash
# 1. 检查环境
python -c "import tvm; print(f'TVM: {tvm.__version__}, CUDA: {tvm.cuda().exist}')"

# 2. 运行示例（脚本版）
cd examples
python 03_vector_add.py --mode auto

# 2b. 运行教学 notebook
jupyter notebook 03_vector_add.ipynb

# 3. 查看输出
# 应该看到 "Test PASSED"；如果没有 TVM 或 CUDA，也会看到 CPU fallback
```

前置要求：

- 建议在 `tilelang-tvm` 环境中运行；如果没有 TVM，脚本会走 NumPy fallback
- 如果没有 CUDA 环境，优先改用 `--mode cpu` 或 `--mode auto`
- 如果只是想先看代码结构，不需要先跑完整 GPU 版本

### 任务 2：修改 GEMM 的分块大小

```python
# 在 04_gemm_basic.ipynb 或 04_gemm_basic.py 中找到这行
BM, BN, BK = 64, 64, 32  # ← 改这里

# 试试不同的值
BM, BN, BK = 32, 32, 16   # 小块，更多 block
BM, BN, BK = 128, 128, 64 # 大块，更少 block

# 观察性能变化
python 04_gemm_basic.py --mode shared --m 1024 --n 1024 --k 1024
```

### 任务 3：用 Roofline 分析 kernel

```python
# 计算算术强度
M, K, N = 1024, 1024, 1024
flops = 2 * M * K * N
bytes_accessed = (M*K + K*N + M*N) * 4  # float32
AI = flops / bytes_accessed

# 判断瓶颈
peak_compute = 312e12  # A100
peak_memory = 2000e9

if AI * peak_memory > peak_compute:
    print("Compute-bound")
else:
    print("Memory-bound")
```

### 任务 4：对比不同优化

```bash
# 运行所有版本
python 04_gemm_basic.py --mode naive    # 朴素版本
python 04_gemm_basic.py --mode shared   # 加 shared memory
python 04_gemm_basic.py --mode all      # 两种实现 + 对比

# 观察性能提升
```

### 任务 5：理解 Attention 的数据流

```bash
# 1. 打开教学 notebook
jupyter notebook 07_attention_baseline.ipynb

# 2. 观察 Q/K/V、scores、probs、out 的形状
# 3. 对照第 7 章正文理解 FlashAttention 的动机
```

### 任务 6：理解跨平台 GEMM 迁移

```bash
# 1. 打开教学 notebook
jupyter notebook 08_gemm_porting.ipynb

# 2. 观察迁移记录模板
# 3. 对照第 8 章理解 target / tile / layout 的作用
```

### 任务 7：理解 TileLang vs Triton

```bash
# 1. 打开教学 notebook
jupyter notebook 09_triton_comparison.ipynb

# 2. 对照第 9 章理解抽象层级和选型标准
# 3. 观察为什么“快速实验”和“跨平台迁移”会导向不同路线
```

### 任务 8：理解端到端迁移流程

```bash
# 1. 打开教学 notebook
jupyter notebook 10_end_to_end_migration.ipynb

# 2. 再看总述层 / 算子层 / 趋势扫描三个 notebook
jupyter notebook 10_end_to_end_generic.ipynb
jupyter notebook 10_end_to_end_operator.ipynb
jupyter notebook 10_end_to_end_sweep.ipynb

# 3. 对照第 10 章理解子图、优先级和 benchmark 分层
# 4. 观察端到端项目如何收口成迁移模板
```

### 任务 9：查看 benchmark 记录示例

```bash
# 1. 打开记录示例
python examples/benchmark_record_demo.py

# 2. 对照第 6、7、10 章里的记录样例理解字段用法
```

---

## ❓ 常见问题快速解决

### 环境安装速查
```bash
# 推荐先进入独立环境
conda activate tilelang-tvm

# 只验证 CPU 路径
pip install apache-tvm

# 需要 CUDA 路径时，换成对应 CUDA 版本的 TVM 包
pip install apache-tvm-cu121
```

> 规则：
> - 先保证 `import tvm` 成功，再看示例
> - 没有 NVIDIA 驱动时，GPU 路径会跳过，优先跑 CPU fallback
> - 环境不通时，先看第 3 章的环境与排错

### "ImportError: No module named 'tvm'"
```bash
pip install apache-tvm
```

### "RuntimeError: CUDA is not available"
```python
# 方案 1: 用 CPU 测试
f = tvm.build(ir_module, target="llvm")

# 方案 2: 检查 CUDA 环境
nvidia-smi  # 看是否有 GPU

# 方案 3: 直接用脚本的 CPU fallback
python examples/03_vector_add.py --mode cpu
```

### "性能很低（只有几 GB/s）"
这是正常的！Vector Add 没有优化。
- 第 4 章教怎样优化到 50%+ 峰值
- 看 examples/04_gemm_basic.py 对比

### "不知道从哪开始"
1. 先读 README.md (5 分钟)
2. 然后读第 1 章 (2 小时)
3. 再读第 3 章 (2 小时)
4. 运行 examples/03_vector_add.py --mode auto (30 分钟)

---

## 🎓 学习检查清单

### 基础阶段完成？
- [ ] 理解为什么需要 DSL（第 1 章）
- [ ] 懂 GPU 执行层次（第 2 章）
- [ ] 会写 Vector Add（第 3 章）
- [ ] 理解分块为什么重要（第 4 章）

### 进阶阶段完成？
- [ ] 能用 Roofline 分析瓶颈（第 5 章）
- [ ] 知道如何优化矩阵乘法（第 5 章）
- [ ] 理解多硬件差异（第 6 章）
- [ ] 知道 Triton vs TileLang 的权衡（第 7 章）

### 实战能力？
- [ ] 能实现 GEMM（第 8 章）
- [ ] 能实现 Attention（第 9 章）
- [ ] 能做端到端迁移（第 10 章）

---

## 📞 获取帮助

### 遇到问题的解决方案

1. **查文档**：在本文档的"常见问题"部分查找
2. **看示例**：examples/ 中有完整的工作代码
3. **检查错误**：优先对照本文件的常见问题和第 3 章排错节
4. **提问讨论**：[待建立论坛]

### 提反馈的方式

- 发现错误 → 说明位置和问题
- 难度过高 → 说明哪里卡住了
- 代码无法运行 → 给出完整错误信息和环境
- 性能不对 → 给出实测数据和硬件信息

---

## 🎯 下一步建议

### 如果你有 1 小时
→ 读第 1 章，了解为什么需要 DSL

### 如果你有 1 天
→ 读第 1-3 章，写一个 Vector Add

### 如果你有 1 周
→ 读第 1-5 章，理解优化方法，运行所有代码示例

### 如果你有 1 个月
→ 完整学习所有章节，做一个小项目（如优化 GEMM）

### 如果你有 3 个月
→ 深入学习，参与更复杂的项目（如大模型迁移）

---

## 📊 进度跟踪模板

用这个表格跟踪你的学习进度：

```markdown
## 我的学习进度

### 第 1-5 章（基础）
- [ ] 第 1 章：为什么需要 DSL？ (预计 2 小时)
- [ ] 第 2 章：硬件基础 (预计 3 小时)
- [ ] 第 3 章：快速入门 (预计 2 小时)
- [ ] 例子：Vector Add
- [ ] 第 4 章：分块与内存 (预计 4 小时)
- [ ] 例子：GEMM 基础
- [ ] 第 5 章：调度与优化 (预计 3 小时)

### 完成日期：_______
### 花费时间：_______
### 难度反馈：_______
```

---

**最后更新**：2024-06-20
**维护者**：[待定]
**版本**：0.1

💡 **提示**：在浏览器中用 Ctrl+F 搜索关键词，快速找到需要的内容。

图表统计总表：`chapters/FIGURE_TABLE_INDEX.md`
