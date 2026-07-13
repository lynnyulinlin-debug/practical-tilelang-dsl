# practical-tilelang-dsl
## 实战 TileLang DSL：面向 AI Kernel 的编程、优化与迁移教程

这是一套面向真实工程的 TileLang 教程，目标是把 kernel 编程、性能优化、多后端迁移和端到端实战串成一条可复用的学习路径。

## 适合谁

- 已经会写 Python，想系统学习 GPU kernel 编程的人
- 想把手写 CUDA、Triton 或现有算子迁移到 TileLang 的工程师
- 做推理加速、编译器、系统优化或算子开发的人
- 想从“会跑一个 kernel”走到“能做迁移和调优”的读者

## 当前状态

- 10 章章节结构已经建立
- 第 1-5 章已拆出判断与过渡入口，第 6-7 章已拆出算子实战入口，第 10 章已拆出项目入口
- 关键示例和教学 notebook 已覆盖到前 10 章主线
- `tilelang-tvm` conda 环境下的 CPU 验证路径已跑通
- 当前机器没有可用 NVIDIA 驱动时，GPU 验证会跳过

## 章节总览

### 已成型

- 第 1 章：Kernel 与 DSL：AI 编译器的动机
- 第 2 章：GPU 硬件模型与执行层次
- 第 3 章：TileLang 快速入门与第一个 Kernel
- 第 4 章：分块、内存层次与数据复用
- 第 5 章：调度优化与性能分析

### 内容成型，持续增强中

- 第 6 章：基础算子实战：Vector Add 与 GEMM
- 第 7 章：高级算子实战：Attention 与融合算子
- 第 8 章：多后端适配与性能迁移
- 第 9 章：TileLang 与 Triton：设计与取舍
- 第 10 章：端到端工程实战：为大模型适配新硬件

第 10 章已拆成三层案例表述：

- 总述层：`某大模型相关子图`
- 算子层：`某推理场景中的部分算子`
- 公开案例层：`DeepSeek 相关子图（在可公开范围内）`

第 10 章教学 notebook：

- `examples/10_end_to_end_migration.ipynb`
- `examples/10_end_to_end_generic.ipynb`
- `examples/10_end_to_end_operator.ipynb`
- `examples/10_end_to_end_sweep.ipynb`

第 8-9 章教学 notebook：

- `examples/08_gemm_porting.ipynb`
- `examples/08_gemm_porting_overview.ipynb`
- `examples/08_gemm_porting_platform.ipynb`
- `examples/09_triton_comparison.ipynb`
- `examples/09_triton_comparison_overview.ipynb`
- `examples/09_triton_selection.ipynb`

## 脚本总览

### 第 1-5 章：判断与过渡入口

- `examples/01_why_dsl.py`
- `examples/02_hardware_model.py`
- `examples/03_quick_start.py`
- `examples/04_tiling_memory.py`
- `examples/05_schedule_optimize_demo.py`
- `examples/05_roofline_analysis.py`
- `examples/05_autotuning_demo.py`

### 第 6-7 章：算子实战入口

- `examples/06_basic_ops_demo.py`
- `examples/06_gemm_naive.py`
- `examples/06_gemm_blocked.py`
- `examples/07_attention_baseline.py`
- `examples/07_attention_reference.py`
- `examples/07_attention_blockwise.py`

### 第 10 章：端到端项目入口

- `examples/10_end_to_end_migration.py`
- `examples/10_end_to_end_generic.py`
- `examples/10_end_to_end_operator.py`
- `examples/10_end_to_end_sweep.py`

### 记录模板

- `examples/benchmark_record_template.py`
- `examples/benchmark_record_demo.py`

## 快速开始

1. 先看 [QUICKREF.md](QUICKREF.md) 了解如何运行示例和查找章节
2. 再看 [MAINTENANCE.md](MAINTENANCE.md) 了解维护规则和写作口径
3. 从第 3 章开始，配合 `examples/03_vector_add.py` 和 `examples/03_vector_add.ipynb`
4. 需要看项目级流程时，直接跳到第 10 章

## 维护入口

- 维护规范：`MAINTENANCE.md`
- 使用说明：`QUICKREF.md`
- 图表统计总表：[`chapters/FIGURE_TABLE_INDEX.md`](chapters/FIGURE_TABLE_INDEX.md)
