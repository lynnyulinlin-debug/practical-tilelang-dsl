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
- 第 1-5 章已成型，第 6、7 章持续增强，第 8、9 章已成型，第 10 章持续收口
- 代表脚本和教学 notebook 入口已完成复核，覆盖第 3、4、5、6、7、10 章主线
- 教学 notebook 入口已改为自定位仓库根目录，不再依赖手工 `PYTHONPATH`
- `tilelang-tvm` conda 环境（基于 TVM）的 CPU 验证路径已跑通，适合作为第 3 章和后续示例的运行底座
- 当前机器没有可用 NVIDIA 驱动时，GPU 验证会跳过

## 推荐阅读路线

### 路线 A：从零入门

1. 第 1 章 -> 第 3 章 -> 第 2 章
2. 第 4 章 -> 第 5 章
3. 第 6-7 章
4. 第 8-10 章

### 路线 B：先看性能和优化

1. 第 2 章 -> 第 4 章 -> 第 5 章
2. 第 6-7 章
3. 第 10 章

### 路线 C：先看性能分析

1. 第 5 章，先看 Roofline 和判断顺序
2. 第 7 章，看复杂算子的性能分析
3. 第 10 章，看端到端 benchmark 和项目报告

### 路线 D：先看迁移和项目

1. 第 8 章 -> 第 9 章 -> 第 10 章
2. 再回第 3-7 章补基础和优化方法

> 主线判断逻辑很简单：第 1 章先讲编译器，第 4 章讲访存和分块，第 5 章讲性能判断，第 8-10 章讲迁移和项目闭环。
> 如果你想要最短路径，就按：第 1 章 -> 第 3 章 -> 第 4 章 -> 第 5 章 -> 第 8 章 -> 第 9 章 -> 第 10 章。

## 工具选型

| 你更需要 | 优先看 | 这本教程为什么这样安排 |
| --- | --- | --- |
| 结构化优化、迁移和工程闭环 | TileLang | 把 tile、layout、memory tier 和 benchmark 放到同一条 DSL 路径里 |
| 快速写热点 kernel | Triton | 强调高层 tile 视角和局部优化效率 |
| 底层控制和 NVIDIA 生态 | CUDA | 提供最细粒度的硬件控制和成熟生态 |

TileLang 更偏结构、调度、迁移和证据闭环，Triton 更偏快速试验热点算子，CUDA 更偏底层控制和 NVIDIA 生态。

**读者可以这样理解：**

- 想解决“算子结构怎么组织、怎么迁移、怎么回填证据”，先看 TileLang
- 想快速验证热点 kernel 写法，先看 Triton
- 想直接控制底层执行和资源，或者强依赖 NVIDIA 生态，先看 CUDA

**对应到教程里，最能看出 TileLang 特色的是：**

- 第 4 章：原语如何把硬件层次显式写出来
- 第 8 章：多后端迁移时如何保持结构和口径一致
- 第 10 章：如何把项目、benchmark 和报告放进同一条闭环

换句话说，TileLang 在这本教程里承担的不是“替代 CUDA 或 Triton”，而是把它们各自擅长的事情放到一个更容易管理的工程路径里。

## 章节总览

| 章节 | 状态 | 作用 | Notebook |
| --- | --- | --- | --- |
| 第 1 章 | 已成型 | DSL、compiler、TileLang 动机 | 单 notebook |
| 第 2 章 | 已成型 | 桥前的硬件因果层：GPGPU、SIMT、内存层次、tile | 单 notebook |
| 第 3 章 | 已成型 | 最小闭环：环境、编译、验证、排错 | 单 notebook |
| 第 4 章 | 已成型 | `global / shared / fragment / tile` 映射 | 单 notebook |
| 第 5 章 | 已成型 | 并行、流水线、布局、Roofline、autotuning | 3 层 notebook |
| 第 6 章 | 持续增强 | 桥后的基础通行层：Vector Add 与 GEMM | 3 层 notebook |
| 第 7 章 | 持续增强 | 高级算子：Attention、fusion、性能分析 | 3 层 notebook |
| 第 8 章 | 已成型 | 多后端迁移与性能迁移 | 3 层 notebook |
| 第 9 章 | 已成型 | 桥上的路线对比层：TileLang vs Triton 取舍 | 3 层 notebook |
| 第 10 章 | 持续收口 | 端到端迁移、benchmark、report、sweep | 6 层 notebook |

### 已成型

- 第 1 章：Kernel 与 DSL：AI 编译器的动机
- 第 2 章：GPU 硬件模型与执行层次，作为桥前的硬件因果层
- 第 3 章：TileLang 快速入门与第一个 Kernel
- 第 4 章：分块、内存层次与数据复用
- 第 5 章：调度优化与性能分析

### 内容成型，持续增强中

- 第 6 章：基础算子实战：桥后的基础通行层
- 第 7 章：高级算子实战：Attention 与融合算子
- 第 8 章：多后端适配与性能迁移
- 第 9 章：桥上的路线对比层：TileLang 与 Triton
- 第 10 章：端到端工程实战：为大模型适配新硬件

第 10 章已拆成三层案例表述：

- 总述层：`某大模型相关子图`
- 算子层：`某推理场景中的部分算子`
- 公开案例层：`DeepSeek 相关子图（在可公开范围内）`

第 10 章教学 notebook：

- `examples/10_end_to_end_overview.ipynb`
- `examples/10_end_to_end_generic.ipynb`
- `examples/10_end_to_end_operator.ipynb`
- `examples/10_end_to_end_migration.ipynb`
- `examples/10_end_to_end_report.ipynb`
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
5. 教学 notebook 入口会自行定位仓库根目录，不再依赖手工设置 `PYTHONPATH`

## 维护入口

- 维护规范：`MAINTENANCE.md`
- 使用说明：`QUICKREF.md`
- 图表统计总表：[`chapters/FIGURE_TABLE_INDEX.md`](chapters/FIGURE_TABLE_INDEX.md)
