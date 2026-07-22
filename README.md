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

## 快速开始

1. 从 [QUICKREF.md](QUICKREF.md) 看运行口径
2. 从 [MAINTENANCE.md](MAINTENANCE.md) 看维护规则
3. 从第 3 章和 `examples/03_vector_add.py` 入手
4. 项目级流程直接看第 10 章

## 阅读路径

**主线**

1. 第 1 章 -> 第 3 章 -> 第 4 章 -> 第 5 章
2. 第 6-7 章
3. 第 8-10 章

**项目线**

1. 第 8 章 -> 第 9 章 -> 第 10 章
2. 再回第 3-7 章补基础和优化方法

> 核心顺序：编译器动机 -> 快速入门 -> 分块与内存 -> 性能判断 -> 迁移与闭环。

## 工具选型

| 你更需要 | 优先看 |
| --- | --- |
| 结构化优化、迁移和工程闭环 | TileLang |
| 快速写热点 kernel | Triton |
| 底层控制和 NVIDIA 生态 | CUDA |

> TileLang 更偏结构、调度、迁移和证据闭环；Triton 更偏快速试验热点算子；CUDA 更偏底层控制和 NVIDIA 生态。

## 维护入口

- 维护规范：`MAINTENANCE.md`
- 使用说明：`QUICKREF.md`
- 图表统计总表：[`chapters/FIGURE_TABLE_INDEX.md`](chapters/FIGURE_TABLE_INDEX.md)
- notebook 入口会自行定位仓库根目录，不再依赖手工 `PYTHONPATH`
