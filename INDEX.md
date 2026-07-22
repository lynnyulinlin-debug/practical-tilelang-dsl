# TileLang 教程 - 完整索引

## 🎯 快速开始

1. 先看 [README.md](README.md) 了解项目和学习路径
2. 如果要从零建环境，先看 [QUICKREF.md](QUICKREF.md) 的 `0. 从零创建环境`
3. 从 [第 3 章](chapters/03-quick-start/) 和 `examples/03_vector_add.py` 开始
4. 再看 [第 4 章](chapters/04-tiling-memory/) 和 `examples/04_gemm_basic.py`
5. 继续按需进入 [第 5-7 章](chapters/05-schedule-optimize/) / [chapters/06-basic-ops/](chapters/06-basic-ops/) / [chapters/07-advanced-ops/](chapters/07-advanced-ops/)
6. 项目级流程直接看 [第 10 章](chapters/10-end-to-end/)

---

## 📚 完整文档导航

| 文件 | 内容 | 时间 | 难度 |
|------|------|------|------|
| [第 1 章](chapters/01-why-dsl/) | Kernel 与 DSL：AI 编译器的动机 / From Kernel to DSL | 2 小时 | ⭐ |
| [第 2 章](chapters/02-hardware-model/) | GPU 硬件模型与执行层次 | 3 小时 | ⭐⭐ |
| [第 3 章](chapters/03-quick-start/) | TileLang 快速入门与第一个 Kernel | 2 小时 | ⭐ |
| [第 4 章](chapters/04-tiling-memory/) | 分块、内存层次与数据复用 | 4 小时 | ⭐⭐ |
| [第 5 章](chapters/05-schedule-optimize/) | 调度优化与性能分析 | 3 小时 | ⭐⭐ |
| [第 6 章](chapters/06-basic-ops/) | 基础算子实战：Vector Add 与 GEMM | 3 小时 | ⭐⭐ |
| [第 7 章](chapters/07-advanced-ops/) | 高级算子实战：Attention 与融合算子 | 8 小时 | ⭐⭐⭐ |
| [第 8 章](chapters/08-multi-hardware/) | 多后端适配与性能迁移 | 3 小时 | ⭐⭐ |
| [第 9 章](chapters/09-triton-comparison/) | TileLang 与 Triton：设计与取舍 | 2 小时 | ⭐⭐ |
| [第 10 章](chapters/10-end-to-end/) | 端到端工程实战：为大模型适配新硬件 | 20-40 小时 | ⭐⭐⭐⭐ |

### 参考和指南

| 文件 | 用途 |
|------|------|
| [README.md](README.md) | 项目概览和学习路径 |
| [QUICKREF.md](QUICKREF.md) | 快速参考和运行口径 |
| [HANDOFF.md](HANDOFF.md) | 当前事实与交接 |
| [PROGRESS.md](PROGRESS.md) | 进度跟踪和待办 |

### 代码示例

| 文件 | 内容 |
|------|------|
| [examples/03_vector_add.py](examples/03_vector_add.py) | Vector Add 完整示例 |
| [examples/04_gemm_basic.py](examples/04_gemm_basic.py) | GEMM 优化演进 |

---

## 📊 项目统计

```
📈 规模
  • 总文档：18+ 个关键文件
  • 总代码：~6900 行 Markdown + Python
  • 磁盘占用：220 KB

✅ 完成度
  • 章节：10/10（骨架完成）
  • 持续增强：第 6、7 章
  • 持续收口：第 10 章
  • 代码示例：主线已覆盖，代表脚本与 notebook 入口已复核
  • 支持文档：README / QUICKREF / HANDOFF / MAINTENANCE / INDEX 已收口
  • 环境：`tilelang-tvm` 重建命令已同步，notebook 本地模块路径已修复

⭐ 质量评分
  • 框架完整性：⭐⭐⭐⭐⭐
  • 内容质量：⭐⭐⭐⭐
  • 代码质量：⭐⭐⭐
  • 综合评分：4/5
```

---

## 🚀 下一步

- 看 [HANDOFF.md](HANDOFF.md) 了解当前事实和优先级
- 看 [PROGRESS.md](PROGRESS.md) 了解任务颗粒和待办
- 看 [MAINTENANCE.md](MAINTENANCE.md) 了解维护规则
- 从 [第 3 章](chapters/03-quick-start/) 和 `examples/03_vector_add.py` 开始

*更新*：2026-07-22
