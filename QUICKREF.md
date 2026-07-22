# TileLang 教程快速开始卡

## 先看什么

1. [README.md](README.md) - 项目总览和学习入口
2. [HANDOFF.md](HANDOFF.md) - 当前事实来源和下一步优先级
3. [MAINTENANCE.md](MAINTENANCE.md) - 章节写作和维护规则

## 当前状态

- 主线已经切到目录式章节：`chapters/01-why-dsl/` 到 `chapters/10-end-to-end/`
- 第 1-5 章已成型，第 6、7 章持续增强，第 8、9 章已成型，第 10 章持续收口
- 主交接文档只有一份：`HANDOFF.md`
- 现阶段优先级是第 10 章收口、示例验证、性能数据补齐

## 推荐阅读路线

### 最推荐路线

1. 第 1 章：为什么需要 DSL、编译器和 TileLang
2. 第 3 章：最小张量入口、first kernel、验证和排错
3. 第 2 章：GPU 硬件模型、SIMT、内存层次和 tile
4. 第 4 章：shared memory、fragment 和数据复用
5. 第 5 章：并行、流水线、布局、Roofline、autotuning
6. 第 6 章：Vector Add 和 GEMM 基础优化
7. 第 7 章：Attention、FlashAttention、fusion
8. 第 8 章：多后端迁移
9. 第 9 章：TileLang 与 Triton 对比
10. 第 10 章：端到端工程实战

### 最短学习路径

1. 第 1 章：先建立 TileLang 为什么需要编译器
2. 第 3 章：再看 `lowering` 怎么把 DSL 接到后端
3. 第 4 章：再看原语怎么接住硬件层次
4. 第 5 章：再把这些结构翻成性能判断
5. 第 8 章：再看同一份逻辑怎么落到不同后端
6. 第 9 章：最后看 TileLang、Triton、CUDA 的路线差异
7. 第 10 章：把前面所有东西收成项目闭环

### 性能分析路线

1. 第 5 章
2. 第 7 章
3. 第 10 章

## 当前最重要的任务顺序

1. 收第 10 章正文和报告层口径，让它真正成为项目模板
2. 检查第 5-7 章 notebook 入口顺序，确保 `overview -> baseline/reference -> optimized` 一眼能走通
3. 验证 `examples/03_vector_add.py` 和 `examples/04_gemm_basic.py`
4. 继续补 benchmark 和图表
5. 再收口 `README.md`、`INDEX.md`、`PROGRESS.md`

## 文件入口

### 章节

- `chapters/01-why-dsl/`
- `chapters/02-hardware-model/`
- `chapters/03-quick-start/`
- `chapters/04-tiling-memory/`
- `chapters/05-schedule-optimize/`
- `chapters/06-basic-ops/`
- `chapters/07-advanced-ops/`
- `chapters/08-multi-hardware/`
- `chapters/09-triton-comparison/`
- `chapters/10-end-to-end/`

### 示例

- `examples/03_vector_add.py`
- `examples/04_gemm_basic.py`
- `examples/05_schedule_optimize_demo.py`
- `examples/06_basic_ops_demo.py`
- `examples/07_attention_baseline.py`
- `examples/10_end_to_end_overview.ipynb`
- `examples/10_end_to_end_generic.ipynb`
- `examples/10_end_to_end_operator.ipynb`
- `examples/08_gemm_porting.ipynb`
- `examples/10_end_to_end_migration.ipynb`
- `examples/10_end_to_end_report.ipynb`
- `examples/10_end_to_end_sweep.ipynb`
- `examples/09_triton_comparison.ipynb`

## 快速运行

### 0. 从零创建环境

如果你本机还没有 `tilelang-tvm` 环境，可以先创建它：

```bash
conda create -n tilelang-tvm python=3.11 -y
conda activate tilelang-tvm
python -m pip install --upgrade pip
pip install apache-tvm numpy jupyter nbconvert nbformat ipykernel
python -m ipykernel install --user --name tilelang-tvm --display-name "Python (tilelang-tvm)"
```

如果 notebook 里找不到本仓库的本地模块，再补一次：

```bash
export PYTHONPATH=/data/tilelang-in-action
```

在当前约定下，教学 notebook 本身也会尝试自定位仓库根目录并把它加入 `sys.path`。上面的 `PYTHONPATH` 只作为旧环境或特殊启动方式下的兜底手段。

### 1. 先确认环境

```bash
python -c "import tvm; print(tvm.__version__)"
```

如果 `import tvm` 失败，先别看后面的 GPU 路径，直接回到第 3 章环境说明和 `HANDOFF.md` 里的收口建议。

### 2. 跑最小示例

```bash
cd examples
python 03_vector_add.py --mode auto
python 04_gemm_basic.py --mode naive
python 04_gemm_basic.py --mode shared
```

### 3. 看教学 notebook

```bash
jupyter notebook 03_vector_add.ipynb
jupyter notebook 04_gemm_basic.ipynb
jupyter notebook 07_attention_baseline.ipynb
jupyter notebook 08_gemm_porting.ipynb
jupyter notebook 09_triton_comparison.ipynb
jupyter notebook 10_end_to_end_migration.ipynb
```

## 读文档的顺序

1. `README.md` 看项目现在写到哪了
2. `HANDOFF.md` 看当前事实、风险和下一步
3. `MAINTENANCE.md` 看章节和代码怎么维护
4. 再进章节正文

## 维护口径

- 主示例以 `.py` 为准，notebook 负责讲解和演示
- 教学 notebook 如果导入仓库内本地模块，入口代码必须自定位仓库根目录并加入 `sys.path`
- 章节优先补正文，再补代码，再补 benchmark
- 图和公式可以先占位，但要写清楚它们支持的结论
- 不要再把旧的单文件章节当主线入口

## 常见阻塞

- `import tvm` 失败
  - 先确认当前环境是否就是 `tilelang-tvm`
  - 再看 `HANDOFF.md` 里的环境和示例验证说明

- GPU 不可用
  - 先跑 CPU fallback
  - 不要因为 GPU 不通就停在示例验证之前

- 文档口径不一致
  - 以 `HANDOFF.md` 为准
  - 其他状态页只做辅助参考
- notebook 找不到本地模块
  - 优先检查 notebook 入口是否已经自定位仓库根目录
  - 旧环境下再临时设置 `PYTHONPATH=/data/tilelang-in-action`

## 最短检查清单

- [ ] 读过 `HANDOFF.md`
- [ ] 知道当前优先级是第 7 章、第 8-10 章、示例验证、benchmark
- [ ] 能跑 `examples/03_vector_add.py`
- [ ] 能跑 `examples/04_gemm_basic.py`
- [ ] 知道 `README.md`、`INDEX.md`、`PROGRESS.md` 需要一起收口
