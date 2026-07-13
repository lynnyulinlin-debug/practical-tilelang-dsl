# 第 6 章导言：基础算子实战：Vector Add 与 GEMM

## 章节定位

这一章开始进入第一轮实战。  
目标不是再讲概念，而是把前面学到的分块、内存和调度方法落到基础算子上。

## 本章目标

- 完成基础算子从 baseline 到优化版的演进
- 用 GEMM 作为主线案例串起前面的方法
- 让读者第一次看到明确的性能提升
- 为后续高级算子章节建立信心和模板
- 让读者学会判断一个优化是“结构性改进”还是“参数性微调”

## 本章判断标准

读完这一章，读者应该能判断：

- 一个算子是先补正确性，还是先做结构优化
- 朴素 GEMM 的瓶颈主要来自哪里
- shared memory 优化什么时候已经足够
- 什么时候应该继续走到 fragment / Tensor Core 路径

## 内容块安排

### 6.1 Vector Add
- 最小算子
- 正确性验证
- 基础性能基准

### 6.2 Naive GEMM
- 朴素实现
- 性能瓶颈

### 6.3 Shared memory GEMM
- tile 和数据复用
- shared memory 版本

### 6.4 Tensor Core / Fragment GEMM
- 更接近硬件的实现方式
- 进一步性能提升

### 6.5 基础优化总结
- 向量化、对齐、布局
- 算子优化模板

### 6.6 本章总结
- 从“能写”走向“能优化”

## 图和公式占位

- Vector Add 基本流程图
- Naive vs optimized GEMM 数据流图
- GEMM 算术强度公式
- 性能对比图

这一章既有演进路径也有结果对比，后续建议补一张 GEMM 数据流图和一张性能对比图，把 baseline、shared memory 和 fragment 的差异讲清楚。

## 教学入口

- 对应 notebook：`examples/06_basic_ops_overview.ipynb`
- 对应总览脚本：`examples/06_basic_ops_demo.py`
- 对应拆分脚本：`examples/06_gemm_naive.py`、`examples/06_gemm_blocked.py`

## 与后续章节的关系

- 第 7 章会继续把复杂算子做深
- 第 8 章会把多后端要求加入这些算子
- 第 10 章会把这些基础算子作为工程迁移的组成部分
