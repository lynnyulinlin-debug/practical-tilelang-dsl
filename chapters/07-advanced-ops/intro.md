# 第 7 章导言：高级算子实战：Attention 与融合算子

## 章节定位

这一章开始处理复杂算子。  
这里的重点不是单点性能，而是阶段划分、融合策略、数据复用和长链路调度。

## 本章目标

- 通过 Attention 类算子理解复杂数据流
- 解释为什么需要 FlashAttention
- 说明融合算子为什么能减少内存流量
- 让读者看到 TileLang 在复杂算子上的表达力
- 把“复杂但可控”的优化方法讲成可复用模板

## 本章判断标准

读完这一章，读者应该能判断：

- Attention 的主要成本到底来自数据流还是算式本身
- FlashAttention 解决的是哪类瓶颈
- 哪些步骤适合融合，哪些步骤不该强行融合
- 复杂调度应该先改什么、后改什么

## 当前写作口径

- 这一章不追求把每个 attention 变体都写成完整实现
- 优先讲清数据流、成本和优化方向
- 如果没有可用 GPU 驱动，仍然可以把它当成结构化分析章来读
- 这一章还要让读者知道“融合到什么程度就够了”，而不是一味追求更多操作堆叠

## 内容块安排

### 7.1 Attention baseline
- 标准 attention 结构
- 计算和内存开销

### 7.2 FlashAttention
- IO-aware 设计
- tile 化和 block-wise softmax

### 7.3 Fused Attention
- Attention + RoPE / RMSNorm / masking
- 中间张量消除

### 7.4 复杂调度与优化
- shared memory
- pipeline
- layout

### 7.5 性能分析
- 显存占用
- 吞吐
- 延迟

### 7.6 本章总结
- 从基础算子过渡到复杂算子
- 收束成“系统化优化模板”

## 图和公式占位

- Attention 数据流图
- baseline vs FlashAttention 对比图
- 融合前后图
- 复杂算子性能图

这一章的数据流更长、阶段更多，后续建议补一张 Attention 主路径图和一张融合前后对比图，避免读者只记住算法名。

## 与后续章节的关系

- 第 8 章会把这些算子迁移到更多后端
- 第 9 章会回到 Triton 做设计对比
- 第 10 章会把这些能力放进端到端工程

## 写作提醒

这一章要延续第 4、5 章的工程风格：先说成本，再说结构，最后说收益。  
不要把注意力放在算法名词堆叠上，而要让读者看懂为什么这些优化会成立。

## 教学入口

- 对应 notebook：`examples/07_attention_baseline.ipynb`
- 对应总览脚本：`examples/07_attention_baseline.py`
- 对应拆分脚本：`examples/07_attention_reference.py`、`examples/07_attention_blockwise.py`
