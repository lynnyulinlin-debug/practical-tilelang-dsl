# 7.5 性能分析

## 要解决什么问题

复杂算子不能只看“写没写出来”，必须看收益是否真实。

这一节要把“感觉快了”变成“数据上真的快了”，否则前面的优化都只能算是实现技巧，不算工程结论。

## 这一节要讲什么

### 1. 指标

- 显存占用
- 吞吐
- 延迟

### 2. 基准模板

- baseline 实现
- FlashAttention 实现
- fused attention 实现
- 相同输入规模
- 相同硬件环境

### 3. 对比维度

- 短序列和长序列
- 单 batch 和多 batch
- 显存峰值和稳态吞吐
- 端到端延迟

### 4. 结论模板

- 优化是否有效
- 收益来自哪里
- 代价是什么

### 5. 当前写作口径

- 先统一比较条件，再谈结果
- 先看显存和延迟，再看吞吐
- 最后把结论收束成“适用场景”和“边界条件”

## 这里应该放的图

**图：复杂算子的性能对比图**

> 这里要展示不同实现方式在吞吐和显存占用上的差异。
> 如果补图，建议把短序列和长序列分开，不要混在一张图里。

这里的图要承担“单点结果”和“规模趋势”的双重说明，后续优先把不同序列长度拆开画。

## 这里应该放的表

**表：Attention benchmark 记录表**

> 这里要统一记录模型规模、序列长度、batch size、显存占用、延迟和吞吐。
> 表格的目的，是让后面任何一次复测都能对齐口径。
> 字段建议直接复用 [QUICKREF.md](../../QUICKREF.md) 里的 benchmark 模板，只按 attention 场景增删字段。

这张表最好和图配套出现，一张图看趋势，一张表看条件，避免只看数值不看上下文。

## 这里应该放的记录样例

> 这里可以直接引用统一模板，先给出一个 attention 场景的最小样例。

```python
from examples.benchmark_record_template import set_benchmark_record

record = set_benchmark_record(
    scenario="chapter 7 attention benchmark",
    operator="Attention",
    platform="NVIDIA",
    target="cuda",
    dtype="float16",
    shape="B=1,S=4096,D=128",
    baseline="standard attention",
    optimization="FlashAttention + fusion",
)
```

## 这一节的结论

高级算子的价值不在“复杂”，而在“复杂但能被系统化优化”。
只要指标和口径一致，复杂算子同样可以形成稳定的 benchmark 结论。

## 这一节可以进一步追问的问题

- 你比较的是哪一类成本：显存、吞吐还是延迟
- baseline 和优化版是否在同一输入、同一硬件、同一统计口径下测量
- 长序列和短序列是否应该分开看

## 常见误区

- 误区 1：只看吞吐就足够
- 误区 2：只要 benchmark 变快，优化就一定成立
- 误区 3：不同输入规模的结果可以直接混在一起看

性能分析的价值，是把“感觉快了”变成可复核的结果。
