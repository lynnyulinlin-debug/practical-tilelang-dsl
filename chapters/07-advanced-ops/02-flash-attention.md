# 7.2 FlashAttention

## 这一节要解决什么问题

FlashAttention 的核心不是“更复杂”，而是通过重新组织计算过程，减少 attention 对中间矩阵和显存的依赖。

从教学角度看，它是第 4、5 章方法在复杂算子上的自然延伸：  
先分块，再控制片上数据，再把计算顺序改成更省 IO 的形式。

## 这一节要讲什么

### 1. IO-aware 设计

- 尽量让中间数据留在片上
- 减少 score 矩阵的写回

### 2. tile 化注意力

- Q / K / V 按块处理
- block-wise softmax
- 先算局部最大值和局部和，再做稳定归一化

### 3. 长序列优势

- 序列越长，收益越明显
- 这也是 FlashAttention 最典型的使用场景

### 4. 当前写作口径

- 这一节要把 IO-aware 作为核心概念说透
- 不要只停留在“更快”，要明确“更少写回”是为什么更快
- 如果后面补代码，优先展示块化 softmax 的思想，而不是完整工程细节

## 这里应该放的公式

**公式：Block-wise softmax**

> 这里要说明为什么 softmax 可以按块累积，而不必先形成完整 score 矩阵。
> 公式的目的，是解释“块内归一化 + 累积更新”为何成立。

> 讲义里建议把它写成“在线更新”的形式：
>
> - 每读入一个 `K/V` 块，更新当前 query block 的局部统计量
> - 维护 running max，避免数值溢出
> - 维护 running sum，保证最终归一化正确
> - 最后直接把累积结果写成输出，而不是把完整 `scores` 落回内存

## 一个更直观的流程

```text
for each query block:
  init running_max, running_sum, running_out
  for each key/value block:
    score_block = Q_block @ K_block^T
    update running_max
    rescale previous accumulators if needed
    accumulate softmax-weighted V_block
  write output block
```

这个流程的关键不是“省掉一条乘法”，而是把原来要落到慢层内存的中间矩阵，改成只在片上维护的局部状态。

## 这里应该配的图

**图：FlashAttention 的块化数据流图**

> 这里要展示 Q/K/V 如何按块进入计算，score 如何在片上被处理。
> 图里最好把“片上累积、减少显存峰值”标成明显的收益点。

## 这一节的结论

FlashAttention 的本质是 IO-aware attention，不是单纯“更快的 attention”。
它的优化方向是重排计算和减少中间状态落到慢层内存。

## 这一节应该留下的判断标准

- 如果序列越长收益越明显，说明当前问题更偏 IO 而不是纯计算
- 如果块化以后结果更稳定，说明在线归一化的数值路径是成立的
- 如果块太小，控制开销可能抵消收益
- 如果块太大，中间状态又会变重

## 这一节可以进一步追问的问题

- 在线更新为什么能保证数值稳定
- 块大小如何影响 IO 压缩和控制开销
- 为什么 FlashAttention 更像重排数据流，而不是简单加速某一步

## 常见误区

- 误区 1：FlashAttention 的本质是某个更快的 softmax
- 误区 2：只要使用 block-wise softmax 就一定更快
- 误区 3：块越大越好

FlashAttention 的关键是把 attention 变成更少依赖慢层内存的在线过程。

所以 FlashAttention 不是“块越大越好”，而是在数值稳定、IO 压缩和执行开销之间找平衡。

## 讲义提示

如果要补代码，优先演示两件事：

- `scores` 不再作为完整矩阵长期保存
- softmax 的归一化改成块级在线更新

这两点比把完整工程代码摆出来更适合教学。
