# 7.1 Attention baseline

## 这一节要解决什么问题

在进入 FlashAttention 之前，必须先把标准 attention 的数据流和成本说清楚。  
否则后面所有优化都缺少参照点。

这一节的任务不是证明 attention 不能用，而是说明：  
当序列变长时，中间矩阵和显存流量会成为实际成本。

## 这一节要讲什么

### 1. 标准 attention 的结构

- Q / K / V
- score 计算
- softmax
- 输出加权求和

### 1b. 最小形状例子

- `Q, K, V` 的形状通常写成 `B x S x D`
- `scores = QK^T / sqrt(D)` 的形状是 `B x S x S`
- `softmax(scores)` 仍然是 `B x S x S`
- `out = softmax(scores) V` 的形状回到 `B x S x D`

这个形状链路本身就说明了问题：  
当 `S` 变大时，`S x S` 的中间矩阵会迅速膨胀。

### 1c. 一个最小的读法

- 每一行表示“某个 query 对所有 key 的权重”
- 每一行都要做一次归一化
- 所以 attention 的成本不是一次矩阵乘法，而是一整串中间状态的生成、归一化和回写

### 2. 主要成本

- 中间矩阵大
- 内存流量高
- 长序列时容易成为瓶颈

### 3. baseline 的意义

- 它是后面 FlashAttention 的对照组
- 它帮助我们理解“为什么需要重构数据流”

### 4. 当前写作口径

- 先把标准 attention 作为 baseline 讲完整
- 再把它为什么会膨胀成显存问题讲清楚
- 这一节不需要展开所有变体，只需要把最基本的成本边界立住

## 这里应该放的公式

**公式：Attention 基本计算式**

> 这里要写出 QK^T、softmax、AV 的标准公式，用来定义后续优化目标。
> 这也是后面所有块化、融合和调度策略的比较起点。

> 记号建议统一写成：
>
> `scores = Q K^T / sqrt(D)`
>
> `P = softmax(scores)`
>
> `out = P V`

## 这里应该配的图

**图：标准 attention 数据流图**

> 这里要展示 Q/K/V、score、softmax 和输出之间的中间张量流向。
> 图里最好明确标出 `score` 和 `softmax` 中间结果的写回位置。

## 这一节的结论

Standard attention 不是“错的”，而是“太依赖中间结果写回内存”。
这正是后面 FlashAttention 要解决的核心问题。

## 这一节应该留下的判断标准

- 如果序列长度变长，`scores` 的代价会迅速放大
- 如果 `B x S x S` 中间矩阵已经明显昂贵，就该考虑块化重排
- 如果当前阶段还看不出瓶颈，就先别急着优化，先把 baseline 的数据流看清楚

## 这一节可以进一步追问的问题

- `scores` 和 `probs` 为什么会成为成本中心
- 哪些张量必须写回，哪些张量只是中间状态
- 序列长度变长后，成本为什么会迅速放大

## 常见误区

- 误区 1：attention 的复杂性主要来自公式
- 误区 2：baseline 只用于对比，没有教学价值
- 误区 3：如果结果正确，就说明 attention 的成本不重要

baseline 的作用，是把复杂性来源先准确定位，而不是证明它不够好。

## 讲义提示

如果要把这一节讲得更直观，建议在 notebook 里补一个 4-token 的 toy example：

- 先打印 `Q / K / V`
- 再打印 `scores`
- 然后打印 `probs`
- 最后打印 `out`

这样读者能直接看到：  
attention 的核心不是公式本身，而是 `scores` 和 `probs` 这两个中间张量如何随着序列长度放大。
