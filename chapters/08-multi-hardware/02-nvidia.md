# 8.2 NVIDIA 后端

## 要解决什么问题

先把最成熟、最常见的后端讲清楚，后面再对比其他平台。

这一节不是在讲 CUDA 入门，而是在讲 TileLang 代码到 NVIDIA 平台时，哪些层次最影响最终性能。

## 要讲什么

### 1. PTX / SASS

- PTX 作为中间表示
- SASS 作为最终机器指令
- 编译链路对性能的影响

### 2. Tensor Core

- 矩阵计算单元
- 何时能发挥优势
- 数据布局和 tile 形状的重要性

### 3. 性能关注点

- shared memory 容量
- register 使用量
- block 大小
- memory coalescing
- bank conflict

### 3b. 典型失误

- 只看 Tensor Core，不看 tile 是否对齐
- 把 block 设大后忽略 register spill
- 把 shared memory 拉满后忽略占用率下降
- 只看峰值 TFLOPS，不看 latency 和稳定性

### 4. 迁移时先看什么

1. 先确认能否正确编译到 NVIDIA target
2. 再确认基本 shape 和 dtype 是否跑通
3. 然后检查 block / tile 是否和 Tensor Core 形状对齐
4. 最后再调 shared memory、寄存器和并发度

### 4b. 收敛顺序

1. 先保证输出正确
2. 再保证 tile 和硬件单元匹配
3. 再看访存是否合并
4. 再看 shared memory 是否足够但不过量
5. 最后看寄存器压力和并发是否平衡

### 4. 当前写作口径

- 这一节优先讲落地路径和性能关注点
- 不是要重讲 CUDA 语法，而是要解释为什么 TileLang 的参数在 NVIDIA 上要这样选
- 如果后面补 benchmark，要和第 8.5 节的统一口径一致

### 5. 不要默认照搬的经验

- 不要默认把别的平台上的 tile 形状直接拿过来
- 不要默认 block 越大越好
- 不要默认 shared memory 越多越好
- 不要默认“Tensor Core 一定优于普通路径”，要看 shape 和对齐条件

### 6. 这一节应该留下什么

- 一个 NVIDIA 上的参数检查顺序
- 一个判断 Tensor Core 是否值得用的思路
- 一个看到性能异常时的第一轮排查方向

## 这里应该配的图

**图：NVIDIA 上的 kernel 落地路径**

> 这里要说明 TileLang 生成到 NVIDIA 时，哪些层次最影响性能。

这张图最好放在“迁移时先看什么”之后、进入小结之前，用来把编译链路、资源层和形状层连成一条线。

## 这里应该放的表

**表：NVIDIA 参数检查清单**

> 这里要记录 block、tile、shared memory、register、Tensor Core 对齐条件。

这张表适合放在性能关注点和收敛顺序之后，作为 NVIDIA 平台的调参速查。

## 小结

NVIDIA 后端的重点不是“CUDA 基础”，而是“如何让 TileLang 输出更接近硬件极限的实现”。
换句话说，这一节关注的是映射和约束，不是基础语法。

## 讲义提示

如果想把这一节讲深一点，建议把“为什么快”拆成三层：

- 指令层：PTX / SASS
- 资源层：shared memory / register / occupancy
- 形状层：tile / 对齐 / coalescing

## 这一节可以进一步追问的问题

- 这一步的性能瓶颈更像指令、资源还是形状问题
- Tensor Core 是否真的适合当前形状
- block 和 shared memory 的配置是否已经逼近资源上限

## 常见误区

- 误区 1：NVIDIA 后端就是 CUDA 教程
- 误区 2：只要启用了 Tensor Core 就结束了
- 误区 3：越接近峰值 TFLOPS 就一定越适合当前场景

这节要教的是 NVIDIA 上的参数判断顺序，而不是 CUDA 语法本身。

读者一旦能把这三层区分开，就能看懂很多性能问题为什么不是单一因素造成的。
