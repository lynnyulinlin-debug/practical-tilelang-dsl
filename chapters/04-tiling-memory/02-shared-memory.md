# 4.2 共享内存：显式控制片上缓存

## 要说明的问题

shared memory 的价值在于 block 内的数据复用，而不是简单“更快的内存”。

## 要点

- `T.alloc_shared` 的作用
- 为什么矩阵乘法特别适合用 shared memory
- 同一块数据如何被多个线程重复使用
- 同步为什么必不可少

## 这里应该放的公式

**公式：共享内存带来的复用收益**

> 这里要说明一个 tile 的数据在 block 内被重复使用多少次，从而为何能提高算术强度。

## 这里应该配的图

**图：shared memory 数据复用示意图**

> 这里要说明 A / B 的 tile 如何被加载一次、计算多次。

## 这里应该解释的现象

- 为什么同一块数据会被 block 内多个线程重复使用
- 为什么 shared memory 能改善复用，但也会增加同步和占用成本
- 为什么不是所有算子都值得搬进 shared memory

## 这一节应该留下的判断标准

- 如果同一块数据会被 block 内多个线程重复用，shared memory 值得考虑
- 如果数据只读一次就走，不一定值得搬进 shared memory
- 如果 shared memory 让 occupancy 掉得太多，收益可能被抵消

这也是为什么“shared memory 越多越好”是错误的直觉。

## 常见误区

- 误区 1：shared memory 只要用了就会更快
- 误区 2：shared memory 可以替代所有 global memory 访问
- 误区 3：只要把数据搬进片上层，性能问题就解决了

shared memory 的作用是放大复用，不是替代判断。
