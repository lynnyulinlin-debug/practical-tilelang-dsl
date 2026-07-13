# 4.3 寄存器与 Fragment

## 要说明的问题

当数据进一步进入计算核心时，shared memory 还不够，最终需要落到寄存器或 fragment 级别。

## 要点

- `T.alloc_fragment` 的角色
- fragment 为什么更接近计算单元
- Tensor Core 相关的块形状为何重要
- 寄存器压力为什么会影响 occupancy

## 这里应该配的图

**图：shared memory 到 register/fragment 的过渡图**

> 这里要说明数据如何从块级缓存进入最终计算单元。

## 这里应该解释的现象

- 为什么过大的 tile 会增加寄存器压力
- 为什么 fragment 的形状要和硬件计算单元匹配

## 这一节应该留下的判断标准

- 如果数据马上要参与计算，fragment / register 比 shared memory 更合适
- 如果数据还要被多个线程复用，先停在 shared memory 可能更划算
- 如果寄存器压力过高，occupancy 和调度灵活性都会受影响

## 常见误区

- 误区 1：fragment 只是更小的 shared memory
- 误区 2：寄存器越多越好
- 误区 3：只要靠近计算单元就一定更快

fragment 和 register 的价值在于“更接近计算”，但它们的容量和持续时间都更受限制。
