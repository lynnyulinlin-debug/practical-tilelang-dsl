# 9.2 Triton 编程模型概览

## 要解决什么问题

这一节只讲 Triton 怎么写，不先讨论取舍。

教学上，这一节要让读者建立一个足够稳定的心智模型：  
一个 kernel 先按 tile / block 来看，然后围绕这个 tile 组织 load、compute、store。

## 要讲什么

### 1. 核函数入口

- `@triton.jit`
- program id 的分区方式
- 一个 kernel 对应一个 tile 或 block 的执行单元

### 2. 读写接口

- `tl.load`
- `tl.store`
- `tl.dot`
- mask 的基本用法

### 2b. 一个最小的工作流

1. 先用 program id 切出当前 tile
2. 用 `tl.load` 把这一块数据取进来
3. 用 `tl.dot` 或其他算子做局部计算
4. 用 `tl.store` 写回结果
5. 用 mask 处理边界 tile

这个顺序很重要，因为它说明 Triton 主要是围绕 tile 组织 kernel，而不是围绕完整张量逐层展开。

### 3. 编程直觉

- 先定义 block 级 tile
- 再围绕 tile 做数据访问
- 让编译器帮助处理部分低层细节
- 常见写法更像“局部 kernel 组合”，而不是“完整调度显式编排”

### 4. 当前写作口径

- 先讲程序入口和 tile 视角
- 再讲 load / store / dot / mask 的角色
- 不要把这一节写成 API 清单

## 这里应该配的图

**图：Triton kernel 的 tile 视角图**

> 这里要说明 Triton 的编程模型如何从 block / tile 视角组织计算。

这张图最好放在最小工作流讲完之后、进入小结之前，作为本节的核心心智模型图。

## 这里应该放的表

**表：Triton 核函数四步表**

> 这里要记录 `tile 切分 -> load -> compute -> store` 每一步的职责。

这张表适合放在图的后面，用来把流程图压成一眼能复述的步骤清单。

## 小结

理解 Triton 的关键，不是记住 API 列表，而是形成“一个 tile 就是一段局部并行计算”的心智模型。
只要这个心智模型成立，后面的自动优化和选型讨论就容易跟上。

## 这一节可以进一步追问的问题

- tile 视角为什么是 Triton 的基础
- `load / compute / store / mask` 为什么是最小工作流
- 为什么这个模型更适合局部热点 kernel

## 常见误区

- 误区 1：Triton 的编程模型只是 API 列表
- 误区 2：写 kernel 只需要学会 load 和 store
- 误区 3：mask 只是边界处理，不影响模型理解

这一节的价值，是让读者形成“局部 tile 计算”的心智模型。

## 讲义提示

如果要把这一节讲得更像教程，建议补一个 4 步图：

- tile 切分
- load
- compute
- store

读者只要抓住这四步，就能理解为什么 Triton 在常见 GPU 热点上很顺手。
