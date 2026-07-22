# 9.2 Triton 编程模型概览

## Triton 的 tile 工作流

这一节只讲 Triton 怎么写，不先讨论取舍。

这一节要建立一个足够稳定的心智模型：  
一个 kernel 先按 tile / block 来看，然后围绕这个 tile 组织 load、compute、store。

如果先学会这个模型，后面的自动优化和选型就有了共同参照。  
因为 Triton 的很多工程特征，最终都会落回这个 tile 视角上。

### 1. 核函数入口

- `@triton.jit`
- program id 的分区方式
- 一个 kernel 对应一个 tile 或 block 的执行单元

### 2. 读写接口

- `tl.load`
- `tl.store`
- `tl.dot`
- mask 的基本用法

### 3. 一个最小的工作流

1. 先用 program id 切出当前 tile
2. 用 `tl.load` 把这一块数据取进来
3. 用 `tl.dot` 或其他算子做局部计算
4. 用 `tl.store` 写回结果
5. 用 mask 处理边界 tile

这个顺序很重要，因为它说明 Triton 主要是围绕 tile 组织 kernel，而不是围绕完整张量逐层展开。

### 4. Triton 的 tile 直觉

- 先定义 block 级 tile
- 再围绕 tile 做数据访问
- 让编译器帮助处理部分低层细节
- 常见写法更像“局部 kernel 组合”，而不是“完整调度显式编排”

这里最重要的不是语法，而是心智模型：Triton 先把问题压缩成局部 tile，再在这个局部里完成数据搬运和计算。  
这让它特别适合快速写出热点 kernel，但也意味着它把很多结构化控制交给了编译器。

### 5. Triton 的局部 tile 心智模型

- 先讲程序入口和 tile 视角
- 再讲 load / store / dot / mask 的角色
- 不要把这一节写成 API 清单

## Triton 的 tile 视角图

**图：Triton kernel 的 tile 视角图**

> 这里要说明 Triton 的编程模型如何从 block / tile 视角组织计算。

这张图最好放在最小工作流讲完之后、进入小结之前，作为本节的核心心智模型图。

## Triton 核函数四步表

**表：Triton 核函数四步表**

> 这里要记录 `tile 切分 -> load -> compute -> store` 每一步的职责。

这张表适合放在图的后面，用来把流程图压成一眼能复述的步骤清单。

## Triton 的 tile 视角判断

理解 Triton 的关键，不是记住 API 列表，而是形成“一个 tile 就是一段局部并行计算”的心智模型。
只要这个心智模型成立，后面的自动优化和选型讨论就容易跟上。

Triton 的编程模型是在说明：先把问题局部化，再让编译器帮忙处理局部内部的实现细节。  
这也是它和 TileLang 的第一层分叉点。

## 常见误区

- 误区 1：Triton 的编程模型只是 API 列表
- 误区 2：写 kernel 只需要学会 load 和 store
- 误区 3：mask 只是边界处理，不影响模型理解

这一节的价值，是形成“局部 tile 计算”的心智模型。
