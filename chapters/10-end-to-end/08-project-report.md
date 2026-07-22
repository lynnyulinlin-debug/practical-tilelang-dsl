# 10.8 项目报告：Toy Decoder 子图迁移

## 报告目的

这份报告对应第 10 章的最小可执行项目，用来把正文里的迁移流程和脚本里的实际结果连接起来。

它不是最终生产案例，而是一个可以复用的项目记录模板。

它负责把迁移过程、benchmark 口径和报告结论组织成一套可重复的工程闭环。  
脚本负责跑出结果，报告负责把结果固定成证据页，再回填到正文和 benchmark 里。

- 对应 notebook：`examples/10_end_to_end_report.ipynb`

## 项目层次和对象

- 场景：toy decoder subgraph migration
- 目标：验证“选子图 -> 排优先级 -> 先跑闭环 -> 逐步优化 -> 多后端记录 -> benchmark 收口”的流程
- 形式：CPU / NumPy 可运行的最小项目
- 脚本：`examples/10_end_to_end_migration.py`、`examples/10_end_to_end_operator.py`

这份报告对应的是第 10 章的算子层项目，而不是公开案例层项目。

- 总述层项目：`某大模型相关子图`
  - 对应第 10 章的主线叙事
- 算子层项目：`某推理场景中的部分算子`
  - 对应当前 toy decoder 子图迁移报告
- 公开案例层项目：`DeepSeek 相关子图（在可公开范围内）`
  - 只有在公开可验证时才启用

这份报告的作用在于：

1. 证明算子层项目可以跑通
2. 给总述层项目提供可复用的证据模板
3. 为未来公开案例层项目预留命名和报告格式

## 报告字段说明

这份报告会显式记录第 10 章项目的三个关键字段：

- `project_layer`
  - 指明当前是总述层、算子层还是公开案例层
- `case_style`
  - 指明案例表述层采用哪种写法
- `subgraph_label`
  - 指明具体迁移的子图名称
- `sweep_enabled`
  - 指明是否做了小规模序列长度扫描

这样做的目的，是让报告同时记录“结果是什么”和“结果属于哪一层项目、哪一个子图”。

## 迁移对象和路线

本项目选择的子图是一个 toy decoder subgraph，包含：

- RMSNorm
- QKV projection
- Attention
- Output projection
- Residual

其中最先迁移的热点是 Attention 路径，因为它最容易体现：

- 中间张量写回的成本
- baseline 和块化实现的差异
- 子图级 benchmark 的收益

1. 先建立 baseline
2. 再实现 blockwise attention 迁移版
3. 固定相同输入和统计口径
4. 对比延迟、显存和误差
5. 记录多后端的参数建议
6. 生成统一 benchmark 记录

## benchmark 结果

- `batch=1`
- `seq_len=8`
- `dim=16`
- `block_size=4`
- `runs=2`

以下结果来自当前脚本实际运行：

- baseline: `0.1221 ms/run`
- migrated: `0.1236 ms/run`
- max error: `2.38e-07`
- speedup: `0.99x`

说明：

- 在这个极小配置下，块化版本主要承担的是结构验证作用
- 真实收益更依赖更长序列、更稳定的热点分布和更大的子图规模
- 因此这份项目报告的重点是“流程成立”，不是“峰值收益”

本项目在脚本里额外支持了小规模 `seq_len` 扫描，用来观察块化实现的趋势是否稳定。

当前公开的扫描结果是：

| seq_len | baseline ms/run | migrated ms/run | speedup |
| --- | ---: | ---: | ---: |
| 8 | 0.2181 | 0.1418 | 1.54x |
| 16 | 0.0662 | 0.3030 | 0.22x |

这组结果说明两件事：

1. 单点 benchmark 不足以说明结构性收益
2. 真实项目需要看规模变化下的趋势，而不是只看一个 shape

对第 10 章来说，这个结论很重要，因为它告诉人：

- benchmark 记录不只是保存一个数值
- 更应该保存“在什么规模下有效、在什么规模下反转”
- 项目层面的判断必须结合趋势，而不是只看最优点

脚本中给出的估算显示：

- baseline 和 migrated 都处于很小的内存规模
- 在小规模 toy 设置下，显存占用差异不显著

这符合本项目的定位：

- 先打通迁移路径
- 再放大到更真实的子图和更完整的 benchmark

## 记录和回填

脚本目前给出了三类目标平台的记录建议：

- `cuda`
- `rocm`
- `ascend`

对应的记录关注点分别是：

- CUDA: shared-memory reuse 和 Tensor Core 友好形状
- ROCm: local memory reuse 和 vectorization-aware layout
- Ascend: on-chip buffer locality 和 matrix-path alignment

这些建议属于迁移计划，不是最终结论。

`blockwise` 路径在不同规模下不一定单调占优，因此 benchmark 需要保留趋势扫描而不是只保留单点结果。  
后续如果补图，优先补一张趋势折线图，把“单点最优”和“规模反转”直接画出来。

本项目复用了统一 benchmark 模板，记录字段包括：

- scenario
- operator
- platform
- target
- dtype
- shape
- baseline
- optimization
- latency_ms
- tflops
- peak_mem_mb

这份报告证明这个模板足够承载第 10 章的项目记录。

## 结论与后续

这个 toy decoder 子图项目的作用，是把第 10 章的流程变成一个可运行的最小闭环：

1. 选择子图
2. 排优先级
3. 先闭合 baseline 与迁移版
4. 固定 benchmark 口径
5. 记录多后端迁移建议
6. 给正文回填证据

这份报告不是“项目做完后的附录”，而是 TileLang 工程闭环的一部分：正文讲方法，脚本跑结果，报告固化证据。

- 放大 `seq_len`，观察 blockwise 路径是否更能体现 IO 优化
- 把更多算子纳入同一子图，形成更真实的项目结构
- 将报告结果回填到第 10 章正文中的“经验总结”和“benchmark”部分

## 与正文和脚本的关系

- 正文：讲迁移方法
- 脚本：跑项目流程
- 本报告：存结果和结论

三者分离，才能让项目既可读又可复用。
