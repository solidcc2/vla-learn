# 规格与库存

从子项目的模型、训练配置和资源配置确定产品、地域、GPU 数量、显存、CPU/内存与计费方式。按当前训练需求选型；未来模型尚未确定时，不为假设需求购买大规格。

## 查询依据

- DLC 规格：先看 `aliyun pai-dlc ListEcsSpecs --help`，读取 GPU 类型、卡数、显存、CPU、内存、支付类型与可用标记。
- 规格详情：需要补齐时查 `ecs DescribeInstanceTypes`；GPU 平台代号需用官方规格资料解释，不能凭名称推断。`2xlarge` 不是两张卡。
- 库存：先看 `ecs DescribeAvailableResource --help`，明确地域、可用区、规格、按量/抢占条件。读取具体规格的库存项，不能只看可用区整体可用。
- 分页查询必须覆盖候选所在页；未查询到和不可用分开报告。

DLC 列表可用、ECS 有货、DLC 实际分配成功是不同证据；抢占库存为空也不等于有货。

需要成本比较时读取 [计费与释放](billing-release.md)；环境选择读取 [GPU 兼容性](gpu-compatibility.md)。不要只按单价或显存大小宣布总成本最优。

官方入口：[OpenAPI](https://api.aliyun.com/)、[ECS GPU 规格](https://help.aliyun.com/zh/ecs/user-guide/gpu-accelerated-compute-optimized-and-vgpu-accelerated-instance-families-1)。
