# 实例选型与价格核验

## 先确定查询条件

从用户要求和项目配置中取得产品（ECS 或 PAI DLC 等）、地域、GPU 需求、CPU/内存需求、镜像/CUDA、计费方式和预计时长。只询问会影响选择的缺失项；候选方案可以带清楚的假设，不直接变成付费操作。

## 查询链

1. **产品支持的规格**：DLC 优先检查 `aliyun pai-dlc ListEcsSpecs --help`，再按地域和 GPU 条件查询。读取卡数、GPU 类型、显存、CPU、内存、计费方式、可用标记及支持的驱动。
2. **规格详情**：需要补齐时使用 `aliyun ecs DescribeInstanceTypes`。GPU 类型若是平台代号，查官方规格表确认对应型号。`2xlarge` 是规格档位，不代表 GPU 数量。
3. **库存**：检查 `aliyun ecs DescribeAvailableResource --help`，按可用区、计费方式和规格查询；ECS 库存仅是 ECS 的证据，DLC 的分配还受其资源池和调度限制。区分按量与抢占库存。
4. **报价**：ECS 使用 `DescribePrice`；跨产品先检查 `bssopenapi GetPayAsYouGoPrice` / `GetSubscriptionPrice` 的产品与模块要求。查询 DLC 就取得 DLC 计费依据，不能把底层同型号 ECS 的价格直接当作 DLC 报价。
5. **兼容性**：将所选镜像、框架、CUDA、GPU 架构与驱动一起核对。接口列出的支持驱动不等于任务默认驱动；必要时在配置中显式指定经过核对的驱动。

每步先看当前命令帮助再构造参数。规格/库存数据应按任务刷新；静态知识只用于解释。

## 价格输出

以少量候选对比即可：规格、GPU 型号/数量、显存、CPU/内存、单位价格、计费方式、查询时间、库存依据。

- 分清原价、优惠价、币种、时间单位、地区站点和账号优惠条件。
- 时间估算明确假设，例如“运行 30 分钟”；考虑最低计费单位及实例保留规则。
- 存储、网络、日志和其他附加费用是否包含，要依据报价范围说明。
- 查询不到当前报价时保留未确认状态，给出确切缺口。官网公开价可作标明日期与条件的参考值。

选择应匹配任务需求。不要仅因某规格支持较新驱动就称为最经济，也不要未核对兼容性便推荐最便宜的旧卡。

## 详细接口与规则

- [OpenAPI 开发者门户](https://api.aliyun.com/)：在 ECS 产品下检索规格、库存和报价操作；在 PAI DLC 产品下检索训练规格，按实际适用版本读取参数。
- [Billing OpenAPI](https://api.aliyun.com/product/BssOpenApi)：产品询价与账单接口，确认产品、模块和计费条件。
- [ECS GPU 规格官方说明](https://help.aliyun.com/zh/ecs/user-guide/gpu-accelerated-compute-optimized-and-vgpu-accelerated-instance-families-1)：解释 GPU 型号与规格，不能替代实时库存和报价。
- [PAI 官方文档](https://help.aliyun.com/zh/pai/)：进入目标产品的计费与环境说明，核对费用组成和兼容性要求。

参数定义、枚举、默认值和限制按上述外链现查，不在本技能维护副本。
