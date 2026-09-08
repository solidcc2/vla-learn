# 计算资源工具入口

`pai` 是独立专用 CLI，不是 `aliyun` 的子命令。先看本机帮助，按需求逐级发现；以下名称是入口提示，不冻结参数或版本。

| 需求 | 优先检查 |
|---|---|
| PAI 工作空间、镜像、资源组和配额 | pai workspace、image、resource |
| PAI 开发实例 | pai dsw |
| DLC 支持规格 | aliyun pai-dlc ListEcsSpecs |
| ECS 规格与库存 | aliyun ecs DescribeInstanceTypes、DescribeAvailableResource |
| ECS 报价 | aliyun ecs DescribePrice |
| 产品与计费模块 | aliyun bssopenapi QueryProductList、DescribePricingModule |
| 产品报价 | aliyun bssopenapi GetPayAsYouGoPrice、GetSubscriptionPrice |
| 创建请求的估价能力 | aliyun list-supported-pricing-apis；确认支持后使用 estimate-cost 模式 |
| 实际计费项用量 | aliyun bssopenapi DescribeInstanceBill |
| 身份、角色和服务授权 | 对应产品帮助及 aliyun ram、sts 等官方接口 |

官方入口：[阿里云 CLI](https://help.aliyun.com/zh/cli/)、[OpenAPI](https://api.aliyun.com/)、[PAI](https://help.aliyun.com/zh/pai/)。
