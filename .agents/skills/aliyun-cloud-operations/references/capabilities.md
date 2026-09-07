# 能力域导航

以下是入口索引，使用前以本机帮助为准。`pai` 是独立专用 CLI，不是 `aliyun` 的子命令。

```text
阿里云操作
├── PAI 日常工作 → pai
│   ├── 工作空间／数据集／镜像 → workspace、dataset、image
│   ├── 开发实例 → dsw
│   ├── 训练预览／提交／状态／日志 → dlc job
│   └── 资源组／配额 → resource
│
└── 跨产品查询及底层 API → aliyun
    ├── 身份／角色／令牌 → configure、ram、sts、cloudsso
    ├── 资源与费用
    │   ├── 地域 → ecs DescribeRegions
    │   ├── 规格 → ecs DescribeInstanceTypes
    │   ├── 库存 → ecs DescribeAvailableResource
    │   ├── ECS 报价 → ecs DescribePrice
    │   ├── 产品报价 → bssopenapi GetPayAsYouGoPrice／GetSubscriptionPrice
    │   ├── 账单／余额 → bssopenapi QueryBill／QueryAccountBalance
    │   └── 配额／资源检索 → quotas、resourcecenter
    ├── PAI 补充 API → pai-dlc、pai-dsw、aiworkspace、eas
    ├── 计算／容器／镜像 → ecs、ess、cs、eci、cr、fc、sae
    ├── 存储 → ossutil、nas、ebs、hbr
    ├── 数据库 → rds、polardb、r-kvstore、dds、tablestore
    ├── 网络 → vpc、slb、alb、privatelink、alidns、cdn
    ├── 日志／监控／审计 → sls、cms、arms、xtrace、actiontrail
    ├── 安全 → ram、kms、sas、waf-openapi
    ├── 编排／运维 → ros、oos
    └── 数据处理／消息 → dataworks-public、emr、alikafka、rocketmq
```

## 发现命令

```bash
pai --help
pai dlc job submit --help
aliyun --help
aliyun ecs --help
aliyun ecs DescribePrice --help
```

按域逐级展开即可，不需要遍历全部产品。分页、过滤、等待、预览与估价能力查当前帮助，参数拼写和支持范围不在技能中固定。

## 官方入口

- [阿里云 CLI 文档](https://help.aliyun.com/zh/cli/)：命令结构、认证、调用控制、插件及版本相关说明。
- [OpenAPI 开发者门户](https://api.aliyun.com/)：按产品、操作和适用 API 版本查请求与响应。
- [PAI DLC CLI 文档](https://help.aliyun.com/zh/pai/developer-reference/dlc-distributed-training)：专用训练入口及配置文件支持。
