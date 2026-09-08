# 计费、用量与资源释放

## 报价与估算

先确定产品、地域、规格、节点数、支付方式、币种、用量及其单位。查询 DLC 时使用 DLC 的产品和计费模块，不能代入同规格 ECS 价格。

- 看本机 `list-supported-pricing-apis --help` 和目标产品支持列表，再决定是否使用 `--estimate-cost`。确认该模式跳过实际创建，不能用真实创建测试询价。
- 产品报价先用 Billing 的产品列表定位产品及类型，处理分页，再读取计费模块；不要凭产品名猜 ProductCode 或 Config 字段。
- `GetPayAsYouGoPrice` 的金额、Usage 数量及单位必须一起解释；`UnitPrice=0` 不代表免费。
- 若不同接口或文档对时间单位解释冲突，保存请求、响应和冲突字段，暂停确定性换算。不能凭“这个价格看起来合理”选一个单位；价格排名也需先确认同口径。
- 估价接口不预测模型吞吐。没有实测时只给明确假设的预算情景；有实测时区分恢复/初始化、逐轮计算、归档、调度和镜像准备，说明外推范围。

报价输出：产品、规格、条件、币种、数量/单位、原价/折后价、日期、证据位置；明确是否包含存储、网络等附加费用。

## 实际计费用量

任务生命周期时长、容器运行时长、训练循环耗时、账单计费用量分别记录；不能互相替代。接口字段名叫 duration 也需先查语义，避免把 Pod 创建后的等待算成计算耗时。

优先核对 `bssopenapi DescribeInstanceBill --help` 等当前账单接口，按产品、账期及计费项查询。读取 Usage/UsageUnit、ServicePeriod/ServicePeriodUnit、金额和单位；字段以当前接口为准。

先确认账单的 InstanceID/标签与任务的映射，不假设一定等于 JobId，也不把该产品全部账单归给一个任务。空结果可能是延迟、过滤或权限问题，不表示免费。按官方出账规则说明何时可复查。

## 释放与保留

结合任务实际资源来源、终态、结束保留设置和产品规则判断是否仍计费：

- 公共按量资源结束释放与任务历史记录保留分开解释；无需为停止计算计费而删除已结束记录。
- 预付费资源池、开发主机、OSS 等独立资源不随某个 DLC 任务结束而自动停止自身计费。
- 配置规定应释放，不等于已经读回底层释放结果。说明当前证据是规则、任务状态还是实际资源状态。

官方入口：[DLC 计费](https://help.aliyun.com/zh/pai/product-overview/billing-of-dlc)、[账单明细](https://help.aliyun.com/zh/user-center/developer-reference/api-bssopenapi-2017-12-14-describeinstancebill)、[Billing OpenAPI](https://api.aliyun.com/product/BssOpenApi)。
