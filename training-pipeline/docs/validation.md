# PAI DLC 验证记录

按执行时间倒序记录；同一天内按已知执行顺序排列。历史记录未保留具体执行时间的，仅标日期。使用方法见 [README](../README.md)，职责和持久化契约见 [设计说明](design.md)。

## 2026-09-07 — A10 DLC 首轮冒烟成功

- 配置提交：`52cb36cea9d5b9d2acb6f00a6c2df7ff07c82cc1`（`chore(training): 配置 A10 单卡完整训练任务`），仅包含 `dlc/job.yaml`；README、验证记录和 devlog 未包含在该提交中。提交前 YAML dry-run 和 `git diff --check` 通过；完整 20 轮训练尚未执行。

- 任务：`cifar10-a10-smoke`，JobId `dlc1iydukqa2wkd3`；上海工作空间 `vla_learn`（`1506433`）。
- 制品：代码和数据均为 `release-20260907-100259`；提交前通过 OSS API 确认远端对象已发布，本地制品与工作区源码一致且校验通过。制品 manifest 记录 `git_revision=38dcbdf48dd121e2e92be7b86e43c3994feae611`、`git_dirty=true`，代码摘要为 `d5cb3b4da23817bf1271e6db69853a29f74045980fe78dd4323d095c8d2bf9cb`。制品早于上述配置提交生成，配置 commit 不代表训练制品的源码版本。
- 环境：`ecs.gn7i-c8g1.2xlarge`，单 Worker、按量付费，驱动设置 `550.127.08`；使用 README 中固定的 PyTorch 2.9.1 / Python 3.12 / CUDA 12.8 公共镜像。结果 config.json 确认 `device=cuda`、`gpu=NVIDIA A10`。
- 参数：`configs/smoke.json`，1 轮、batch size 128、2 个数据加载 worker；通过提交参数覆盖完整训练 YAML，最长运行 15 分钟，结束保留 0 分钟。
- 结果：DLC 状态 `Succeeded`；训练样本 50,000，训练 loss 1.4796286177825928、准确率 45.55%；测试样本 10,000，测试 loss 1.1513895341873168、准确率 57.64%。
- 耗时：从提交到结束总计 318 秒；容器运行约 17 秒，包含初始化、训练、评估和归档，并非纯训练耗时。系统事件显示主要等待在资源准备与镜像拉取。Pod 的 duration=228 秒包含创建后的等待，不作为容器运行时长。
- run ID：`cifar10-20260907T042008Z-d71ddf6699dd`。
- 持久化：OSS API 确认该 run 的 config.json、started.json、epochs/0001/checkpoint.pt、metrics.json、complete.json、succeeded.json、train.log 均存在；通过本地只读挂载读取成功标记及首轮 complete.json。checkpoint 大小 7,487,234 字节；complete.json 记录的 SHA256 为 `8fd2fbb25b7f98c6b786b2de87e1a395d78dd2c7170daa7d8454fb3c7a47e880`，本次未独立下载重算。
- 边界：已验证此 A10/镜像/驱动配置完成首轮 GPU 训练和 OSS 归档；未独立检查 DLC 实际挂载版本及 sync_upload 参数，未验证故障下的持久化语义、新 DLC 任务恢复或 20 轮训练；容器中未另行读取驱动版本。
- 后续：完整训练 YAML 使用 `configs/train.json`（目标总轮数 20）、90 分钟超时。续训时在 UserCommand 追加 `--resume-run cifar10-20260907T042008Z-d71ddf6699dd`，从第 2 轮跑到第 20 轮；当前尚未追加该参数或提交续训。
- 费用：尚未核对实际账单；Billing Usage 与 estimate-cost 返回的时间单位存在未解决差异，不将此前小时换算结果记为已确认报价。

## 2026-09-07 — 实际资源配置与 YAML 预览

- 对象：`dlc/job.yaml`，目标制品版本 `release-20260907-100259`。
- 环境：本机 PAI CLI；上海工作空间 `vla_learn`（`1506433`），Bucket `vla-learn2`。
- 操作：查询工作空间和 DLC GPU 规格，填写实际资源后执行 `pai dlc job submit --region cn-shanghai --body-json @dlc/job.yaml --dry-run -o json`。
- 结果：预览通过；请求中的工作空间为字符串，已无占位符。规格为单卡 `ecs.gn8is.2xlarge`，驱动为其支持列表中的 `580`。查询时 OSS 的 code/data 目录为空，本地完整制品包含代码、数据压缩包及 manifest。
- 边界：没有上传制品或提交训练；规格查询与预览不证明实时库存、挂载权限、驱动及训练运行成功。未查询价格。

## 2026-09-07 — 官方 PAI YAML 入口迁移

- 对象：由提交 Shell 脚本内嵌 JSON 转换得到的 `dlc/job.yaml`，当时仍含占位符。
- 环境：本机已安装的 PAI CLI。
- 操作：使用 `--body-json` 和 `--dry-run` 解析 YAML，将预览输出中的请求体与原 JSON 比较。
- 结果：请求字段、类型、挂载配置和折叠后的启动命令完全一致；文档引用与 `git diff --check` 检查通过。
- 边界：仅验证格式迁移和本地请求生成，未调用任务创建接口。

## 2026-09-07 — 提交 Shell 精简后的本地验证

- 对象：当时工作区的提交 Shell 入口及测试集；删除 Python 包装和其专属测试。
- 环境：子项目 Python 3.12 CPU venv，本地临时目录。
- 操作：替身 CLI 接收 Shell 传入的参数，解析内嵌 JSON；在子项目目录执行完整测试、Bash 语法和差异格式检查。
- 结果：请求体完整传递，输入只读、runs 读写，镜像与环境变量一致；44 项测试通过，脚本与差异检查通过。
- 边界：使用替身 CLI，未提交任务；该 Shell 入口随后由 YAML 替代。

## 2026-09-07 — 提交包装入口的本地验证

- 对象：当时工作区的提交包装入口与测试集。
- 环境：子项目 Python 3.12 CPU venv，替身 CLI。
- 操作：验证默认预览、提交请求一致性、参数转义、CLI 失败退出码传播；执行完整测试及依赖检查。
- 结果：45 项测试通过，`pip check` 通过。从仓库根目录首次运行完整测试时出现模块导入错误，改在子项目目录运行后通过。
- 边界：未调用真实 CreateJob；此结果对应当时的包装实现，该入口和专属测试随后删除。

## 2026-09-07 — 独立 GPU 主机首轮与恢复验证

- 对象：用户通过 prepare 制品启动的首次 smoke 及新容器恢复 smoke；所用制品版本未在提供的输出中明确记录。
- 环境：独立 GPU 主机，上海公共 PyTorch 2.9.1 / CUDA 12.8 镜像的公网地址；work/runs 使用主机本地目录挂载。
- 操作：首次完成 epoch 1，再以旧 run 为恢复来源，在新容器中完成 epoch 2。
- 结果：以下数据来自用户提供的容器输出。

| 项目 | 首次 smoke | 新容器恢复 smoke |
|---|---|---|
| run ID | cifar10-20260907T022230Z-d37f1c65955d | cifar10-20260907T022437Z-0a25dca34b35 |
| 完成轮次 | epoch 1 | 从 epoch 2 恢复并完成 |
| 训练样本数 | 50,000 | 50,000 |
| 测试样本数 | 10,000 | 10,000 |
| 训练准确率 | 45.992% | 59.724% |
| 测试准确率 | 56.87% | 68.61% |
| 测试 loss | 1.163036 | 0.889442 |
| 日志归档 | 输出 Log archived | 输出 Log archived |

恢复日志记录 optimizer learning_rate=0.001。未单独读取该主机的退出码、完成标记或显卡型号，也未进行 GPU 连续/恢复逐位一致性比较。该结果不验证 DLC 的 OSS 挂载。

## 2026-09-07 — 早期本地测试与发布入口验证

- 对象：当时工作区的训练、恢复、持久化与发布实现，未记录对应 Git 提交或制品版本。
- 环境：本地 CPU venv，临时目录模拟存储。
- 操作：执行自动化测试、`pip check`、Bash 语法与差异检查；校验实际发布目录并加载 Python 入口。
- 结果：44 项测试通过，依赖、脚本、差异检查及发布入口验证通过。覆盖离线训练与评估、连续/恢复一致性、快照独立性、异步反压和失败传播、资源完整性及退出日志归档。
- 边界：早期检查的具体时间与彼此先后未保留，合并记录；本地文件系统结果不替代实际 OSS 持久化验收。

## 待完成的 DLC 验收

- 独立核对实际 runs 挂载版本和 sync_upload=true 的关闭上传语义。
- 在新的 DLC 任务中恢复首轮 checkpoint，验证跨任务可见性并训练到目标总轮数 20。
- 下载 checkpoint 做独立校验及评估，核对实际账单。

已完成制品发布、A10 首轮训练和 OSS 归档；尚未启动 DLC 续训任务。
