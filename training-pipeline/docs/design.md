# PAI DLC 训练设计

## 目标与边界

本地开发，上海公共镜像单机单卡训练，epoch 边界恢复。Python 3.12、PyTorch 2.9.1、torchvision 0.24.1；本地 CPU，云端 CUDA 12.8。镜像地址、挂载选项与操作命令以 [README](../README.md) 为准。

`dlc/job.yaml` 使用官方 CreateJob 字段，通过 PAI CLI 的 `--body-json` 入口提交到上海公共资源组，采用单 Worker/GPU 运行。DLC 根据请求配置存储挂载并启动容器，再执行 bootstrap 入口。

## 数据流

1. 本地 `cloud.prepare` 生成白名单代码目录、SHA256 清单和数据压缩包；由用户放入版本化 OSS 目录。
2. 代码与数据只读挂载，runs 读写挂载。Bash 复制并校验代码目录后直接执行 cloud.launch。
3. 代码在一次性临时目录运行，结束后清理；数据按 SHA256 在 work-dir 缓存，复用前校验归档与解压树，拒绝路径穿越、链接、修改后的缓存和空间不足。
4. 每次启动创建独立 run-id；记录配置、框架/GPU 信息、代码/数据摘要和恢复来源。
5. 每轮取独立 CPU 快照交给有界保存器，后台直接在 runs 挂载创建 checkpoint、指标，最后创建 complete.json。
6. 恢复选最新完整轮次或其最佳 checkpoint，校验路径和摘要后复制到本地加载；继承旧 run 的最佳引用。

## 持久化契约

- 本地 CLI 保存使用原子替换与本地指标追加；挂载保存仅创建独立对象，不依赖覆盖、追加或 rename。
- 两种保存方式共用 checkpoint 字段定义。快照包括模型、Adam、配置、最佳指标与 Python/NumPy/Torch CPU/CUDA RNG。
- 后台队列包含在途写入，默认容量 1、可配置为正整数；反压发生在快照分配之前。CPU 快照同步复制，后台负责序列化和 I/O。
- 挂载必须在 close 时等待上传完成并返回错误。应用 flush/fsync/close 后读回校验，complete.json 最后写；依赖 ossfs 2.0 sync_upload=true 的持久化契约。
- 错误阻止后续队列发布；正常结束等待所有保存。未完成轮次不可恢复，已完成数据损坏直接报错。
- `--epochs` 是目标总轮数；恢复保留优化器学习率并核对数据版本、模型、batch size、workers 和 seed。
- 只承诺同一环境下测试覆盖的连续/恢复一致性；旧 checkpoint 缺少 RNG/配置时警告，跨设备/版本不保证逐位一致。

## 验收

本地验证快照隔离、队列反压、退出等待、错误传播、安全解压、完整轮次恢复和 CNN 训练/评估。本机 GPU 首轮训练、新容器恢复到第 2 轮及日志归档已由用户提供的运行输出验证；DLC 实际 OSS 挂载持久化、权限与新容器可见性仍待验收。验证证据见 [验证记录](validation.md)。

## 职责边界

环境版本与 GPU 运算在环境准备阶段验收；launcher 不进行环境准入或设备选择。启动器在输入和恢复来源确定后一次构造启动 metadata（含框架版本）；训练层原样保存 metadata，解释训练参数、选择设备并单独记录实际 device/gpu；资源模块负责完整性与 staging；持久化层统一 run 路径和有界队列约束；checkpoint 保留恢复兼容性检查。launcher 仅编排这些能力及任务状态。

## 日志归档

入口脚本使用 tee 将输出同时送往控制台和临时本地日志，等待 Python 与 tee 结束后一次复制到 run/train.log。launcher 只通过内部目标文件传递已确定的归档路径；日志复制独立于训练与 checkpoint 保存。返回训练原退出码，归档失败报告到控制台并保留本地日志；容器强杀不保证归档。
