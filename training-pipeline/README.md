# CIFAR-10 本地开发与 PAI DLC 训练

工作流：本地开发/打包 → 将资源放入版本化 OSS 目录 → 公共镜像启动 → 输入复制到本地 → 训练结果异步写入 runs 挂载 → 新任务恢复。

OSS 访问由 DLC 存储挂载负责。通过 PAI CLI 读取 `dlc/job.yaml` 创建任务。

设计与职责详见 [设计说明](docs/design.md)。

源码按职责组织：根目录的 `model.py`、`data.py`、`engine.py`、`checkpoint.py` 负责模型、数据、训练循环与训练状态；`train.py` / `evaluate.py` 是入口。`persistence/files.py` 提供基础文件操作，`persistence/runs.py` 负责后台保存和 run 恢复；`cloud/` 只负责云端启动、源码发布和数据准备。数据目录 `data/` 与源码 `data.py` 分别存放数据文件和加载逻辑。

职责边界：环境准备负责依赖与 GPU 验收；`cloud.launch` 负责衔接资源、恢复来源、输出目录及任务状态；`train` 负责训练参数与设备选择；`checkpoint` 负责训练状态兼容性；`persistence` 负责 run 路径、队列和文件完整性。启动器只保留自身参数语义约束，例如 `resume-best` 必须配合 `resume-run`，云端恢复不能混用训练 JSON 的本地 `resume`。

## 1. 环境与本地训练

固定 Python 3.12、torch 2.9.1、torchvision 0.24.1。本地 venv 使用 CPU wheel，`requirements-cpu.txt` 保存完整依赖版本。每次使用 Python 前激活子项目环境：

```bash
source training-pipeline/venv/bin/activate
cd training-pipeline
# 需要复现依赖时：
python -m pip install -r requirements-cpu.txt
python -m pip check
```

首次创建环境时，在仓库根目录执行 `python3.12 -m venv training-pipeline/venv`，随后激活。

本地训练使用同步文件保存：

```bash
python train.py --epochs 1 --data-dir data --output-dir outputs/first --device cpu
python train.py --epochs 2 --data-dir data --output-dir outputs/resumed \
  --device cpu --no-download --resume outputs/first/last.pt
python evaluate.py outputs/resumed/last.pt --data-dir data --device cpu --no-download
```

`last.pt`、刷新后的 `best.pt` 在本地盘原子替换；`metrics.jsonl` 在本地追加。此模式的输出目录不要设为 OSS 挂载。云端通过 `cloud.launch` 使用下面的异步挂载协议。

`--epochs` 是目标总轮数，恢复从已完成轮次 + 1 开始。checkpoint 包含模型、Adam、最佳准确率、Python/NumPy/Torch CPU/CUDA RNG；恢复保留优化器学习率，检查模型、数据版本、batch size、worker 数和 seed。旧 checkpoint 缺失 RNG/配置会警告；跨设备或框架版本不保证逐位一致。只支持 epoch 边界恢复。

## 2. 准备与放置资源

在本地准备官方 CIFAR-10：

```bash
python -c 'from torchvision.datasets import CIFAR10; CIFAR10(root="data", train=True, download=True)'
python -m cloud.prepare --data-dir data --output-dir dist/release-003
# 数据不变时仅生成新代码版本：
python -m cloud.prepare --code-only --output-dir dist/code-release-004
```

每次 prepare 使用新的输出目录。生成的制品位于本地，下一步将其复制到目标存储。

以下 OSS 相对目录均位于 `oss://<bucket>/training-pipeline/` 下。生成文件的放置关系：

| 目标 OSS 目录 | 本地生成的文件 |
|---|---|
| `code/<code-version>/` | 本地 `dist/<版本>/code/` 下全部文件，包括普通 Python 源码、`bootstrap.sh`、`code.manifest.json`、`SHA256SUMS` |
| `datasets/cifar10/<data-version>/` | `data.tar.gz`、`data.manifest.json` |
| `runs/` | 无需预先放文件，由训练创建独立 run 目录 |

通过 OSS 控制台上传上述文件，或在已有可写 OSS 挂载的环境里直接复制。代码先复制其他文件，最后复制 `SHA256SUMS`；数据先复制压缩包，最后复制数据 manifest。全部完成后再提交训练。每次代码或数据变更使用新版本目录，不覆盖正在使用的资源。不要混入其他版本的文件或校验清单。

代码制品按白名单复制运行所需源码、训练配置和 Bash 入口。清单记录文件 SHA256 和 Git revision/dirty 状态；数据制品打包前验证官方 CIFAR 文件 MD5。发布目录应仅允许可信写入，校验清单用于检测文件损坏或复制不完整。

发布代码到已挂载的 ECS 目录（在子项目目录执行，使用新的版本名）：

```bash
set -e
release=release-005
python -m cloud.prepare --code-only --output-dir "dist/$release"
mkdir "/mnt/oss/training-pipeline/code/$release"
for entry in "dist/$release/code/"*; do
  [ "${entry##*/}" = SHA256SUMS ] && continue
  cp -R "$entry" "/mnt/oss/training-pipeline/code/$release/"
done
cp "dist/$release/code/SHA256SUMS" "/mnt/oss/training-pipeline/code/$release/"
```

复制任何文件失败时停止发布，不能继续复制最后的 SHA256SUMS。DLC 应挂载该具体版本目录。

## 3. 公共镜像和挂载配置

上海官方公共镜像：

```text
dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai/pytorch:2.9.1-gpu-py312-cu128-ubuntu24.04-590381cf-1764375854
```

镜像标签为 Python 3.12.12、CUDA 12.8、cuDNN 9.8.0、NCCL 2.25.1、Linux x86_64。任务配置固定完整 tag。环境版本与基本 GPU 运算在准备镜像或宿主机环境时验收；启动器不执行版本准入或 GPU 探测。启动器在训练前一次构造 metadata，记录任务来源与 Python、Torch、torchvision、CUDA 版本，不作准入判断；训练层原样保存 metadata，将实际 device 和 gpu 单独写入 config.json。

GPU 运行环境由上方固定公共镜像提供。

| 存储 | 容器目录 | 权限 |
|---|---|---|
| `code/<code-version>/` | `/mnt/oss/training-pipeline/code` | 只读 |
| `datasets/cifar10/<data-version>/` | `/mnt/oss/training-pipeline/data` | 只读 |
| `runs/` | `/mnt/oss/training-pipeline/runs` | 读写 |
| 容器本地盘 | `/tmp/vla-training` | 本地可写，用于解压和恢复读取 |

**runs 挂载必须在文件 close 时等待 OSS 上传完成并返回错误。** 建议使用 ossfs 2.0，保持 `sync_upload=true`（默认）。不要在挂载层再次启用异步上传，否则应用无法确认 complete.json 之前的数据已持久化。`mountType=ossfs` 本身不能证明实际安装的是 ossfs 2.0；提交前核对 DLC 实际挂载版本/参数。应用会执行 flush、fsync、close 和读回摘要验证，不支持这些操作的挂载会报错，不静默降级。

参考：[DLC 挂载配置](https://help.aliyun.com/zh/pai/use-cloud-storage-for-a-dlc-job)、[ossfs 2.0 sync_upload 参数](https://help.aliyun.com/zh/oss/developer-reference/description-of-mount-options)。本地文件系统测试不能证明远端持久化语义，读回也可能命中挂载缓存。

准备阶段完成 [DLC 服务授权](https://help.aliyun.com/zh/pai/grant-the-permissions-that-are-required-to-use-dlc)，使 PAI 服务角色具备输入目录读取和 runs 读写权限；配置 PAI CLI（`pai`）的身份凭据，并选定上海工作空间及可用的单 GPU ECS 规格。

当前任务配置对应提交 `52cb36cea9d5b9d2acb6f00a6c2df7ff07c82cc1`。

`dlc/job.yaml` 已配置上海工作空间 `vla_learn`（`1506433`）、Bucket `vla-learn2`，代码和数据版本均为 `release-20260907-100259`。实例采用 `ecs.gn7i-c8g1.2xlarge`（单 NVIDIA A10、24 GiB 显存、8 vCPU、30 GiB 内存），驱动设置为该规格支持的 `550.127.08`。采用按量付费，实际库存以提交时为准。

该版本对应本地 `dist/release-20260907-100259`。发布新版本时，按第 2 节的发布顺序，将其中 `code/` 的内容复制到 OSS 的 `training-pipeline/code/release-20260907-100259/`，将 `data.tar.gz` 和 `data.manifest.json` 复制到 `training-pipeline/datasets/cifar10/release-20260907-100259/`。更换制品时同步修改 YAML 中的版本路径。

配置固定上海公共镜像、1 Worker、按量公共资源组及上表中的三个挂载，默认训练配置为 `configs/train.json`（目标总轮数 20），最长运行 90 分钟、结束保留时长为 0。任务名和时限直接修改 `DisplayName`、`JobMaxRunningTimeMinutes`；PAI CLI 配置通过 `--profile` 选择（按需添加）。

在子项目目录中，使用 [PAI CLI 的 YAML 入口](https://help.aliyun.com/zh/pai/developer-reference/dlc-distributed-training) 预览请求：

```bash
pai dlc job submit \
  --region cn-shanghai \
  --body-json @dlc/job.yaml \
  --dry-run
```

检查输出后提交，获得 CLI 返回的 JobId：

```bash
pai dlc job submit \
  --region cn-shanghai \
  --body-json @dlc/job.yaml
```

每次提交创建一个新任务。存储挂载由 DLC 在容器启动前完成，`UserCommand` 调用下面的入口。续训时修改 `DisplayName`，在 `UserCommand` 中保留 `configs/train.json`、追加 `--resume-run <previous-run-id>`，随后再次预览并提交。目标总轮数为 20，不是额外再跑 20 轮；不追加恢复参数则从头训练。

单轮冒烟的启动命令（完整训练使用 `configs/train.json`）：

```bash
bash /mnt/oss/training-pipeline/code/bootstrap.sh \
  --data-resource-dir /mnt/oss/training-pipeline/data \
  --work-dir /tmp/vla-training \
  --config configs/smoke.json \
  --runs-dir /mnt/oss/training-pipeline/runs \
  --save-queue-size 1
```

Bash 复制源码到本地临时目录，校验文件清单后启动 `python -m cloud.launch`，退出时清理源码副本。资源模块将指定数据归档解压到 work-dir，按摘要缓存并在复用前校验。训练读取这些已准备的数据。

## 4. 异步写盘与完成协议

每次启动产生唯一 run-id，输出直接写 `/mnt/oss/training-pipeline/runs/<run-id>/`，不先落本地 last.pt 再复制：

```text
<run-id>/
  started.json
  config.json
  epochs/
    0001/
      checkpoint.pt
      metrics.json
      complete.json
  succeeded.json      # 所有后台保存成功后才生成
  failed.json         # 出错时尽力写；网络失败时可能不存在
  train.log           # 入口退出时归档，失败不改变训练退出码
```

每轮结束，训练线程先取得独立 CPU 快照，复制模型、优化器、配置和 RNG。复制完成后才开始下一轮，确保训练不会修改后台正保存的数据。后台负责序列化 checkpoint、写指标、关闭文件、校验摘要，最后写 complete.json。每个文件只创建一次，不在 OSS 上追加、覆盖或 rename。

训练层通过 publish_config、publish_epoch 和 check 使用保存器，启动器管理其关闭和等待。训练线程同步取得 CPU 快照，单个后台 I/O 线程执行序列化和文件写入，与下一轮训练重叠。

默认最多 1 个未完成保存（包含正在写的任务），`--save-queue-size` 可设为正整数，由持久化层管理队列约束。队列满时先等空位，再分配快照，限制 CPU 内存。后台失败会阻止后续排队任务发布并在下一次检查/提交/退出时抛出；正常退出等待全部保存。阻塞的挂载系统调用无法由 Python 强制取消，总超时由 DLC 任务控制。

突然终止只丢失未提交轮次。没有完整标记的目录不能恢复；标记或已完成文件损坏会报错，不默默跳过。succeeded.json 表示正常完成；节点被杀时可能只有 started.json，整体运行状态以 DLC 为准。

## 5. 续训与评估

新任务保持相同镜像、数据版本和关键训练参数。仅验证恢复时使用 `configs/resume-smoke.json`（总计 2 轮）；从首轮 checkpoint 继续完整训练时使用 `configs/train.json`（总计 20 轮），并追加：

```text
--resume-run <previous-run-directory-name>
```

默认选旧 run 中最新的完整轮次；也可加 `--resume-best`。恢复检查 checkpoint 与指标的路径/摘要，复制选中的 checkpoint 到本地并再次校验，再恢复训练状态。索引路径相对于 runs 根目录，容器挂载路径变化不影响引用。

最佳 checkpoint 引用随 complete.json 保存，恢复时继承。新 run 没有刷新最佳结果时，引用仍指向旧 run，不能删除被引用的历史目录。清理时需保留仍被引用的 run。本地 .pt 可用 train.py/evaluate.py 读取。

在可以访问 runs 挂载的环境中，提取最佳 checkpoint 并评估：

```bash
python -c 'from pathlib import Path; from persistence.runs import restore_run; restore_run(Path("/mnt/oss/training-pipeline/runs"), "<run-id>", Path("outputs/best.pt"), best=True)'
python evaluate.py outputs/best.pt --data-dir data --device cpu --no-download
```

## 6. 本地验证与云端验收

```bash
source venv/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m pip check
bash -n dlc/bootstrap.sh
```

测试使用普通临时目录模拟挂载，无网络访问，覆盖快照独立、后台重叠、队列反压、退出等待、错误回传、部分归档忽略、摘要拒绝及恢复一致性。

验证结果及待验收事项见 [验证记录](docs/validation.md)；任务、版本与续训关系见 [实验索引](experiments/index.json)。

当前按 CIFAR-10 测试集准确率选择最佳模型，正式比较前应拆分验证集。

## ECS 手动挂载

脚本要求 ECS 安装 ossfs2，并绑定具有目标 OSS 访问权限的 RAM 角色。设置目标存储后按需执行：

```bash
export VLA_OSS_BUCKET="<bucket>"
export VLA_OSS_REGION="cn-shanghai"
export VLA_OSS_ROLE="<ecs-ram-role>"

bash training-pipeline/scripts/oss-mount.sh mount
bash training-pipeline/scripts/oss-mount.sh status
bash training-pipeline/scripts/oss-mount.sh unmount
```

以上命令从仓库根目录执行。code 和 data 在 ECS 读写，runs 只读；每个挂载配置 128 MiB 的软内存预算及较低并发。重复 mount 会检查目标来源和权限后跳过，遇到其他文件系统或非空目录会停止。卸载繁忙目录会报错，不强制卸载。

子项目脚本只设置项目名和数据目录，共用仓库根目录 `scripts/oss-mount.sh` 的挂载逻辑。默认 OSS 前缀为 `training-pipeline/`，默认挂载根目录为 `/mnt/oss/training-pipeline`；后续子项目使用不同项目名即可隔离。`VLA_OSS_BUCKET`、`VLA_OSS_REGION`、`VLA_OSS_ROLE` 为必填配置，可通过 `VLA_OSS_MOUNT_ROOT` 调整挂载根目录；内网挂载要求 ECS 与 Bucket 同地域。挂载完成后，通过文件复制发布资源。

## 容器日志归档

通过 `dlc/bootstrap.sh` 启动时，代码校验和 Python 的 stdout/stderr 同时输出到控制台和本地 `train.log`。训练退出且日志输出结束后，入口将日志一次复制到 `runs/<run-id>/train.log`，返回训练原始退出码。目标可以是 OSS 挂载或本地目录。

正常训练结束以及 Python 报错退出均执行归档；数据准备失败但已分配 run 路径时也可归档。源码校验、参数解析等发生在 run 初始化之前的失败只保留控制台输出及临时本地日志。归档失败会向标准错误报告本地日志路径，不覆盖训练退出码；`succeeded.json` 只表示训练及 checkpoint 保存成功，不代表日志归档成功。整个容器被强杀、节点故障或 OSS 不可写时不保证归档完成。日志仅在退出时上传，训练期间请查看容器日志。

归档后清理本地临时文件；失败时保留临时日志供容器存活期间排查。直接运行 `python -m cloud.launch` 或 `python train.py` 不执行入口脚本的日志捕获和归档。
