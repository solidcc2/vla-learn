---
name: training-experiment-tracking
description: 在阿里云 PAI DLC 训练前检查仓库变更，或在任务结束后登记训练结果、代码与任务配置的仓库版本，关联 JobId、run_id 和 checkpoint 续训来源时使用。
---

# PAI DLC 训练结果溯源

激活目标子项目的 Python 3.12 venv，从仓库根目录调用 [tracking.py](scripts/tracking.py)。训练前使用 `check-workspace`，任务结束后使用 `record-result`；参数、输入示例及处理规则查看对应子命令的 `--help`：

```bash
python .agents/skills/training-experiment-tracking/scripts/tracking.py --help
```

每个子项目统一使用 `experiments/index.json`。允许变更的记录路径在 [config.json](config.json) 中配置。

`git_commit` 表示训练前锁定的代码和基础任务配置版本，发布制品应与该提交对应，不能用结束时的 HEAD 代替。镜像、实际参数、指标及日志通过 JobId 和 OSS 结果目录查阅。
