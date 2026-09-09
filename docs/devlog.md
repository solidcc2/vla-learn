## 20260909
1. 增加笔记，对于SimpleCNN的网络结构&主要算子总结。

## 20260908
1. 补齐两次 A10 DLC 训练的[实验索引](../training-pipeline/experiments/index.json)，关联首轮冒烟和完整续训；挂载读回核对两份 checkpoint 摘要。详见[验证记录](../training-pipeline/docs/validation.md)。
2. 拆分阿里云资源、DLC 任务、训练制品和实验溯源 skills，新增工作区检查与实验索引登记脚本。

## 20260907
1. 确定基础镜像，基于基础镜像重建当前环境依赖。
2. 基于基础镜像，本地gpu运行验证通过。
3. 整理aliyun cli & pai相关skill。
4. 发布训练制品至 OSS；完成 A10 单卡 DLC 首轮冒烟，测试准确率 57.64%，checkpoint 和日志归档成功。完整训练配置已切换为 A10、20 轮，对应提交 `52cb36cea9d5b9d2acb6f00a6c2df7ff07c82cc1`。详见 [验证记录](../training-pipeline/docs/validation.md)。

5. 新 DLC 任务从冒烟 checkpoint 的第 2 轮恢复，完成目标总轮数 20；最终及最佳测试准确率 82.27%。结果、耗时和待验收项见上方验证记录链接。

## 20260906
1. 搭建基于阿里云2c2g的vm的开发环境，压缩资源使用，增加pai & oss等云平台训练 & 网络环境配置。
2. 完成初版最小cifar-10 simplecnn分类器训练代码，学习并理解，完成cpu首个epoch训练。
3. 回顾多分类交叉熵loss和相关延伸loss函数。
4. 回顾主要的神经网络层的语义。

## 20260904
1. 初始化项目，初始化个人workbench配置，初始化代理 & ai工具。
2. 完成项目计划初版敲定。
