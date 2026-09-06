# CIFAR-10 CNN 设计

提供一个便于阅读的 CIFAR-10 十分类基线，展示数据加载、模型、训练、验证和 checkpoint 的边界。使用 PyTorch、torchvision 和三层卷积 `SimpleCNN`；本轮只写代码，不安装依赖、下载数据或执行训练。

核心包按职责拆分：`data.py` 负责数据，`model.py` 负责网络，`engine.py` 负责单轮训练与评估，`checkpoint.py` 负责持久化。入口脚本只解析参数和编排组件。

测试覆盖输出形状、训练指标、checkpoint 往返和数据变换。当前虚拟环境没有 PyTorch，因此本轮只进行语法检查。
