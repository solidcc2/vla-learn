前面主要讨论了这些主题：

1. CIFAR-10 十分类实现方案

- 为什么第一版选择简单 CNN
- 数据加载、训练、验证、checkpoint 和评估的整体结构
- 已在仓库中写入 CNN 代码框架，但没有安装 PyTorch 或启动训练

2. 损失函数基础

- `criterion` 是什么
- 损失计算、反向传播和参数更新的区别
- MSE、L1、Smooth L1、交叉熵、BCE、Focal Loss 等常见损失
- 如何根据任务、输出和标签格式选择损失函数

3. `CrossEntropyLoss`

- Softmax 为什么能把 logits 转换成概率
- 为什么使用指数函数和自然对数
- 负对数概率的信息论含义
- one-hot 编码
- batch 损失为什么通常取平均
- 梯度公式：

\[
\frac{\partial L}{\partial z_c}=p_c-y_c
\]

- `CrossEntropyLoss = LogSoftmax + NLLLoss`
- 与最大似然估计的关系

4. 信息论与 KL 散度

- bit 和 nat 的区别
- 香农熵表示分布的不确定性
- 交叉熵衡量用预测分布描述真实数据的成本
- KL 散度衡量两个概率分布的差异
- 为什么 KL 散度不是严格的“距离”
- 关系式：

\[
H(q,p)=H(q)+D_{\mathrm{KL}}(q\|p)
\]

- 为什么训练时直接优化交叉熵，不需要减去真实熵

5. 多分类与多标签分类

- 单标签多分类使用 Softmax + CrossEntropyLoss
- 多标签任务可以拆成多个二分类问题
- `BCEWithLogitsLoss` 的条件独立近似
- 共享网络如何间接学习标签相关性
- 标签依赖较强时，可以考虑自回归或结构化预测

6. MSE 的概率解释

- MSE 与高斯噪声、最大似然估计的关系
- 高斯假设针对的是预测残差，不是要求目标值本身服从高斯分布
- 异常值、异方差、多峰结果等不适合普通 MSE 的情况

7. PyTorch 基础

- `loss.item()` 把单元素 Tensor 转成 Python 数字
- 为什么日志统计时使用 `.item()`
- 为什么不能用 `.item()` 的结果反向传播

8. CNN 卷积模块

讨论了：

```text
Conv2d
→ BatchNorm2d
→ ReLU
→ MaxPool2d
```

包括：

- `Conv2d` 不是局部取平均，而是可学习的局部加权求和
- 卷积核形状，例如 `[32, 3, 3, 3]`
- 卷积是带有局部连接和权重共享约束的线性变换
- 卷积也可以展开成特殊的稀疏矩阵 \(Wx\)
- BatchNorm 的均值、方差、\(\gamma\) 和 \(\beta\)
- 为什么 Conv 后紧跟 BatchNorm 时通常可以设置 `bias=False`
- ReLU 和 MaxPool 的作用

9. 通道增加与空间降采样

- 为什么通道从 `32 → 64 → 128`
- 为什么空间尺寸从 `32 → 16 → 8 → 4`
- 通道是连续特征空间的维度，不是严格的编码 bit
- 每个空间位置对应一个局部特征向量
- 感受野如何逐层扩大
- MaxPool 不是严格的低通滤波
- MaxPool 更像按通道保留局部最强特征证据
- 可以使用步长卷积、平均池化或全局平均池化替代

10. 分类器结构

讨论了：

```text
Flatten
→ Linear
→ ReLU
→ Dropout
→ Linear
```

包括：

- 两层 MLP 为什么比单层线性分类器表达能力更强
- CNN 特征后面是否可以接 SVM
- 为什么当前网络在 `4×4` 停止下采样
- `4×4` 在空间信息、参数量和计算量之间的折中
- 是否可以降到 `1×1`
- `AdaptiveAvgPool2d(1) + Linear` 的现代 CNN 设计

11. Codex 目录访问控制

- `.gitignore` 不是安全边界
- `AGENTS.md` 是行为约束，不是真正的文件权限
- 可以通过 Codex filesystem permission profile 的 `deny` 禁止读取特定目录或文件模式，例如 `.env`、密钥和私有数据目录

整体学习路线可以概括为：

```text
训练流程
→ 损失函数与反向传播
→ 概率论和信息论基础
→ CNN 局部特征提取
→ 特征空间与降采样
→ 最终分类器
```