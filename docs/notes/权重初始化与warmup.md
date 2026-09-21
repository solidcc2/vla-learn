## 权重初始化方法与warmup

都是为了让训练初期更稳定，选择一个好的训练起点，避免刚开始训练就出现梯度爆炸/消失。

### kaiming初始化会更适合ReLU

ReLU会把负数全部变为0，$relu(x) = max(0, x)$, 经过多层后，激活方差会不断缩小。

Kaiming初始化会根据输入通道数量调整权重方差，大致是：
$ Var(weight) = 2 / fan_in $
其中fan_in是神经元的输入数量，系数2用来补偿relu丢失的信号, 统计上可以理解为，
2补偿由relu丢失的二阶矩。

这样的好处是：
- 前向传播时，激活值不会快速消失或者爆炸
- 反向传播时，梯度尺度更加稳定
- 深层神经网络更容易开始训练

### resnet会使用kaiming fanout

resnet选择用fanout，是经验选择。个人直觉觉得，是因为深层网络 & 残差网络，
残差（天然小项）+ 反向传播（梯度也是小项）可能更不稳定，因此优先关注反向
传播路径。

resnet的conv2d通常是, conv2d+BN+ReLU的组合，观察非线性算子ReLU，它是引入不稳定
性/二阶矩发生较大变化的主要因素，在正向传播时，fanin经过conv2d+BN，作为ReLU的
输入，已经完成了一定的修正。而反向传播时，fanoutloss经过conv2d到达ReLU，没有
任何稳定性措施，因此考虑选择fanout。

ResNet常用的Conv2d初始化配置
```python
    nn.init.kaiming_normal_(
        module.weight, mode="fan_out", nonlinearity="relu",
    )
```


### BatchNorm的初始化

BatchNorm本身是一个线性变换，因此初始化时通常weight=1，bias=0，维持原样输出，
对于残差块的最后一个BN，因为可以考虑维持$branch(x)+shortcut(x)$和shortcut
一样，$branch(x)==0$的设定。所以最后一个BN的weight也可以初始化为0.

### warmup

通常对于大batch训练（batch=1024等），会选择更大的初始学习率，有可能会在权重
初始位置（数值特征更不稳定），直接跑飞到极端地形导致梯度爆炸/消失，导致loss
传播失败。因此，通常会组合Schedular， 让初期的5个或几个epoch用更保守的固定
学习率来缓慢更新权重（或者从小学习率逐步增加到启动学习率），以到达一个相对
正确的方向，做warmup，再按照标准的调度器去决策学习率。
