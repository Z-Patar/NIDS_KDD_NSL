#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

# python3.10
# -*- coding:utf-8 -*-
# @Time    :2024/5/2 上午11:38
# @Author  :Z_Patar
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，1x1卷积层后接两个3*3卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1[0], kernel_size=1)
        self.p1_2 = nn.Conv2d(c1[0], c1[1], kernel_size=3, padding=1)
        self.p1_3 = nn.Conv2d(c1[1], c1[2], kernel_size=3, padding=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，3x3最大汇聚层后接1x1卷积层
        self.p3_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p3_2 = nn.Conv2d(in_channels, c3, kernel_size=1)
        # 线路4，单1x1卷积层
        self.p4_1 = nn.Conv2d(in_channels, c4, kernel_size=1)

        self.bn1_1 = nn.BatchNorm2d(c1[0])
        self.bn1_2 = nn.BatchNorm2d(c1[1])
        self.bn1_3 = nn.BatchNorm2d(c1[2])
        self.bn2_1 = nn.BatchNorm2d(c2[0])
        self.bn2_2 = nn.BatchNorm2d(c2[1])
        self.bn3_2 = nn.BatchNorm2d(c3)  # maxPool不用BN
        self.bn4_1 = nn.BatchNorm2d(c4)

    def forward(self, x):
        # 线路1，1x1卷积层后接两个3*3卷积层
        p1 = self.bn1_3(self.p1_3(  # 对第3个卷积做BN
            F.relu(self.bn1_2(self.p1_2(  # 对第2个卷积先BN后ReLU
                F.relu(self.bn1_1(self.p1_1(x)))  # 对第1个卷积先BN后ReLU
            )))
        ))
        # 线路2，1x1卷积层后接3x3卷积层
        p2 = self.bn2_2(self.p2_2(  # 对第2个卷积做BN
            F.relu(self.bn2_1(self.p2_1(x)))  # 对第1个卷积先BN后ReLU
        ))
        # 线路3，3x3最大汇聚层后接1x1卷积层
        p3 = self.bn3_2(self.p3_2(  # 1x1卷积，对第2个卷积做BN
            self.p3_1(x)  # 池化
        ))
        # 线路4，单1x1卷积层，卷积后BN
        p4 = self.bn4_1(self.p4_1(x))

        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# todo 根据Inception搭建resnet
class Residual(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.inception = Inception(in_channels, c1, c2, c3, c4)
        self.use_1x1conv = use_1x1conv
        if use_1x1conv:  # 如果这个条件被IDE标红报错了，那只是它在抽风，你和代码有一个能跑就行
            num_channels = c1[2] + c2[1] + c3 + c4
            self.conv1x1 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=strides)
            self.bn = nn.BatchNorm2d(num_channels)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.inception(x)
        if self.use_1x1conv:
            x = self.bn(self.conv1x1(x))
        y += x
        return F.relu(y)


# 定义Resnet前的网络部分, out.shape = (batch_size, 32, 11, 11)
block1 = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=1),  # 1*1卷积
    nn.BatchNorm2d(16), nn.ReLU(),  # BN+ReLU
    nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 3*3卷积，padding=1
    # nn.MaxPool2d(kernel_size=3, padding=1)
)
# Residual*2+一个3x3conv, out.shape = (batch_size, 48, 9, 9)
block2 = nn.Sequential(
    # todo 完善所有Residual
    # Residual  channel=32
    # Residual  channel=32
    nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(48), nn.ReLU(),  # 注意通道数
)
# Residual*2+一个3x3conv, out.shape = (batch_size, 64, 7, 7)
block3 = nn.Sequential(
    # Residual  channel=48
    # Residual  channel=48
    nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(),  # 注意通道数
)
# Residual*2+一个3x3conv, out.shape = (batch_size, 80, 5, 5)
block4 = nn.Sequential(
    # Residual*2
    nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(80), nn.ReLU(),
)
# Residual*2+一个3x3conv, out.shape = (batch_size, 128, 3, 3)
block5 = nn.Sequential(
    # Residual*2
    nn.Conv2d(80, 128, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(128), nn.ReLU(),
)
net = nn.Sequential(
    block1, block2, block3, block4, block5,
    nn.MaxPool2d(kernel_size=3),  # 对3x3大小的窗口池化，out.shape=(batch_size, 128, 1, 1)
    nn.Flatten(),
    nn.Linear(128, 10),
)

if __name__ == "__main__":
    X = torch.rand(size=(1, 1, 11, 11))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)