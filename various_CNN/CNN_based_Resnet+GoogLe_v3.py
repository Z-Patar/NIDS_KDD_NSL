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
    def __init__(self, in_channels, c1, c2, c3, c4, num_channels, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，1x1卷积层后接两个3*3卷积层
        self.p1_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p1_2 = nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1)
        self.p1_3 = nn.Conv2d(c3[1], c3[2], kernel_size=3, padding=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，3x3最大汇聚层后接1x1卷积层
        self.p3_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p3_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        # 线路4，单1x1卷积层
        self.p4_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        self.bn1_1 = nn.BatchNorm2d(num_channels)
        self.bn1_2 = nn.BatchNorm2d(num_channels)
        self.bn1_3 = nn.BatchNorm2d(num_channels)
        self.bn2_1 = nn.BatchNorm2d(num_channels)
        self.bn2_2 = nn.BatchNorm2d(num_channels)
        self.bn3_1 = nn.BatchNorm2d(num_channels)
        self.bn3_2 = nn.BatchNorm2d(num_channels)
        self.bn4_1 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        # 线路1，1x1卷积层后接两个3*3卷积层
        p1 = self.bn3_3(self.p1_3(  # 对第3个卷积做BN
            F.relu(self.bn1_2(self.p1_2(  # 对第2个卷积先BN后ReLU
                F.relu(self.bn1_1(self.p1_1(x)))  # 对第1个卷积先BN后ReLU
            )))
        ))
        # 线路2，1x1卷积层后接3x3卷积层
        p2 = self.bn2_2(self.p2_2(  # 对第2个卷积做BN
            F.relu(self.bn2_1(self.p2_1(x)))  # 对第1个卷积先BN后ReLU
        ))
        # 线路3，3x3最大汇聚层后接1x1卷积层
        p3 = self.bn3_2(self.p3_2(  # 对第2个卷积做BN
            self.bn3_1(self.p3_1(x))  # 对第1个池化做BN
        ))
        # 线路4，单1x1卷积层，卷积后BN
        p4 = self.bn4_1(self.p4_1(x))

        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# todo 根据Inception搭建resnet class
# class Residual(nn.Module):
#     def __init__(self, in_channels, c1, c2, c3, c4, num_channels,use_1conv=False, strides=1, **kwargs)
#         super(Residual, self).__init__(**kwargs)
#         Inception(in_channels, c1, c2, c3, c4, num_channels)
#         if use_1conv:
#             self.conv3 = nn.Conv2d(in_channels, num_channels,
#                                    kernel_size=1, stride=strides)
#         else:
#             self.conv3 = None
#
#     def forward(self, X):
#         y = # todo定义Y的类型,应该是一个张量，但要根据Inception，forward函数的返回值确定
#         if self.conv3:
#             X = F.relu(self.conv3(X))
#         Y += X
#         return F.relu(Y)


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
