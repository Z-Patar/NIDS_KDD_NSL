# python3.10
# -*- coding:utf-8 -*-
# Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.
# @Time   :2024/4/15 下午3:11
# @Author :Z_Patar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1：通道1，输出通道32，卷积核大小5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # 卷积层2：输入通道32，输出通道64，卷积核大小5
        self.conv2 = nn.Conv2d(32, 64, 3)
        # 池化层，采用最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 两个全连接层
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.25)  # 添加 dropout 层

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 在全连接层后应用 dropout
        x = self.fc2(x)
        return x


def next_batch(feature_list, label_list, size):
    feature_batch_temp = []
    label_batch_temp = []
    f_list = random.sample(range(len(feature_list)), size)
    for i in f_list:
        feature_batch_temp.append(feature_list[i])
    for i in f_list:
        label_batch_temp.append(label_list[i])
    return feature_batch_temp, label_batch_temp


# 使用Pandas加载数据
def load_data():
    global feature
    global label
    global feature_full
    global label_full
    file_path = '/home/peter/Desktop/pycharm/ids-kdd99/kddcup.data_10_percent_corrected_handled2.cvs'
    df = pd.read_csv(file_path, header=None)
    feature = df.iloc[:, :36].values
    label = np.zeros((len(df), 23))
    label[np.arange(len(df)), df.iloc[:, 41].values] = 1

    file_path_full = '/home/peter/Desktop/pycharm/ids-kdd99/kddcup.data.corrected_handled2.cvs'
    df_full = pd.read_csv(file_path_full, header=None)
    feature_full = df_full.iloc[:, :36].values
    label_full = np.zeros((len(df_full), 23))
    label_full[np.arange(len(df_full)), df_full.iloc[:, 41].values] = 1


def main():
    # 初始化模型
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 加载数据
    load_data()

    # 训练模型
    batch_size = 32
    for epoch in range(10):
        for i in range(0, len(feature), batch_size):
            inputs = torch.Tensor(feature[i:i + batch_size])
            labels = torch.Tensor(label[i:i + batch_size])

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, 10, i, len(feature), loss.item()))

    print('Finished Training')


if __name__ == '__main__':
    main()
