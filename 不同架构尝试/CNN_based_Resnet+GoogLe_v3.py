#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

# python3.10
# -*- coding:utf-8 -*-
# @Time    :2024/5/2 上午11:38
# @Author  :Z_Patar
import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


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


# todo 当前所有resnet_inception块都没有改变通道数，稍后尝试在resnet_inception中改变通道数，
# 定义Resnet前的网络部分,
before_resnet_block = nn.Sequential(
    # nn.Conv2d(1, 8, kernel_size=1),  # 1*1卷积
    # # nn.BatchNorm2d(8), nn.ReLU(),  # BN+ReLU
    # nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 3*3卷积，padding=1
    # nn.BatchNorm2d(8), nn.ReLU(),  # BN+ReLU
)  # out.shape = (batch_size, 8, 11, 11)

# Resnet 1:Residual*2+一个3x3conv,in.shape = (-1, 8, 11, 11)
resnet_block1 = nn.Sequential(
    # Residual(in_channels, (c1[0], c1[1], c1[2]), (c2[0], c2[1]), c3, c4, use_1x1conv)
    Residual(1, (4, 1, 1), (2, 4), 1, 2, True),
    Residual(8, (4, 1, 1), (2, 4), 1, 2, ),
    nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),  # 8,11x11 -> 16,9x9
    nn.BatchNorm2d(16), nn.ReLU(),  # 注意通道数
)  # out.shape = (batch_size, 16, 9, 9)

# Resnet 2:Residual*2 + 一个3x3conv,
resnet_block2 = nn.Sequential(
    # Residual(in_channels, (c1[0], c1[1], c1[2]), (c2[0], c2[1]), c3, c4, use_1x1conv)
    Residual(16, (2, 2, 2), (4, 8), 2, 4, ),
    Residual(16, (2, 2, 2), (4, 8), 2, 4, ),
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),  # 16,9x9 -> 32,7x7
    nn.BatchNorm2d(32), nn.ReLU(),  # 注意通道数
)  # out.shape = (batch_size, 32, 7, 7)

resnet_block3 = nn.Sequential(
    # Residual(in_channels, (c1[0], c1[1], c1[2]), (c2[0], c2[1]), c3, c4, use_1x1conv)
    Residual(32, (2, 2, 4), (8, 16), 4, 8, ),
    Residual(32, (2, 2, 4), (8, 16), 4, 8, ),
    # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),  # 64,7x7 -> 64,3x3
    Residual(32, (2, 4, 6), (12, 24), 6, 12, use_1x1conv=True),
    Residual(48, (2, 4, 8), (16, 32), 8, 16, use_1x1conv=True),
    nn.BatchNorm2d(64), nn.ReLU(),
)  # out.shape = (batch_size, 64, 3, 3)

# # Resnet 3: Residual*2+一个3x3conv,
# resnet_block3 = nn.Sequential(
#     # Residual(in_channels, (c1[0], c1[1], c1[2]), (c2[0], c2[1]), c3, c4, use_1x1conv)
#     Residual(64, (2, 4, 8), (16, 32), 8, 16, ),
#     Residual(64, (2, 4, 8), (16, 32), 8, 16, ),
#     nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0),  # 64,7x7 -> 96,5x5
#     nn.BatchNorm2d(96), nn.ReLU(),
# )  # out.shape = (batch_size, 96, 5, 5)
#
# # Resnet 4:Residual*2+一个3x3conv,
# resnet_block4 = nn.Sequential(
#     # Residual(in_channels, (c1[0], c1[1], c1[2]), (c2[0], c2[1]), c3, c4, use_1x1conv)
#     Residual(96, (4, 8, 12), (24, 48), 12, 24, ),
#     Residual(96, (4, 8, 12), (24, 48), 12, 24, ),
#     nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(128), nn.ReLU(),
# )  # out.shape = (batch_size, 128, 3, 3)
# todo 模型过拟合严重,需要修改网络架构

net = nn.Sequential(
    before_resnet_block, resnet_block1, resnet_block2, resnet_block3,
    nn.MaxPool2d(kernel_size=4),  # 对3x3大小的窗口池化，out.shape=(batch_size, 64, 1, 1)
    nn.Flatten(),
    nn.Linear(64, 5),  # 五分类问题
)


# 以下是自定义的Dataset类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        # self.labels = labels
        # 假设labels是独热编码的，取最大值的索引作为类别标签 # nn.CrossEntropyLoss 期望目标是类别的索引，而不是独热编码的形式。
        self.labels = torch.argmax(torch.tensor(labels, dtype=torch.float32), dim=1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        # label = torch.tensor(self.labels[idx], dtype=torch.long)
        label = self.labels[idx]
        return feature, label


def load_data(train_file_path, test_file_path):
    # 处理训练集
    train_df = pd.read_csv(train_file_path, header=0)  # 读取数据文件
    train_df = train_df.drop(columns=['is_host_login'])  # 删除is_host_login列，该列值全为0，删除不影响结果
    train_features = train_df.iloc[:, :-5].values  # 特征项为前121列
    train_labels = train_df.iloc[:, -5:].values  # label项为后5列
    # 将特征reshape为一个四维向量，其中 batch_size 等于数据项数, channel = 1, 高，宽为11的特征矩阵
    train_features = train_features.reshape(-1, 1, 11, 11)

    # 处理测试集，方法同训练集
    test_df = pd.read_csv(test_file_path, header=0)
    test_df = test_df.drop(columns=['is_host_login'])
    test_features = test_df.iloc[:, :-5].values
    test_labels = test_df.iloc[:, -5:].values
    test_features = test_features.reshape(-1, 1, 11, 11)

    # 创建数据集和数据加载器
    # 训练集
    train_dataset = CustomDataset(train_features, train_labels)
    # 测试集
    test_dataset = CustomDataset(test_features, test_labels)

    return train_dataset, test_dataset


# 检查CUDA是否可用
def try_device():
    if torch.cuda.is_available():
        # 选择第一个CUDA设备
        device = torch.device("cuda:0")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")
    return device


# 重写的train_ch6函数
def train_ch6(net, train_loader, test_loader, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('Training on', device)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # 训练模型
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_accuracy = correct / total
        train_loss /= len(train_loader)

        # Evaluate test set accuracy
        test_accuracy = evaluate_accuracy(net, test_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')

    # 绘制学习曲线
    plot_learning_curve(train_losses, train_accuracies, test_accuracies, num_epochs)
    # 绘制混淆矩阵
    class_labels = ['DOS 0', 'Normal 1', 'Probe', 'R2L', 'U2R']
    plot_confusion_matrix(net, test_loader, device, class_labels)


def plot_learning_curve(train_losses, train_accuracies, test_accuracies, num_epochs):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_accuracy(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    return accuracy


# todo 混淆矩阵的坐标轴从数字改为坐标
def plot_confusion_matrix(net, data_loader, device, class_names):
    net.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    # 查看模型每一层的输出
    X = torch.rand(size=(1, 1, 11, 11))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    # 设定超参数
    learning_rate = 0.01
    numb_epochs = 10
    batch_size = 256
    train_file_path = '../Data_encoded/matrix_data/Train_encoded.csv'  # KDD_NSL的训练集
    # train_file_path = '../KDDCup99/data_encoded/Train_encoded.csv'  # KDD Cup99的训练集
    test_file_path = '../Data_encoded/matrix_data/Test_encoded.csv'
    # 加载数据
    train_dataset, test_dataset = load_data(train_file_path, test_file_path)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test__data_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = net
    train_ch6(model, train_data_loader, test__data_loader,
              num_epochs=numb_epochs, lr=learning_rate, device=try_device())
