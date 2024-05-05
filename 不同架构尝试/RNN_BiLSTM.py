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
from sklearn.model_selection import StratifiedKFold
import seaborn as sns


net = nn.Sequential(
    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=122, padding=61), nn.ReLU(),
    nn.MaxPool1d(kernel_size=5),
    nn.BatchNorm1d(64),
    nn.LSTM(64, 64, bidirectional=True, batch_first=True),
    nn.Flatten(),
    nn.Linear(128, 5),  # 五分类问题
    nn.Softmax(dim=1),
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


# 取消将一维向量reshape为矩阵
def load_data(train_file_path, test_file_path):
    # 处理训练集
    train_df = pd.read_csv(train_file_path, header=0)  # 读取数据文件
    train_features = train_df.iloc[:, :-5].values  # 特征项为前121列
    train_labels = train_df.iloc[:, -5:].values  # label项为后5列

    # 处理测试集，方法同训练集
    test_df = pd.read_csv(test_file_path, header=0)
    test_features = test_df.iloc[:, :-5].values
    test_labels = test_df.iloc[:, -5:].values

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
    print(net)
    X = torch.rand(size=(1, 1, 122))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    # 设定超参数
    # learning_rate = 0.01
    # numb_epochs = 10
    # batch_size = 64
    # train_file_path = '../Data_encoded/matrix_data/Train_encoded.csv'  # KDD_NSL的训练集
    # test_file_path = '../Data_encoded/matrix_data/Test_encoded.csv'
    # # 加载数据
    # train_dataset, test_dataset = load_data(train_file_path, test_file_path)
    # train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test__data_loader = DataLoader(test_dataset, batch_size=batch_size)
    # model = net
    # train_ch6(model, train_data_loader, test__data_loader,
    #           num_epochs=numb_epochs, lr=learning_rate, device=try_device())
