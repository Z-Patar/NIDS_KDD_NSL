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


# 定义一个自定义的LSTM模块
class BiLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # LSTM的输出包括所有隐藏状态、最后的隐藏状态和最后的细胞状态
        output, _ = self.lstm(x)
        # 只返回输出张量，不返回隐藏状态和细胞状态
        return output[:, -1, :]  # 只返回最后一个时间步的输出


# 以下是模型的构建过程
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=122, padding=61)  # 保持输出尺寸不变
        self.maxpool1 = nn.MaxPool1d(kernel_size=5)
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.bilstm1 = BiLSTMLayer(input_dim=64, hidden_dim=64)
        self.maxpool1d2 = nn.MaxPool1d(kernel_size=5)
        self.batchnorm2 = nn.BatchNorm1d(1)  # BiLSTM只取了最后一个时间步的输出

        self.bilstm2 = BiLSTMLayer(input_dim=128, hidden_dim=128)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 5)  # 除以5是因为池化层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1d(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = x.permute(0, 2, 1)  # 重排维度以适配LSTM输入

        # 第一个BiLSTM
        x = self.bilstm1(x)
        x = x.unsqueeze(1)  # 增加一个维度以适配MaxPool1d
        x = self.maxpool1d2(x)

        x = self.batchnorm2(x)
        # 第二个BiLSTM
        x = self.bilstm2(x)
        print("Shape after maxpool1d_lstm:", x.shape)

        x = self.dropout(x)
        x = torch.flatten(x, 1)  # 展平除batch_size外的所有维度
        x = self.fc(x)
        x = self.softmax(x)
        return x


# 打印每一层的输出形状
def print_layer_shapes(model, input_tensor):
    def hook(module, input, output):
        print(f"{module.__class__.__name__}: {output.shape}")

    # 注册hook
    hooks = []
    for layer in model.children():
        hook_handle = layer.register_forward_hook(hook)
        hooks.append(hook_handle)

    # 前向传播
    with torch.no_grad():
        model(input_tensor)

    # 移除hooks
    for hook in hooks:
        hook.remove()


# Sequential定义网络，不完全体
# net = nn.Sequential(
#     nn.Conv1d(in_channels=1, out_channels=64, kernel_size=122, padding=61), nn.ReLU(),
#     nn.MaxPool1d(kernel_size=5), nn.BatchNorm1d(64),
#     BiLSTMLayer(64, 64),    # out = torch.Size([1, 64, 128])
#     nn.Flatten(),
#     。。。。。。。。。。
#     nn.Linear(128, 5),  # 五分类问题
#     # nn.Softmax(dim=1),
# )


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
        # 增加一个通道维度
        feature = feature.unsqueeze(0)  # 现在形状变为 [length] -> [1, length]
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
def train_CNN_LSTM(net, train_loader, test_loader, num_epochs, lr, device, w_decay = 1e-5,):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('Training on', device)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=w_decay)
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
    CNN_LSTM_model = MyModel()
    in_tensor = torch.randn(32, 1, 122)  # batch_size=32, in_channels=1, sequence_length=122
    print_layer_shapes(CNN_LSTM_model, in_tensor)

    # 设定超参数
    learning_rate = 0.01
    numb_epochs = 10
    batch_size = 64
    weight_decay = 0.05
    train_file_path = '../Data_encoded/LSTM_data/Train_encoded.csv'  # KDD_NSL的训练集
    test_file_path = '../Data_encoded/LSTM_data/Test_encoded.csv'
    # 加载数据
    train_dataset, test_dataset = load_data(train_file_path, test_file_path)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test__data_loader = DataLoader(test_dataset, batch_size=batch_size)
    myCNNLSTM = MyModel()
    train_CNN_LSTM(myCNNLSTM, train_data_loader, test__data_loader,
              num_epochs=numb_epochs, lr=learning_rate, device=try_device())
