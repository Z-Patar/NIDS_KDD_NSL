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
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import seaborn as sns


# 定义一个BiLSTM模块
class BiLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # LSTM的输出包括所有隐藏状态、最后的隐藏状态和最后的细胞状态
        output, _ = self.lstm(x)
        # 只返回输出张量，不返回隐藏状态和细胞状态
        # return output
        return output[:, -1, :]  # 只返回最后一个时间步的输出


# CNN_BiLSTM_Model构建
class CNNBiLSTMModel(nn.Module):
    def __init__(self):
        super(CNNBiLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=122, padding='same')  # 保持输出尺寸不变
        self.maxpool1 = nn.MaxPool1d(kernel_size=5)
        self.batchnorm1 = nn.BatchNorm1d(64)
        # out.shape=(batch=32, channel = 64, seq=24(122池化后的数字))

        # input_dim = 就是nn.LSTM(input_size(x的特征维度),hidden_size,...)中的input_size,
        # 在该数据中,input_size恒为1

        self.bilstm1 = BiLSTMLayer(input_dim=1, hidden_dim=64)   # hidden_size即为上一层的输出channel

        # 此处需要将(128, ) reshape为(1,128), 因为要沿着128的方向做池化,# todo 为啥要沿128的方向没看懂
        self.maxpool1d2 = nn.MaxPool1d(kernel_size=5)
        self.batchnorm2 = nn.BatchNorm1d(1)

        # 第二个BiLSTM
        # input=(input_size=1, hidden_size=128, 其他默认) ,seq=25(根据上一层的输出判断的)
        self.bilstm2 = BiLSTMLayer(input_dim=1, hidden_dim=128) # BiLSTM只取了最后一个时间步的输出
        # out.shape = (batch=32, 1(啥意思暂不明白), 256(就是128池化后的数字))

        self.dropout = nn.Dropout(0.5)  # 将上一层随机丢弃一半传入下层
        self.fc = nn.Linear(256, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1d(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        # shape=(32, 64, 24)

        x = x.permute(0, 2, 1)  # 重排维度以适配LSTM输入
        # shape=(32, 24, 64)

        # 第一个BiLSTM
        # BiLSTM.output.shape = (batch, seq, hidden_size*2) = (32, 24, 128)
        x = self.bilstm1(x)     # 但此处只取了最后一个seq, 此时x.shape=(32,128)
        x = x.unsqueeze(1)  # 增加一个维度以适配MaxPool1d
        # shape=(32,1,128)

        x = self.maxpool1d2(x)  # shape=(32, 1, 24)
        x = self.batchnorm2(x)
        # out.shape=(32, 1, 24)

        x = x.permute(0, 2, 1)  # 重排维度以适配LSTM输入
        # out.shape=(32, 24, 1)

        # 第二个BiLSTM
        x = self.bilstm2(x)
        # out.shape=(batch=32, 256)

        x = self.dropout(x)
        # x = torch.flatten(x, 1)  # 展平除batch_size外的所有维度, 但是维度已经是(batch, 256)了,没得展了
        x = self.fc(x)
        x = self.softmax(x)
        return x


# [调试用]打印每一层的输出形状
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


# 加载数据
def load_combined_data():
    # 读取数据文件
    data = pd.read_csv('../Data_encoded/LSTM_data/combined_data_processed.csv')

    return data


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


# 混淆矩阵的坐标轴label在PyCharm里看不见,但另存图片后可见
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


# todo 模型已搭建完毕,根据搭建好的模型写出训练函数,K折交叉验证
# def train_CNN_LSTM(net, train_loader, test_loader, num_epochs, lr, device, w_decay=1e-5,):
#     def init_weights(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             nn.init.xavier_uniform_(m.weight)
#
#     net.apply(init_weights)
#     print('Training on', device)
#     net.to(device)
#
#     optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=w_decay)
#     loss_fn = nn.CrossEntropyLoss()
#
#     train_losses = []
#     train_accuracies = []
#     test_accuracies = []
#
#     # 训练模型
#     for epoch in range(num_epochs):
#         net.train()
#         train_loss = 0.0
#         correct = 0
#         total = 0
#
#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             y_hat = net(X)
#             loss = loss_fn(y_hat, y)
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#             _, predicted = torch.max(y_hat, 1)
#             total += y.size(0)
#             correct += (predicted == y).sum().item()
#
#         train_accuracy = correct / total
#         train_loss /= len(train_loader)
#
#         # Evaluate test set accuracy
#         test_accuracy = evaluate_accuracy(net, test_loader, device)
#
#         train_losses.append(train_loss)
#         train_accuracies.append(train_accuracy)
#         test_accuracies.append(test_accuracy)
#
#         print(
#             f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
#
#     # 绘制学习曲线
#     plot_learning_curve(train_losses, train_accuracies, test_accuracies, num_epochs)
#     # 绘制混淆矩阵
#     class_labels = ['DOS 0', 'Normal 1', 'Probe', 'R2L', 'U2R']
#     plot_confusion_matrix(net, test_loader, device, class_labels)
def train_CNN_LSTM_with_cross_validation(net, kfold, data, num_epochs, lr, device, w_decay=1e-5):
    oos_pred = []  # 用于存储每个验证集的准确率
    combined_data = data.copy()
    combined_data_X = pd.DataFrame(combined_data.iloc[:, :-1].values)  # 要求数据为dataframe形式
    y_train = combined_data.iloc[:, -5:].values  # label项为后5列

    for train_index, test_index in kfold.split(combined_data_X, y_train):
        train_X, test_X = combined_data_X.iloc[train_index], combined_data_X.iloc[test_index]
        train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]

        x_columns_train = train_X.columns
        x_train_array = train_X[x_columns_train].values
        x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))

        dummies = pd.get_dummies(train_y)
        y_train_1 = dummies.values

        x_columns_test = test_X.columns
        x_test_array = test_X[x_columns_test].values
        x_test_2 = np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))

        dummies_test = pd.get_dummies(test_y)
        y_test_2 = dummies_test.values

        net.apply(init_weights)
        net.to(device)

        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=w_decay)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            net.train()
            optimizer.zero_grad()
            y_hat = net(torch.Tensor(x_train_1).to(device))
            loss = loss_fn(y_hat, torch.LongTensor(np.argmax(y_train_1, axis=1)).to(device))
            loss.backward()
            optimizer.step()

        net.eval()
        pred = net(torch.Tensor(x_test_2).to(device))
        pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
        y_eval = np.argmax(y_test_2, axis=1)
        score = metrics.accuracy_score(y_eval, pred)
        oos_pred.append(score)
        print("Validation score: {}".format(score))

    return oos_pred


if __name__ == "__main__":
    # # 查看模型每一层的输出
    # CNN_LSTM_model = CNNBiLSTMModel()
    # in_tensor = torch.randn(32, 1, 122)  # batch_size=32, in_channels=1, sequence_length=122
    # print_layer_shapes(CNN_LSTM_model, in_tensor)
    #
    # # 数据文件路径
    # train_file = '../Data_encoded/LSTM_data/Train_encoded.csv'  # KDD_NSL的训练集
    # test_file = '../Data_encoded/LSTM_data/Test_encoded.csv'

    # 设定超参数
    learning_rate = 0.01
    numb_epochs = 10
    batch_size = 64
    weight_decay = 0.05
    num_folds = 6
    k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)   # 随机种子固定,保证每次生成的都一样

    # 加载数据
    combined_data = load_combined_data()
    labels = combined_data['Class']  # 先分离Class
    combined_data_features = combined_data.drop(columns='Class')

    print(f'combined_data.shape = {combined_data.shape}')

    # myModel = CNNBiLSTMModel()
    # train_CNN_LSTM_with_cross_validation(myModel, k_fold, combined_data, num_epochs=numb_epochs, lr=learning_rate, device=try_device(), w_decay=weight_decay)

    # 查看随机划分后的数据样式
    for train_index, test_index in k_fold.split(combined_data_features, labels):
        train_X, test_X = combined_data_features.iloc[train_index], combined_data_features.iloc[test_index]
        print(f'train_X.shape = {train_X.shape}, test_X.shape = {test_X.shape}\n')
        train_y, test_y = labels.iloc[train_index], labels.iloc[test_index]
        print(f'train_y.shape = {train_y.shape}, test_y.shape = {test_y.shape}\n')

        # todo 继续修改train函数
        # x_columns_train = new_train_df.columns.drop('Class')
        # x_train_array = train_X[x_columns_train].values
        # x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))

