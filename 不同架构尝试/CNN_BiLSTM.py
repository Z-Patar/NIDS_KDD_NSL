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


# 定义了一个自定义的数据集类CustomDataset，这个类继承了PyTorch的Dataset类，可以用于加载数据。
class CustomDataset(Dataset):
    # 类的构造函数。它接受两个参数features和labels，分别表示数据集的特征和标签。
    # 在初始化过程中，将这些特征和标签存储在类的实例变量中。
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    # 这是一个特殊方法，用于返回数据集的长度（即数据样本的数量）。
    # 在这个方法中，它返回了存储在features中的样本数量，即数据集的长度。
    def __len__(self):
        return len(self.features)

    # 这也是一个特殊方法，用于根据给定索引idx来获取数据集中的样本。
    # 在这个方法中，它根据索引idx从features和labels中获取对应索引的特征和标签，并将它们作为元组返回。
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        # 确保 feature 是一个数值型数组
        if isinstance(feature, np.ndarray):
            if feature.dtype.type is np.str_ or feature.dtype.type is np.object_:
                raise ValueError("Features must be numeric")

        # 如果 feature 不是一个 ndarray，或者它的 dtype 不是浮点数，尝试将其转换
        if not isinstance(feature, np.ndarray) or feature.dtype != 'float32':
            feature = np.array(feature, dtype=np.float32)

        # 转换为 PyTorch 张量
        feature = torch.tensor(feature, dtype=torch.float32)

        # 如果标签不是一个张量，转换它
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)

        return feature, label


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

        self.bilstm1 = BiLSTMLayer(input_dim=1, hidden_dim=64)  # hidden_size即为上一层的输出channel

        # 此处需要将(128, ) reshape为(1,128), 因为要沿着128的方向做池化,# todo 为啥要沿128的方向没看懂
        self.maxpool1d2 = nn.MaxPool1d(kernel_size=5)
        self.batchnorm2 = nn.BatchNorm1d(1)

        # 第二个BiLSTM
        # input=(input_size=1, hidden_size=128, 其他默认) ,seq=25(根据上一层的输出判断的)
        self.bilstm2 = BiLSTMLayer(input_dim=1, hidden_dim=128)  # BiLSTM只取了最后一个时间步的输出
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
        x = self.bilstm1(x)  # 但此处只取了最后一个seq, 此时x.shape=(32,128)
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


# todo 模型已搭建完毕,根据搭建好的模型写出训练函数
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
def loop_data_loder(data_features, data_labels, batch_size):
    # 设置features
    x_columns = data_features.columns  # 取训练features的全部列名,
    x_array = data_features[x_columns].values  # x_array即为本轮循环中,模型的train_features
    # x_array.shape = (123764, 122), x_array.class=ndarray

    # 重塑features.shape为(-1, c_in=1, seq=122),使其符合网络结构输入
    x_features = np.reshape(x_array, (x_array.shape[0], 1, x_array.shape[1]))
    # shape=(123764, 1, 122)

    # 设置Class
    dummies = pd.get_dummies(data_labels)  # 将训练集的Class独热编码,shape = [123764 rows x 5 columns]
    y_labels = dummies.values
    # train_labels = Index(['DOS', 'Probe', 'R2L', 'U2R', 'normal'], dtype='object')
    # train_labels.shape = (-1,5)

    # 创建数据集和数据加载器
    dataset = CustomDataset(x_features, y_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


# 定义训练函数
def train_CNN_LSTM_with_cross_validation(net, kfold, data, batch_size, num_epochs, lr, device, w_decay=1e-5):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)

    oos_pred = []  # 用于存储每个验证集的准确率
    train_data = data.copy()
    train_labels = train_data.pop('Class')
    train_features = train_data  # 要求数据为dataframe形式

    for train_index, test_index in kfold.split(train_features, train_labels):
        train_X, test_X = train_features.iloc[train_index], train_features.iloc[test_index]
        train_y, test_y = train_labels.iloc[train_index], train_labels.iloc[test_index]

        # 确定本轮循环中的data_loder
        loop_train_loder = loop_data_loder(train_X, train_y, batch_size)
        loop_test_loder = loop_data_loder(test_X, test_y, batch_size)

        net.apply(init_weights)  # 初始化参数
        print('Training on', device)
        net.to(device)  # 模型传入device
        # 设置优化器和损失函数
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

    # 分层K折
    '''
    Kfold = KFold(n_splits=splits, shuffle=False)
    KFold（K - 折叠）
    在KFold交叉验证中，原始数据集被分为K个子集。
    每次，其中的一个子集被用作测试集，而其余的K - 1
    个子集合并后被用作训练集。
    这个过程重复进行K次，每次选择不同的子集作为测试集。
    KFold不保证每个折叠的类分布与完整数据集中的分布相同。
    stratKfold = StratifiedKFold(n_splits=splits, shuffle=False)
    Stratified - KFold（分层K - 折叠）
    Stratified - KFold是KFold的变体，它会返回分层的折叠：每个折叠中的标签分布都尽可能地与完整数据集中的标签分布相匹配。
    这种方法特别适用于类分布不均衡的情况，确保每个折叠都有代表性的类比例。
    就像KFold一样，每个折叠轮流被用作测试集，其他折叠用作训练集。
    '''
    num_folds = 6
    k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)  # 随机种子固定,保证每次生成的都一样

    # 加载数据
    combined_data = load_combined_data()
    combined_data_labels = combined_data['Class']  # 先分离Class
    combined_data_features = combined_data.drop(columns='Class', inplace=False)

    # myModel = CNNBiLSTMModel()
    # train_CNN_LSTM_with_cross_validation(myModel, k_fold, data, num_epochs=numb_epochs, lr=learning_rate, device=try_device(), w_decay=weight_decay)

    # 查看随机划分后的数据样式
    for train_index, test_index in k_fold.split(combined_data_features, combined_data_labels):
        train_X, test_X = combined_data_features.iloc[train_index], combined_data_features.iloc[test_index]
        train_y, test_y = combined_data_labels.iloc[train_index], combined_data_labels.iloc[test_index]

        # 设置本轮loop的训练数据
        train_features, train_labels = load_loop_data(train_X, train_y)

        # 设置本轮loop的测试数据
        test_features, test_labels = load_loop_data(test_X, test_y)
