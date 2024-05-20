#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.
import time
import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import optuna
import optuna.visualization as vis
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import seaborn as sns
from collections import OrderedDict
import plotly
import kaleido
import warnings

warnings.filterwarnings("ignore")


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


# 类定义
# 自定义数据集类
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
        # return self.features[idx], self.labels[idx]
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


# 定义BiLSTMLayer层模型
class BiLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # LSTM的输出包括所有隐藏状态、最后的隐藏状态和最后的细胞状态
        output, _ = self.lstm(x)
        # 只返回输出张量，不返回隐藏状态和细胞状态
        return output[:, -1, :]  # 只返回最后一个时间步的输出


# 定义网络结构
class CNNBiLSTMModel(nn.Module):
    def __init__(self):
        super(CNNBiLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=122, padding='same')  # 保持输出尺寸不变
        self.maxpool1 = nn.MaxPool1d(kernel_size=5)
        self.batchnorm1 = nn.BatchNorm1d(64)
        # out.shape=(batch=32, channel = 64, seq=24(122池化后的数字))
        # channel = 64,此处channel就是embedding，即input_size
        self.bilstm1 = BiLSTMLayer(input_dim=64, hidden_dim=64)  # hidden_size即为上一层的输出channel

        # 此处需要将(128, ) reshape为(1,128), 因为要沿着128的方向做池化,
        self.maxpool1d2 = nn.MaxPool1d(kernel_size=5)
        self.batchnorm2 = nn.BatchNorm1d(1)

        # 第二个BiLSTM
        # input=(input_size=1, hidden_size=128, 其他默认) ,seq=25(根据上一层的输出判断的)
        self.bilstm2 = BiLSTMLayer(input_dim=1, hidden_dim=128)  # BiLSTM只取了最后一个时间步的输出
        # out.shape = (batch=32, 1(啥意思暂不明白), 256(就是128池化后的数字))

        self.dropout = nn.Dropout(0.5)  # 将上一层随机丢弃一半传入下层
        self.fc = nn.Linear(256, 5)
        # self.softmax = nn.Softmax(dim=1)

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

        x = self.maxpool1d2(x)  # shape=(32, 1, 25)
        x = self.batchnorm2(x)
        # out.shape=(32, 1, 25)

        x = x.permute(0, 2, 1)  # 重排维度以适配LSTM输入
        # out.shape=(32, 25, 1)

        # 第二个BiLSTM
        x = self.bilstm2(x)
        # out.shape=(batch=32, 256)

        x = self.dropout(x)
        x = self.fc(x)
        return x


# 定义数据加载器
def loop_data_loder(data_features, data_labels, batch_size):
    # 设置features
    x_columns = data_features.columns  # 取训练features的全部列名,
    x_array = data_features[x_columns].values  # x_array即为本轮循环中,模型的train_features
    # x_array.shape = (-1, 122), x_array.class=ndarray

    # 重塑features.shape为(-1, c_in=1, seq=122),使其符合网络结构输入
    x_features = np.reshape(x_array, (x_array.shape[0], 1, x_array.shape[1]))
    # shape=(-1, 1, 122)

    # 设置Class
    # 如果data_labels已经是一个包含类别名称的Series或者列，你可以这样获取类别索引:
    # 假设data_labels是类别名称的Series，你需要将这些名称映射到索引
    # 首先获取类别名称到索引的映射字典
    label_to_idx = {label: idx for idx, label in enumerate(data_labels.unique())}
    # 然后将类别名称转换为索引
    y_labels = data_labels.replace(label_to_idx).values

    # 创建数据集和数据加载器
    dataset = CustomDataset(x_features, y_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


# 模型参数初始化函数
def init_weights(m):
    if isinstance(m, nn.Conv1d):  # 一维卷积层
        nn.init.xavier_uniform_(m.weight)  # weight用了Xavier均匀初始化
        if m.bias is not None:  # bias存在时，初始化为0
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):  # 对于LSTM层
        for name, param in m.named_parameters():
            if 'weight_ih' in name:  # 输入到隐藏层的权重weight_ih
                nn.init.xavier_uniform_(param.data)  # 使用了Xavier均匀初始化

            elif 'weight_hh' in name:  # 隐藏层到隐藏层的权重weight_hh，
                # 随机正态分布初始化后的结果优于正交初始化
                nn.init.xavier_uniform_(param.data)  # 随机正态分布初始化

            elif 'bias' in name:  # 对于LSTM层的偏置项，
                param.data.fill_(0)  # 先所有偏置项初始化为0
                # 偏置全为0的结果和遗忘门偏置为1的结果差不多
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)  # 然后将遗忘门偏置初始化为1，这在某些情况下可以帮助模型记忆顺序，提高学习能力。
    elif isinstance(m, nn.Linear):  # 线性层使用了Xavier均匀初始化
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

    # 第一张图：每个fold的train_accuracy和模型总体的train_accuracy随epoch变化的曲线


def objective(trial):
    # QMCSampler法探索参数空间
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
    weight_decay = 1e-6
    beat1 = trial.suggest_float('beat1', 0.8, 0.999,)

    # 贝叶斯法定义超参数搜索空间
    # lr在0.09+附近，0.065附近
    # learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01, log=True)
    # weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-5, log=True)
    # beat1 = trial.suggest_float('beat1', 0.8, 0.999,)
    # eps = trial.suggest_float('eps', 1e-9, 1e-7)
    # momentum = trial.suggest_float('momentum', 0.94, 0.9999)
    batch_size = 128
    numb_epochs = 30
    """
    去掉了学习率衰减
    53次测试中，'learning_rate'(0.001, 0.1), 'momentum'(0.9, 0.99),最优结果如下
    Trial 38 finished with value: 0.4619097254100284 and parameters: {'learning_rate': 0.0024131048184518476, 'momentum': 0.9799346347439084}
    """

    """初始化epoch早停逻辑参数"""
    patience = 15
    # best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0

    data_file = '../Data_encoded/LSTM_data/combined_data_processed.csv'
    data = pd.read_csv(data_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_sub, y_train, y_sub = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_data_f = pd.concat([X_sub, y_sub], axis=1)
    print(f'\nTrain data is 20% subdata of {data_file}, \n'
          f'Train data shape: {train_data_f.shape}')

    # 设定超参数
    device = try_device()

    # 分层K折交叉验证
    num_folds = 6
    k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)  # 随机种子固定,保证每次生成的都一样

    # 实例化模型
    model = CNNBiLSTMModel()
    # 初始化模型参数
    model.apply(init_weights)

    # 模型传入device
    model.to(device)

    # 设置优化器和损失函数
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate, weight_decay=weight_decay,
                           betas=(beat1, 0.999))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, factor=0.8, patience=2, verbose=True)
    loss_fn = nn.CrossEntropyLoss()

    # 训练数据加载
    train_data = train_data_f.copy()  # 不改变源数据
    # 下面两项操作都不会改变train_data数据,在模型中不需要改变
    labels = train_data['Class']
    features = train_data.drop(['Class'], axis=1, inplace=False)

    # 初始化存储结构，每个fold的每个epoch的数据都会被存储
    fold_metrics = {
        # train
        'train_loss': [[] for _ in range(numb_epochs)],
        'train_accuracy': [[] for _ in range(numb_epochs)],  # accuracy两种方法没用区别

        # val
        'val_loss': [[] for _ in range(numb_epochs)],
        'val_accuracy': [[] for _ in range(numb_epochs)],  # accuracy两种方法没用区别
    }
    start_time = time.time()
    # 完整训练
    for epoch in range(numb_epochs):
        print(f'Epoch {epoch + 1}/{numb_epochs}')

        # K折交叉验证
        for (train_index, val_index) in k_fold.split(features, labels):

            # 根据K折设置训练集和验证集
            train_data, val_data = features.iloc[train_index], features.iloc[val_index]
            train_labels, val_labels = labels.iloc[train_index], labels.iloc[val_index]

            # 设置Data_Loder
            train_loader = loop_data_loder(train_data, train_labels, batch_size)
            val_loader = loop_data_loder(val_data, val_labels, batch_size)

            # 初始化fold统计数据
            fold_train_loss = 0.0
            fold_train_correct = 0
            fold_train_total = 0
            fold_val_loss = 0.0
            fold_val_correct = 0
            fold_val_total = 0
            fold_train_preds = []
            fold_train_targets = []
            fold_val_preds = []
            fold_val_targets = []

            '''在当前fold训练集上训练模型'''
            model.train()
            for train_batch, train_label_batch in train_loader:
                optimizer.zero_grad()
                train_batch, train_label_batch = train_batch.to(device), train_label_batch.to(device)

                outputs = model(train_batch)
                loss = loss_fn(outputs, train_label_batch)
                loss.backward()
                optimizer.step()

                fold_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                fold_train_total += train_label_batch.size(0)
                fold_train_correct += (predicted == train_label_batch).sum().item()
                fold_train_preds.extend(predicted.view(-1).cpu().numpy())
                fold_train_targets.extend(train_label_batch.cpu().numpy())

            # 计算训练集上的统计数据
            train_loss = fold_train_loss / len(train_loader)
            train_accuracy = fold_train_correct / fold_train_total

            '''在当前fold验证集上评估模型'''
            model.eval()
            with torch.no_grad():
                for val_batch, val_label_batch in val_loader:
                    val_batch, val_label_batch = val_batch.to(device), val_label_batch.to(device)
                    outputs = model(val_batch)
                    loss = loss_fn(outputs, val_label_batch)

                    fold_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    fold_val_total += val_label_batch.size(0)
                    fold_val_correct += (predicted == val_label_batch).sum().item()
                    fold_val_preds.extend(predicted.view(-1).cpu().numpy())
                    fold_val_targets.extend(val_label_batch.cpu().numpy())

            # 计算验证集上的统计数据
            val_loss = fold_val_loss / len(val_loader)
            val_accuracy = fold_val_correct / fold_val_total

            # 将当前fold的统计数据添加到fold_metrics中
            for i in range(1):  # 循环没用，单纯想折叠大段代码
                # train
                fold_metrics['train_loss'][epoch].append(train_loss)
                fold_metrics['train_accuracy'][epoch].append(train_accuracy)

                # val
                fold_metrics['val_loss'][epoch].append(val_loss)
                fold_metrics['val_accuracy'][epoch].append(val_accuracy)

        # 使用平均val_loss进行学习率衰减
        # val_loss_avg = sum(fold_metrics['val_loss'][epoch]) / num_folds
        # scheduler.step(val_loss_avg)

        # 打印epoch的平均统计数据
        # 循环没用，单纯想折叠大段代码
        for i in range(1):
            print(f'Train Loss: {np.mean(fold_metrics["train_loss"][epoch]):.4f}, '
                  f'Val Loss: {np.mean(fold_metrics["val_loss"][epoch]):.4f}, '
                  f'Val Acc: {np.mean(fold_metrics["val_accuracy"][epoch]):.4f}, ')

        # 查看当前epoch的平均loss，根据val_loss决定是否epoch早停
        val_loss = np.mean(fold_metrics["val_loss"][epoch])  # mean为平均
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        # 早停判定
        if patience_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch + 1}')
            break

    end_time = time.time()
    print(f'Total training time: {(end_time - start_time):.2f} s  ')

    # 目标指标为平均val_loss
    val_loss_avg = np.mean(fold_metrics["val_loss"][-1])  # mean为平均
    return val_loss_avg


if __name__ == '__main__':
    """
    MedianPruner用于早停，
    它会在每个间隔步骤(interval_steps)检查性能是否低于所有试验的中位数。如果是，则会停止当前试验。
    n_startup_trials参数设置了在开始应用早停之前要完成的最小试验次数，
    n_warmup_steps则设置了在开始考虑早停之前要运行的最小步骤数。
    """
    # 创建一个早停的pruner对象
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=10,
        reduction_factor=2,
        min_early_stopping_rate=2)

    # 用贝叶斯优化自动调参
    # study = optuna.create_study(direction='maximize', pruner=pruner)

    # 用Quasi-Random-Search自动调参
    study = optuna.create_study(
        sampler=optuna.samplers.QMCSampler(),
        direction='minimize',
        pruner=pruner)
    study.optimize(objective, n_trials=100)

    # 输出最优的超参数
    print(study.best_params)

    # 最佳超参数
    best_params = study.best_params
    best_value = study.best_value

    # 保存最佳超参数到txt文件
    with open('best_params.txt', 'w') as f:
        f.write(f"Best trial:\n")
        f.write(f"  Value: {best_value}\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")

    # 将所有试验的详细结果导出到CSV文件
    df = study.trials_dataframe()
    df.to_csv('optuna_results.csv', index=False)

    # 绘制参数重要性
    fig_param_importance = vis.plot_param_importances(study)
    fig_param_importance.show()

    # 绘制参数与目标值的关系
    fig_param_rel = vis.plot_slice(study)
    fig_param_rel.show()

    # 绘制优化历史图
    fig_param_optim_history = vis.plot_optimization_history(study)
    fig_param_optim_history.show()

    # 绘制各参数间的关系
    # 学习率与weight—decay
    fig_param_contour_lr_wd = vis.plot_contour(study, params=['learning_rate', 'weight_decay']) #
    fig_param_contour_lr_wd.show()
    # 学习率与beat1
    fig_param_contour_lr_beat1 = vis.plot_contour(study, params=['learning_rate', 'beat1'])
    fig_param_contour_lr_beat1.show()
    # 学习率与eps
    fig_param_contour_lr_eps = vis.plot_contour(study, params=['learning_rate', 'eps'])
    fig_param_contour_lr_eps.show()
