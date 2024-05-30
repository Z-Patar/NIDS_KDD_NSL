#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# 假设你已经有一个保存的模型文件，例如 'model.pth'
MODEL_PATH = 'CNN_BiLSTM_all.pth'
MODEL_state_PATH = 'CNN_BiLSTM_state.pth'
INPUT_CSV = 'Data_encoded/LSTM_data/combined_data_processed.csv'
OUTPUT_CSV = 'correct_predictions.csv'


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

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
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=122, padding=61)  # 保持输出尺寸不变
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


# 加载模型
model = CNNBiLSTMModel()
model.load_state_dict(torch.load(MODEL_state_PATH))
# 调试查看模型各层输出是否正常
if 1:
    # [调试用]查看模型每一层的输出
    in_tensor = torch.randn(1, 1, 122)  # batch_size=1, in_channels=1, sequence_length=122
    print_layer_shapes(model, in_tensor)
    # [调试结果]输出正常
model.eval()

# 打印特定层的参数
for name, param in model.named_parameters():
    if 'fc1' in name:
        print(f"Parameter name: {name}")
        print(param.data[:100])  # 打印前5个参数值


# 定义数据加载器
def loop_data_loder(data_features, data_labels, batch_size=1):
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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader


data = pd.read_csv(INPUT_CSV)
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]
print(f'features.shape: {features.shape}     label.shape: {labels.shape}')
dataloader = loop_data_loder(features, labels)

# 用于存储每种标签前三次预测正确的数据项
correct_predictions = {label: [] for label in range(5)}

# 记录每个label的预测正确次数
correct_count = {label: 0 for label in range(5)}

# 预测数据
with torch.no_grad():
    for features, label in dataloader:
        output = model(features)
        _, predicted = torch.max(output, 1)
        if predicted.item() == label.item():
            if len(correct_predictions[label.item()]) < 3:
                correct_predictions[label.item()].append((features.numpy().flatten(), label.item()))

# 调试输出，检查数据结构
for label in correct_predictions:
    for item in correct_predictions[label]:
        print(f"Label: {label}, Item: {item}")

# 将所有正确预测的数据项合并到一个列表中
all_correct_predictions = []
for label in correct_predictions:
    for item in correct_predictions[label]:
        features, lbl = item
        all_correct_predictions.append(list(features) + [lbl])

# 保存前三次预测正确的数据项到新的CSV文件
if all_correct_predictions:
    columns = [f'feature_{i}' for i in range(len(all_correct_predictions[0]) - 1)] + ['label']
    correct_predictions_df = pd.DataFrame(all_correct_predictions, columns=columns)
    correct_predictions_df.to_csv(OUTPUT_CSV, index=False)
    print(f"每种标签前三次预测正确的数据项已保存到 {OUTPUT_CSV}")
else:
    print("没有找到任何正确预测的数据项。")
