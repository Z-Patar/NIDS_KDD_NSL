#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 输出尺寸: 11x11
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # 输出尺寸: 11x11
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸: 5x5
        # 计算卷积层输出后的尺寸，用于全连接层的输入
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)  # 64个特征图，每个5x5大小
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 5)  # 输出层大小为5
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)  # 重新塑形为一维向量
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def load_data(file_path):
    df = pd.read_csv(file_path, header=0)
    df = df.drop(columns=['is_host_login'])
    features = df.iloc[:, :-5].values
    labels = df.iloc[:, -5:].values
    # 重塑特征为11x11方阵
    features = features.reshape(-1, 1, 11, 11)
    return features, labels


def main():
    file_path = 'Data_encoded/Train_encoded.csv'  # 数据集文件路径

    # 加载数据
    feature, label = load_data(file_path)

    # 转换为 PyTorch tensors
    feature_tensor = torch.Tensor(feature)
    label_tensor = torch.Tensor(label)

    # 初始化模型、损失函数和优化器
    model = CNN_Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 划分数据集为训练集和测试集
    # 这里需要根据您的情况来划分，以下仅为示例
    train_feature = feature_tensor[:100000]  # 前100000个样本用于训练
    train_label = label_tensor[:100000]
    test_feature = feature_tensor[100000:]  # 剩余的样本用于测试
    test_label = label_tensor[100000:]

    # 训练模型
    batch_size = 32
    for epoch in range(10):  # 运行10个训练周期
        for i in range(0, train_feature.size(0), batch_size):
            inputs = train_feature[i:i + batch_size]
            labels = train_label[i:i + batch_size]

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.max(1)[1])
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/10], Iter [{i}/{train_feature.size(0)}] Loss: {loss.item():.4f}')

    print('Finished Training')


if __name__ == '__main__':
    main()
