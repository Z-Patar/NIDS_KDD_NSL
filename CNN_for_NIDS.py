#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNNet(nn.Module):
    def __init__(self):
        # 假设输入图像尺寸为 Cx11x11 (C是输入通道数，对于灰度图C=1)
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出尺寸: 11x11
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出尺寸: 11x11
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)  # 输出尺寸: 5x5 -> 2x2
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)  # 64个特征图，每个4x4大小
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 5)  # 输出层大小为5
        self.dropout = nn.Dropout(0.5)

        # 初始化权重
        self._initialize_weights()
    # todo 尝试使用该函数来查看每层的输出的shape
    # for layer in net:
    #     X = layer(X)
    # print(layer.__class__.__name__.,'output shape: ', X.shape)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)  # 重新塑形为一维向量  # todo 尝试nn.Flatten() ?
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


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
    model = CNNNet()
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

            outputs = model(inputs)# todo 存在问题
            loss = criterion(outputs, labels.max(1)[1])
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/10], Iter [{i}/{train_feature.size(0)}] Loss: {loss.item():.4f}')

    print('Finished Training')


if __name__ == '__main__':
    main()
