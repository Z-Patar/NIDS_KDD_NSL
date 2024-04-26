#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import csv

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 选择第一个CUDA设备
    device = torch.device("cuda:0")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")


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
        return self.features[idx], self.labels[idx]


# 定义了一个简单的神经网络模型Net，这个模型只有一个全连接层，输入大小为41，输出大小为2，激活函数为softmax。
# 此处定义了一个名为Net的类，它是一个神经网络模型类，并且继承自nn.Module类。
# 在PyTorch中，定义神经网络模型通常需要继承自nn.Module类，并实现__init__和forward方法。
class Net(nn.Module):
    # 构造函数。
    # 首先调用了父类nn.Module的构造函数，
    # 然后定义了一个全连接层(nn.Linear)，该层将输入维度为122，输出维度为5。
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(122, 60)
        self.fc2 = nn.Linear(60, 5)

    # 定义数据在模型中的前向传播过程。
    # 输入x经过全连接层self.fc，
    # 然后通过torch.softmax函数进行softmax操作，
    # 最终返回softmax后的输出
    def forward(self, x):
        # 先通过全连接层self.fc对输入x进行线性变换，然后通过torch.softmax函数对结果进行softmax操作
        # dim = 1参数表示在第1维度（即每行）上进行softmax操作，确保输出的每行元素和为1，可以用于多分类任务。
        # x = torch.softmax(self.fc(x), dim=1)   # 全连接层后接 softmax 操作

        x = torch.relu(self.fc1(x))     # 第一层全连接层后接 ReLU 操作
        x = torch.softmax(self.fc2(x), dim=1)   # 第二层全连接层后接 softmax 操作
        return x


def load_data(file_path):
    features = []
    labels = []
    data = pd.read_csv(file_path, header=0)  # header=0表示第1行为表头
    features = data.iloc[:, :-5:].values
    labels = data.iloc[:, -5:].values
    return features, labels


# def load_data(data_file):
#     features=[]
#     labels=[]
#     file_path = data_file
#     with (open(file_path,'r')) as data_from:
#         csv_reader=csv.reader(data_from)
#         for i in csv_reader:
#             features.append(i[:41])
#             label_list=[0 for i in range(23)]
#             label_list[i[41]]=1
#             labels.append(label_list)
#     return features, labels

# 在main函数中，首先加载训练集和测试集，并分割成训练集和测试集。
# 然后，创建数据加载器，神经网络模型，损失函数和优化器。
# 最后，进行训练和评估，每50个epoch打印一次训练损失和测试集上的准确率
def test_NN_main(data):
    # 设置训练集和测试集
    features, labels = load_data(data)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)
    # 创建训练数据集和测试数据集的实例，使用前面定义的CustomDataset类来加载数据。
    train_data = CustomDataset(features_train, labels_train)
    test_data = CustomDataset(features_test, labels_test)
    # 创建训练数据和测试数据的数据加载器，用于批量加载数据进行训练和测试。
    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    # 创建了一个神经网络模型实例model，定义了交叉熵损失函数criterion和随机梯度下降优化器optimizer。
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(1001):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs.float())
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 每50个epoch，对测试数据进行测试，计算模型的准确率并输出。
        if epoch % 50 == 0:
            correct = 0
            total = 0
            with torch.no_grad():   # 使用torch.no_grad()来确保在测试阶段不进行梯度计算。测试时不需要梯度计算
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images.float())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == torch.max(labels, 1)[1]).sum().item()
            print('Epoch: %d, loss: %.3f, accuracy: %.2f %%' % (epoch, running_loss / 50, 100 * correct / total))
            running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    data_file = 'Data_encoded/Train_encoded.csv'
    test_NN_main(data_file)
    # print(torch.cuda.is_available())
    # # 获取可用的CUDA设备数量
    # device_count = torch.cuda.device_count()
    # print(f"Number of CUDA devices available: {device_count}")
    #
    # # 遍历并打印每个CUDA设备的属性
    # for i in range(device_count):
    #     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # data_path = 'Processed_Train_20Percent.csv'
    # features_train, labels_train = load_data(data_path)
    # print(features_train.shape)
    # print(labels_train.shape)

    # 查看数据集
    # df = pd.read_csv(data_path)
    # print(df.describe())
