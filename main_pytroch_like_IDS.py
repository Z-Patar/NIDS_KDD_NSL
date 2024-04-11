import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import csv


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
    # 然后定义了一个全连接层(nn.Linear)，该层将输入维度为41，输出维度为2。
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(41, 2)

    # 定义数据在模型中的前向传播过程。
    # 输入x经过全连接层self.fc，
    # 然后通过torch.softmax函数进行softmax操作，
    # 最终返回softmax后的输出
    def forward(self, x):
        # 先通过全连接层self.fc对输入x进行线性变换，然后通过torch.softmax函数对结果进行softmax操作
        # dim = 1参数表示在第1维度（即每行）上进行softmax操作，确保输出的每行元素和为1，可以用于多分类任务。
        x = torch.softmax(self.fc(x), dim=1)
        return x


def load_data(file_path):
    features = []
    labels = []
    data = pd.read_csv(file_path, header=0)  # header=0表示第1行为表头
    features = data.iloc[:, :-5:].values
    labels = data.iloc[:, -5:].values
    return features, labels


# 在main函数中，首先加载训练集和测试集，并分割成训练集和测试集。
# 然后，创建数据加载器，神经网络模型，损失函数和优化器。
# 最后，进行训练和评估，每50个epoch打印一次训练损失和测试集上的准确率
def test_CNN_main():
    features, labels = load_data()
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)

    train_data = CustomDataset(features_train, labels_train)
    test_data = CustomDataset(features_test, labels_test)

    trainloader = DataLoader(train_data, batch_size=1000, shuffle=True)
    testloader = DataLoader(test_data, batch_size=1000, shuffle=False)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.5)

    for epoch in range(1001):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs.float())
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 50 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = net(images.float())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == torch.max(labels, 1)[1]).sum().item()
            print('Epoch: %d, loss: %.3f, accuracy: %.2f %%' % (epoch, running_loss / 50, 100 * correct / total))
            running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    # test_CNN_main()
    data_path = 'Processed_Train_20Percent.csv'
    features_train, labels_train = load_data(data_path)
    print(features_train.shape)
    print(labels_train.shape)

    # 查看数据集
    # df = pd.read_csv(data_path)
    # print(df.describe())
