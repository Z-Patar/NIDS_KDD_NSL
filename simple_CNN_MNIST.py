import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from pandas import read_csv, set_option, DataFrame


import numpy as np


# 在这个例子中，我们假设你的数据集已经被处理为一个形状为 (batch_size, num_channels, height, width) 的张量，
# 其中 num_channels 是你的特征数量，height 和 width 是你的数据的维度。
# 这个网络是在MNIST数据集上训练的，你可能需要根据你的数据集调整网络的结构（例如卷积层的数量，全连接层的节点数等）和参数（例如学习率，优化器等）。
# 如果你的数据集的类别数不是10，你需要改变最后一个全连接层的输出节点数。

# 定义CNN网络结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层，输入通道数为1，输出通道数为6，卷积核大小为5x5。
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input image channel, 6 output channels, 5x5 square convolution
        # 池化层，使用最大池化，池化窗口大小为2x2。
        self.pool = nn.MaxPool2d(2, 2)  # maxpooling
        # 第二个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5。
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input image channel, 16 output channels, 5x5 square convolution
        # 三个全连接层，分别将卷积层输出展平后连接到全连接层。
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 定义了数据在网络中的前向传播过程
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 数据x首先经过第一个卷积层self.conv1，然后经过ReLU激活函数和池化层self.pool。
        x = self.pool(F.relu(self.conv2(x)))  # 数据经过第二个卷积层self.conv2，再次经过ReLU激活函数和池化层。
        # 数据展平
        x = x.view(-1, 16 * 4 * 4)  # .view() is similar to .reshape()
        # 数据通过三个全连接层，最终输出网络的预测结果
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(dataset_path, batch_size=1024, shuffle=True):
    data = read_csv(dataset_path)
    labels = torch.tensor(data.iloc[:, 0].values).long()
    features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float).view(-1, 1, 28, 28)  # 将数据reshape成图片格式
    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


# 训练函数
def train_model(train_loader, net, criterion, optimizer, epochs=2, print_freq=1):
    # 将数据转移到CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        total_samples = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        if (epoch + 1) % print_freq == 0:
            print('[%d] loss: %.3f' % (epoch + 1, epoch_loss))

    print('Finished Training')


def test_model(test_loader, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    net.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    precision = TP / (TP + FP)
    accuracy = np.sum(cm) / np.sum(cm)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # 添加表头
    cm_df = DataFrame(cm,
                      index=[f"True {i}" for i in range(cm.shape[0])],
                      columns=[f"Predicted {i}" for i in range(cm.shape[1])]
                      )

    return cm_df, precision, accuracy, recall, f1_score


# 主函数
def main():
    # 加载数据集，初始化网络、损失函数和优化器
    train_dataset_path = 'MNIST/mnist_train.csv'
    test_dataset_path = 'MNIST/mnist_test.csv'
    train_loader = load_data(train_dataset_path, batch_size=1024, shuffle=True)
    test_loader = load_data(test_dataset_path, batch_size=1024, shuffle=False)
    net = CNN()
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降（SGD）优化器，用于更新神经网络的参数以最小化损失函数。

    # 训练模型
    train_model(train_loader, net, criterion, optimizer, epochs=100, print_freq=10)

    # 设置pandas显示选项，使其不缩略显示
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)
    set_option('display.max_colwidth', None)

    # 测试模型并生成混淆矩阵, 计算精确率，准确率，召回率，F1分数
    confusion_matrix, precision, accuracy, recall, f1_score = test_model(test_loader, net)

    # 打印混淆矩阵
    print("Confusion_Matrix: \n", confusion_matrix)
    # 打印精确率，准确率，召回率，F1分数
    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")


if __name__ == '__main__':
    main()
