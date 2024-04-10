import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import csv

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(41, 2)

    def forward(self, x):
        x = torch.softmax(self.fc(x), dim=1)
        return x

def load_data():
    features=[]
    labels=[]
    file_path ='/home/peter/Desktop/pycharm/ids-kdd99/kddcup.data_10_percent_corrected_handled2.cvs'
    with (open(file_path,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        for i in csv_reader:
            features.append(i[:41])
            label_list=[0 for i in range(23)]
            label_list[i[41]]=1
            labels.append(label_list)
    return features, labels

def main():
    features, labels = load_data()
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

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

if __name__  == '__main__':
    main()
