#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QFileDialog, QVBoxLayout, QWidget

"""定义模型类"""


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


# 加载保存的模型参数
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('深度学习模型预测')
        self.setGeometry(200, 200, 400, 200)

        # 创建部件
        self.file_path_input = QLineEdit()
        self.select_file_button = QPushButton('选择文件')
        self.predict_button = QPushButton('预测')
        self.result_label = QLabel('预测结果：')

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.file_path_input)
        layout.addWidget(self.select_file_button)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 信号与槽
        self.select_file_button.clicked.connect(self.select_file)
        self.predict_button.clicked.connect(self.predict)

        # 加载模型
        self.model = CNNBiLSTMModel()
        # 调试查看模型各层输出是否正常
        if 1:
            # [调试用]查看模型每一层的输出
            in_tensor = torch.randn(1, 1, 122)  # batch_size=1, in_channels=1, sequence_length=122
            print_layer_shapes(self.model, in_tensor)
            # [调试结果]输出正常
        self.model_path = 'CNN_BiLSTM_state.pth'  # 替换为你保存的模型参数文件路径
        load_model(self.model, self.model_path)

        # 定义标签映射字典
        self.label_mapping = {
            0: "Normal",
            1: "DoS",
            2: "R2L",
            3: "Probe",
            4: "U2R"
        }

    # 选择文件
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择数据文件', '',
                                                   'Data Files (*.csv *.txt)')  # 可以根据实际文件类型进行修改
        self.file_path_input.setText(file_path)

    # 预测
    def predict(self):
        file_path = self.file_path_input.text()
        try:
            data = pd.read_csv(file_path)
            print(f"Loaded data: {data.head()}")  # 调试信息：打印前几行数据
            input_data = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)
            print(f"Input data shape: {input_data.shape}")  # 调试信息：打印输入数据形状
            if input_data.shape[-1] != 122:
                raise ValueError("输入数据的长度必须为122")
            with torch.no_grad():
                self.model.eval()
                output = self.model(input_data)
                predicted_class = torch.argmax(output, dim=1).item()

            predicted_label = self.label_mapping.get(predicted_class, "未知标签")
            self.result_label.setText('预测结果：{}'.format(predicted_label))
        except ValueError as ve:
            self.result_label.setText('<font color="red">输入数据错误：{}</font>'.format(str(ve)))
        except Exception as e:
            self.result_label.setText('<font color="red">预测出现异常：{}</font>'.format(str(e)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
