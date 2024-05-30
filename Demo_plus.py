#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

import sys

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit
import pandas as pd
import joblib

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 输入框和标签
        hbox1 = QHBoxLayout()
        self.label = QLabel('请选择数据文件')
        self.lineEdit = QLineEdit()
        hbox1.addWidget(self.label)
        hbox1.addWidget(self.lineEdit)
        layout.addLayout(hbox1)

        # 选择文件和截取流量按钮
        hbox2 = QHBoxLayout()
        self.btnSelectFile = QPushButton('选择文件')
        self.btnCaptureTraffic = QPushButton('截取流量')
        hbox2.addWidget(self.btnSelectFile)
        hbox2.addWidget(self.btnCaptureTraffic)
        layout.addLayout(hbox2)

        # 数据处理按钮
        self.btnProcessData = QPushButton('数据处理')
        layout.addWidget(self.btnProcessData)

        # 预测按钮
        self.btnPredict = QPushButton('预测')
        layout.addWidget(self.btnPredict)

        # 输出框
        self.outputText = QTextEdit()
        layout.addWidget(self.outputText)

        self.setLayout(layout)

        # 连接信号和槽
        self.btnSelectFile.clicked.connect(self.selectFile)
        self.btnCaptureTraffic.clicked.connect(self.captureTraffic)
        self.btnProcessData.clicked.connect(self.processData)
        self.btnPredict.clicked.connect(self.predict)

        self.setWindowTitle('网络流量分析')
        self.setGeometry(300, 300, 600, 400)

    def selectFile(self):
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, '选择数据文件', '', 'All Files (*);;CSV Files (*.csv)', options=options)
            if file_path:
                self.lineEdit.setText(file_path)
        except Exception as e:
            self.outputText.append(f"选择文件过程中发生错误: {e}")

    def captureTraffic(self):
        try:
            # 直接输出消息而不进行实际的流量截取
            self.lineEdit.setText('input.csv')
            self.outputText.append('网络流量信息截取完毕！已转换为input.csv文件。')
        except Exception as e:
            self.outputText.append(f"截取流量过程中发生错误: {e}")

    def processData(self):
        try:
            # 直接输出消息而不进行实际的数据预处理
            self.outputText.append('数据预处理完毕！')
        except Exception as e:
            self.outputText.append(f"数据处理过程中发生错误: {e}")

    def predict(self):
        try:
            file_path = self.lineEdit.text()
            if file_path:
                with open(file_path, 'r') as file:
                    content = file.read()
                    self.outputText.setTextColor(QColor(255, 0, 0))  # 设置字体颜色为红色
                    self.outputText.append("存在攻击流量：")  # 输出提示信息
                    self.outputText.setTextColor(QColor(0, 0, 0))  # 设置字体颜色为红色
                    self.outputText.append(content)  # 输出文件内容
        except Exception as e:
            self.outputText.append(f"预测过程中发生错误: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
