#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

# python3.10
# -*- coding:utf-8 -*-
# @Time    :2024/5/30 下午3:58
# @Author  :Z_Patar

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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(122, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


# todo 尝试使用MLP实现分类

if __name__ == "__main__":
    run_code = 0
