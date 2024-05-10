#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

# python3.10
# coding:utf-8

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess(data_file):
    field_name_file = 'KDD_NSL/field_name_file.csv'
    attack_type_file = 'KDD_NSL/attack_type_file.csv'

    # 定义dataframe ，并定义column name，方便索引
    column_name_file = pd.read_csv(field_name_file, header=None, names=['name', 'data_type'])
    column_names = column_name_file['name'].tolist()

    # 读取数据，同时加表头
    data = pd.read_csv(data_file, header=None, names=column_names)

    # 删除第43列的'难度等级'
    data = data.drop(columns='difficult_level')

    # 从CSV文件中读取映射,定义22种攻击小类标签对应的攻击类型
    attack_type_df = pd.read_csv(attack_type_file, header=None, names=['name', 'attack_type'])
    # 定义5大类和22小类的映射字典，方便替代
    mapping = attack_type_df.set_index('name').to_dict()['attack_type']
    # 替换训练集label中22小类为5大类标签
    data['label'] = data['label'].replace(mapping)
    # 将label改名为Class
    data.rename(columns={'label': 'Class'}, inplace=True)

    return data


def one_hot(data, encode_cols):
    df = pd.DataFrame()  # 创建一个空的dataframe，用来存储编码后的数据
    # 编码的列是连续的，所以直接将两段的数据分离出来，然后将label分离，对encode_cols编码后拼接回去
    before_encode_cols = data.columns[0]
    df = pd.concat([df, data[before_encode_cols]], axis=1)  # 先将编码列前面的列数据插入df

    end_encode_cols = data.columns[len(encode_cols)+1:]

    for col in encode_cols:
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)  # 对选定数据列独热编码
        df = pd.concat([df, dummies], axis=1)   # 将编码后生成的列插入到df后面

    df = pd.concat([df, data[end_encode_cols]], axis=1)  # 最后将编码列后面的列数据插到df后
    # check
    print(f"检查one_hot后是否存在空值：{df.isnull().values.any()}")

    return df


def normalize(data):
    label = data.pop('Class')  # 分离label
    result = data.copy()  # 不要改变原始df
    normal_columns = data.columns   # 全部归一化

    scaler = MinMaxScaler()
    for col in normal_columns:
        result[col] = scaler.fit_transform(result[[col]])

    result = pd.concat([result, label], axis=1)  # 还原label
    # check
    print(f"检查normalize后是否存在空值：{result.isnull().values.any()}")

    return result


def data_preProcess(data_indicator):
    """
    # 读取数据，同时加表头name = filed_name, 去难度等级, 换label
    # data = preprocess(data_file)
    # one_hot编码
    # data = one_hot(data, encode_cols)
    # normal最大最小法归一化
    # data = normalize(data)
    """

    # 根据data_indicator确定处理流程
    if data_indicator == 'Train':   # 只处理训练集
        data_file = 'KDD_NSL/KDDTrain+.csv'
        save_path = 'Data_encoded/LSTM_data/Train_processed.csv'
        data = preprocess(data_file)

    # Train_20Percent中，数据类型不足，
    # KDDTrain+.csv中有70种service类型
    # KDDTrain+_20Percent中只有66种service类型
    # elif data_indicator == 'Train-20P':
    #     data_file = 'KDD_NSL/KDDTrain+_20Percent.csv'
    #     save_path = 'Data_encoded/LSTM_data/Train_20P_processed.csv'
    #     data = preprocess(data_file)

    elif data_indicator == 'Test':  # 只处理测试集
        data_file = 'KDD_NSL/KDDTest+.csv'
        save_path = 'Data_encoded/LSTM_data/Test_processed.csv'
        data = preprocess(data_file)

    elif data_indicator == 'Test-21':   # 只处理Test-21
        data_file = 'KDD_NSL/KDDTest-21.csv'
        save_path = 'Data_encoded/LSTM_data/Test_21_processed.csv'
        data = preprocess(data_file)

    elif data_indicator == 'combine':   # 将测试集和训练集合并处理
        train_file = 'KDD_NSL/KDDTrain+.csv'
        test_file = 'KDD_NSL/KDDTest+.csv'
        data = pd.concat([preprocess(train_file), preprocess(test_file)])
        save_path = 'Data_encoded/LSTM_data/combined_data_processed.csv'

    elif data_indicator == 'combine-21':    # 将train和test-21合并
        train_file = 'KDD_NSL/KDDTrain+.csv'
        test_file = 'KDD_NSL/KDDTest-21.csv'
        data = pd.concat([preprocess(train_file), preprocess(test_file)])
        save_path = 'Data_encoded/LSTM_data/combined_data-21_processed.csv'
    else:
        print("please check your data_indicator")
        return

    encode_cols = ['protocol_type', 'service', 'flag']
    data = normalize(one_hot(data, encode_cols))
    print("处理完毕,保存中...")
    data.to_csv(save_path, index=False)
    print(f"{data_indicator} data processed.csv 保存完毕!")


# 用resnet预处理函数处理好的数据生成Train_20Precent
def Train_20precent_preProcess():
    # 将子集按照全集的方式进行独热编码后的数据
    data = pd.read_csv('Data_encoded/matrix_data/Train_20Percent_encoded.csv')
    # 分离label列,即后五列
    labels = data.iloc[:, -5:]
    print(labels)
    print(data)
    labels = labels.to_numpy()
    print(labels)
    # labels还原
    indices = np.argmax(labels, axis=1)
    print(indices)
    # 如果有一个类别列表对应于独热编码的列
    categories = ['Dos', 'Normal', 'Probe', 'R2L', 'U2L']
    # label转为Class
    original_labels = [categories[index] for index in indices]
    print(original_labels)
    # list 转为Dataframe类型
    Class = pd.DataFrame(original_labels, columns=['Class'])
    print(Class)

    # 将Class拼接到data后
    data_class = data.iloc[:, :-5]
    print(data_class)
    data_processed = pd.concat([data_class, Class], axis=1)
    print(data_processed)
    data_processed.to_csv('Data_encoded/LSTM_data/Train_20P_processed.csv', index=False)


if __name__ == '__main__':
    data_preProcess('combine')
    data_preProcess('combine-21')
    data_preProcess('Train')
    data_preProcess('Test')
    data_preProcess('Test-21')
    data_preProcess('21')
    Train_20precent_preProcess()
