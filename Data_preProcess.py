# python3.10
# coding:utf-8


import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# real py file to preprocess data


# preProcess函数
# import pandas as pd
# # 从CSV文件中读取数据
# df = pd.read_csv('your_file.csv')
# # 选择需要独热编码的列
# columns_to_encode = df.columns[1:3]
# # 对这些列进行独热编码,删除原来的列并将新列插入在原来的位置
# df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
# # 将独热编码后的数据集保存到新的CSV文件中
# df_encoded.to_csv('your_file_encoded.csv', index=False)
# 在这段代码中，我们首先从CSV文件中读取数据。
# 然后，我们选择需要独热编码的列，这里假设是第2和第3列（在Python中，索引是从0开始的，所以这里的索引是1和2）。
# 最后，我们使用pandas的get_dummies方法对这些列进行独热编码。
# 注意，get_dummies方法的columns参数接受一个列名的列表，这个列表中的列将被独热编码。其他列将保持不变。
# 这段代码将打印出独热编码后的DataFrame。每个字符串属性值都变成了一个新的列，如果原来的属性值存在，则新的列的值为1，否则为0。

def preProcess(source_data_path, preprocessed_data_path):
    source_file = source_data_path
    processed_file = preprocessed_data_path
    field_name_file = 'Data/field_name_file.csv'
    attack_type_file = 'Data/attack_type_file.csv'

    # 定义dataframe ，并定义column name，方便索引
    df = pd.read_csv(field_name_file, header=None, names=['name', 'data_type'])
    filed_names = df['name'].tolist()

    # 读取数据，带表头
    df = pd.read_csv(source_file, header=None, names=filed_names)
    # 删除第43列的'难度等级'
    df.drop(['difficult_level'], axis=1, inplace=True)
    # print(df)

    # 从CSV文件中读取映射,定义22种攻击小类标签对应的攻击类型
    attack_type_df = pd.read_csv(attack_type_file, header=None, names=['name', 'attack_type'])
    # 定义5大类和22小类的映射字典，方便替代
    mapping = attack_type_df.set_index('name').to_dict()['attack_type']

    # 替换训练集label中22小类为5大类标签
    df['label'] = df['label'].replace(mapping)
    # 将文件写入处理后文件
    df.to_csv(processed_file, index=False)

    # 独热编码
    oneHot_encoding(processed_file)
    # 归一化
    scale_data(processed_file)

    # df = pd.read_csv(processed_file)
    # print(df)


# 独热编码字符型数据
def oneHot_encoding(data_file_path):
    source_file = data_file_path
    # processed_file = preprocessed_data_path

    df = pd.read_csv(source_file)
    # 选择需要独热编码的列
    columns_to_encode = [df.columns[i] for i in [1, 2, 3, 41]]

    # 对每一列进行独热编码，并插入回原始位置，然后删除原始列
    for col in columns_to_encode:
        dummies = pd.get_dummies(df[col], prefix=col)
        # 将独热编码后的数据转换为整数类型
        dummies = dummies.astype(int)
        df = pd.concat([df.loc[:, :col].drop(col, axis=1), dummies, df.loc[:, col:].drop(col, axis=1)], axis=1)

    # 将独热编码后的数据集保存到源文件
    df.to_csv(data_file_path, index=False)


# 数据归一化
def scale_data(source_data_path):
    source_file = source_data_path
    df = pd.read_csv(source_file)

    # 创建归一化对象
    scaler = MinMaxScaler()

    # 对数据集进行归一化处理
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # 将归一化后的数据集保存到新的CSV文件中
    df_normalized.to_csv(source_file, index=False)


if __name__ == '__main__':
    data_file = 'Data/KDDTrain+_20Percent_top200Item.csv'
    preprocessed_data = 'KDDTrain+_20Percent_top200_preprocessed.csv'
    preProcess(data_file, preprocessed_data)

    # data = pd.read_csv(preprocessed_data)
    # print(data)
