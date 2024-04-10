# python3.10
# coding:utf-8

import os
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

    # print(df.info())
    # df = pd.read_csv(processed_file)
    # print(df)


# 独热编码字符型数据
# KDDTrain+.csv中有70种service类型
# KDDTrain+_20Percent中只有66种service类型
# KDDTrain+_20Percent_top200items中只有31种service类型
# 用全集编码后再编码子集
def oneHot_encoding(data_file_path):
    print('One-hot encoding...')

    # 读取完整数据集和子集
    full_file_path = 'Data/full_Train.csv'  # 只有KDDTrain+.csv的service属性是全齐的
    subset_encoded = pd.read_csv(full_file_path)
    subset_dataset = pd.read_csv(data_file_path)

    # 假设第2\3\4列是需要独热编码的列
    columns_to_encode = ['protocol_type', 'service', 'flag', 'label']    # 如果要编码label，只需要将 "label" 添加到 columns_to_encode 列表中

    # 将第2/3/4列转换为字符串类型
    for col in columns_to_encode:
        subset_encoded[col] = subset_encoded[col].astype(str)
        subset_dataset[col] = subset_dataset[col].astype(str)

    # fixme 子集按照全集编码后存在大量null行（详见Processed_data最后面的数据项，可尝试映射
    # 对完整数据集进行独热编码，并添加后缀 '_encoded'
    full_encoded = pd.get_dummies(subset_encoded, columns=columns_to_encode, prefix=columns_to_encode)
    # 将独热编码后的数据转换为整数类型
    full_encoded = full_encoded.astype(int)
    # 使用完整数据集的独热编码方式来对子集进行编码
    subset_encoded = pd.get_dummies(subset_dataset, columns=columns_to_encode, prefix=columns_to_encode)
    # 将独热编码后的数据转换为整数类型
    full_encoded = full_encoded.astype(int)

    # 创建一个集合以提高查找效率
    full_encoded_columns_set = set(full_encoded.columns)

    # 确保子集的独热编码包含完整数据集的所有列，并按照完整数据集的列顺序重新排列
    for col in full_encoded_columns_set:
        if col not in subset_encoded.columns:
            subset_encoded[col] = 0

    subset_encoded = subset_encoded[full_encoded.columns]

    # 独热编码后的列插入到原始列的位置
    for col in reversed(columns_to_encode):  # 反向操作，以防止索引更改影响后续的插入操作
        col_index = subset_dataset.columns.get_loc(col)
        # 提取出独热编码的列
        encoded_cols = [c for c in subset_encoded.columns if c.startswith(col)]
        # 删除原始列
        subset_encoded.drop(columns=encoded_cols, inplace=True)
        # 将独热编码的列插入到原始列的位置
        subset_encoded = pd.concat(
            [subset_encoded.iloc[:, :col_index], full_encoded[encoded_cols], subset_encoded.iloc[:, col_index:]],
            axis=1)

    # # fixme 将第5列和第76列挪到87列附近，先将76列挪到87列，再将第5列挪到86列
    # # 将第76列移动到第87列
    # subset_encoded.insert(86, 'flag_RSTR', subset_encoded.iloc[:, 75])
    #
    # # 将第77-86列向前移动一列
    # for i in range(86, 77, -1):
    #     subset_encoded.iloc[:, i] = subset_encoded.iloc[:, i - 1]
    #
    # # 将第87列移动到第86列的位置
    # subset_encoded.iloc[:, 85] = subset_encoded.iloc[:, 86]
    #
    # # 删除第80列
    # subset_encoded.drop(columns=['flag_RSTR'], inplace=True)

    # 提取子集文件名，生成新的文件名
    subset_file_name = os.path.basename(data_file_path)  # 提取文件名，例如 "Your_Subset.csv"
    subset_file_name_without_ext = os.path.splitext(subset_file_name)[0]  # 去掉扩展名，例如 "Your_Subset"
    new_file_name = f"Processed_{subset_file_name_without_ext}.csv"  # 添加前缀，生成新的文件名，例如 "Your_Subset_encoded.csv"

    # 保存处理后的子集数据集
    subset_encoded.to_csv(new_file_name, index=False)
    print(new_file_name+' One-hot encoding done!')


def print_data_info(data_file_path):
    # 读取CSV文件
    df = pd.read_csv(data_file_path)

    # 计算和打印不同属性的数量
    service_properties = df['service'].nunique()
    flag_properties = df['flag'].nunique()
    protocol_type_properties = df['protocol_type'].nunique()
    print('Number of service properties=', service_properties)
    print('Number of flag properties=', flag_properties)
    print('Number of protocol types=', protocol_type_properties)

    # 显示所有不同的属性
    service_properties = df['service'].unique()
    flag_properties = df['flag'].unique()
    protocol_type_properties = df['protocol_type'].unique()
    print('service properties:', service_properties)
    print('flag properties:', flag_properties)
    print('protocol types:', protocol_type_properties)


# 数据归一化
def scale_data(source_data_path):
    print('Scaling...')
    source_file = source_data_path
    df = pd.read_csv(source_file)

    # 创建归一化对象
    scaler = MinMaxScaler()

    # 对数据集进行归一化处理
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # 将归一化后的数据集保存到新的CSV文件中
    df_normalized.to_csv(source_file, index=False)
    print(source_file + ' Scaling done!')


if __name__ == '__main__':
    # 源文件路径
    # Train_file = 'Data/KDDTrain+.csv'
    # Train_20_file = 'Data/KDDTrain+_20Percent.csv'
    # Train_20_top200_file = 'Data/KDDTrain+_20Percent_top200Item.csv'

    # 加表头，去掉第43列的难度等级后的文件路径
    # Processed_full_Train_file = 'Data/full_Train.csv'   # 作为全service类型文件独热编码，保证其他缺项文件独热编码一致
    Processed_Train_file = 'Data/Train.csv'
    Processed_Train_20_file = 'Data/Train_20Percent.csv'
    Processed_Train_20_top200_file = 'Data/Train_20Percent_top200.csv'

    # 为数据文件添加表头，去除difficult_level，以方便后面的子集使用该文件进行独热编码，运行一次就好
    # preProcess(Train_file, Processed_full_Train_file)
    # preProcess(Train_file, Processed_Train_file)
    # preProcess(Train_20_file, Processed_Train_20_file)
    # preProcess(Train_20_top200_file, Processed_Train_20_top200_file)

    # 独热编码
    oneHot_encoding(Processed_Train_20_top200_file)
    # oneHot_encoding(Processed_Train_20_file)
    # oneHot_encoding(Processed_Train_file)

    # 归一化
    # scale_data('Processed_Train.csv')
    # scale_data('Processed_Train_20Percent.csv')
    # scale_data('Processed_Train_20Percent_top200.csv')

    # print_data_info(Processed_Train_20_top200_file)

    data = pd.read_csv('Processed_Train_20Percent_top200.csv')
    print(data.columns)
    print(data.describe())
    print(data.info)

    print(data.isnull().sum())
