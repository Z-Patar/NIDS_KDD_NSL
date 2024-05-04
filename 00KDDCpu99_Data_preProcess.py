#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 对数据进行初步预处理，方便后面编码
def preProcess(source_data_path, preprocessed_data_path):
    source_file = source_data_path
    processed_file = preprocessed_data_path
    field_name_file = 'KDDCup99/00field_name_file.csv'  # 列名
    attack_type_file = 'KDDCup99/01attack_type_file.csv'  # 22小类和5大类的对应字典

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


# 初步处理,所有数据
def preProcess_all():
    print('all data begin Preprocess ...\n')
    # 源文件路径
    Train_file = 'KDDCup99/data/kddcup.data'
    Train_10_file = 'KDDCup99/data/kddcup.data_10_percent'
    Test_file = 'KDDCup99/data/corrected'

    # 加表头，去掉第43列的难度等级后的文件路径
    Processed_full_Train_file = 'KDDCup99/Temp_data99/full_Train.csv'  # 作为全service类型文件独热编码，保证其他缺项文件独热编码一致
    Processed_Train_file = 'KDDCup99/Temp_data99/Train.csv'
    Processed_Train_10_file = 'KDDCup99/Temp_data99/Train_10Percent.csv'
    Processed_Test_file = 'KDDCup99/Temp_data99/Test.csv'

    # 为数据文件添加表头，去除difficult_level，以方便后面的子集使用该文件进行独热编码，运行一次就好
    preProcess(Train_file, Processed_full_Train_file)
    preProcess(Train_file, Processed_Train_file)
    preProcess(Train_10_file, Processed_Train_10_file)
    preProcess(Test_file, Processed_Test_file)

    print("All data preprocess done!\n")


# 独热编码字符型数据
# 用全集编码后再编码子集
def oneHot_encoding(data_file_path):
    # print(data_file_path + ' One-hot encoding...')

    # 读取完整数据集和子集
    full_file_path = 'KDDCup99/Temp_data99/full_Train.csv'  # 只有KDDcup.data.csv的service属性是全齐的
    full_dataset = pd.read_csv(full_file_path)
    subset_dataset = pd.read_csv(data_file_path)
    # 提取子集文件名，生成新的文件名
    subset_file_name = os.path.basename(data_file_path)  # 提取文件名，例如 "Your_Subset.csv"
    subset_file_name_without_ext = os.path.splitext(subset_file_name)[0]  # 去掉扩展名，例如 "Your_Subset"
    new_file_name = f"KDDCup99/data_encoded/{subset_file_name_without_ext}_encoded.csv"  # 添加后缀，生成新的文件名，例如 "Data_encoded/Your_Subset_encoded.csv"

    # 假设第2\3\4列是需要独热编码的列。如果要编码label，只需要将 "label" 添加到 columns_to_encode 列表中
    columns_to_encode = ['protocol_type', 'service', 'flag', 'label']

    # 将需要编码的列转换为字符串类型
    for col in columns_to_encode:
        full_dataset[col] = full_dataset[col].astype(str)
        subset_dataset[col] = subset_dataset[col].astype(str)

    # 子集按照全集编码后存在大量null行（没关系，编码后直接用dropna()删除有null的行即可
    # 对完整数据集进行独热编码，并添加后缀 '_encoded'
    full_encoded = pd.get_dummies(full_dataset, columns=columns_to_encode, prefix=columns_to_encode)
    # 将独热编码后的数据转换为浮点数类型
    full_encoded = full_encoded.astype(int)

    # 使用完整数据集的独热编码方式来对子集进行编码
    subset_encoded = pd.get_dummies(subset_dataset, columns=columns_to_encode, prefix=columns_to_encode)
    # 将独热编码后的数据转换为浮点数类型
    full_encoded = full_encoded.astype(int)

    # 创建一个集合以提高查找效率
    full_encoded_columns_set = set(full_encoded.columns)

    # 确保子集的独热编码包含完整数据集的所有列
    for col in full_encoded_columns_set:
        if col not in subset_encoded.columns:
            subset_encoded[col] = 0
    # 重新排列子集的列以匹配完整数据集的列顺序
    subset_encoded = subset_encoded[full_encoded.columns]

    # 初始化一个新的DataFrame来存储插入独热编码列后的数据
    encoded_inserted_df = pd.DataFrame()

    # 遍历完整数据集的每一列
    for col in full_dataset.columns:
        if col in columns_to_encode and col != 'label':
            # 如果当前列是需要独热编码的列，插入独热编码后的列
            encoded_cols = [c for c in full_encoded.columns if c.startswith(col + "_")]
            encoded_inserted_df = pd.concat([encoded_inserted_df, subset_encoded[encoded_cols]], axis=1)
        elif col != 'label':
            # 如果当前列不需要独热编码，直接插入原始列
            encoded_inserted_df = pd.concat([encoded_inserted_df, subset_dataset[[col]]], axis=1)

    # 确保最后的 'label' 独热编码列也被添加到 DataFrame 中
    label_encoded_cols = [c for c in full_encoded.columns if c.startswith('label_')]
    encoded_inserted_df = pd.concat([encoded_inserted_df, subset_encoded[label_encoded_cols]], axis=1)

    # 在subset_encoded中计算行数和列数
    print("(去除null前): " + new_file_name + "中的行数和列数 = ", encoded_inserted_df.shape)
    subset_encoded.dropna()  # 去除有null的行
    print("(去除null后): " + new_file_name + "中的行数和列数 = ", encoded_inserted_df.shape)

    # 保存处理后的子集数据集
    encoded_inserted_df.to_csv(new_file_name, index=False)
    print(new_file_name + ' One-hot encoding done!\n')


# 独热编码所有数据
def one_hot_all():
    print('all data begin One-hot encode ...\n')

    Train_file = 'KDDCup99/Temp_data99/Train.csv'
    Train_10_file = 'KDDCup99/Temp_data99/Train_10Percent.csv'
    Test_file = 'KDDCup99/Temp_data99/Test.csv'

    # oneHot_encoding(Train_20_top200_file)
    oneHot_encoding(Train_10_file)
    oneHot_encoding(Train_file)
    oneHot_encoding(Test_file)

    print('all data one-hot encode done!\n')


# 数据归一化
def scale_data(source_data_path):
    # print('Scaling...')

    source_file = source_data_path
    encoded_dataset = pd.read_csv(source_file)

    # 创建归一化对象
    scaler = MinMaxScaler()

    # 对数据集的每一列进行归一化
    for column in encoded_dataset.columns:
        # 由于归一化需要2D数据，我们使用.values.reshape(-1, 1)将数据转换为2D
        encoded_dataset[column] = scaler.fit_transform(encoded_dataset[[column]])

    # # 对数据集进行归一化处理
    # df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # 将归一化后的数据集保存到新的CSV文件中
    encoded_dataset.to_csv(source_file, index=False)
    print(source_file + ' Scaling done!\n')


# 归一化所有数据
def scale_all():
    print("all data begin scale:")

    Train_file = 'KDDCup99/data_encoded/Train_encoded.csv'
    Train_20_file = 'KDDCup99/data_encoded/Train_10Percent_encoded.csv'
    Test_file = 'KDDCup99/data_encoded/Test_encoded.csv'
    scale_data(Train_file)
    scale_data(Train_20_file)
    scale_data(Test_file)

    print("all data scale done!\n")


# 打印数据文件信息
def print_data_info(data_file_path):
    # 读取CSV文件
    df = pd.read_csv(data_file_path)
    print("\n" + data_file_path + "  file info is ")
    print(df.info)


# 打印所有预处理完成的数据文件信息
def print_all_data_info():
    test_data = 'KDDCup99/data_encoded/Test_encoded.csv'
    train_data = 'KDDCup99/data_encoded/Train_encoded.csv'
    train_20Percent_data = 'KDDCup99/data_encoded/Train_10Percent_encoded.csv'

    print("All preprocessed data info is as follows :")
    print_data_info(test_data)
    print_data_info(train_data)
    print_data_info(train_20Percent_data)


# 交换normal列和dos列，不影响结果，但混淆矩阵看着舒服
def exchange_normal_and_dos(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 交换第123列和第124列
    col123 = df.iloc[:, 122].copy()
    col124 = df.iloc[:, 123].copy()
    df.iloc[:, 122] = col124
    df.iloc[:, 123] = col123

    # 交换第123列和第124列的列名
    col_names = list(df.columns)
    col_names[122], col_names[123] = col_names[123], col_names[122]
    df.columns = col_names

    # 保存修改后的文件
    df.to_csv(file_path, index=False)


def exchange_all_nad():
    train_data = 'KDDCup99/data_encoded/Train_encoded.csv'
    train_10p_data = 'KDDCup99/data_encoded/Train_10Percent_encoded.csv'
    test_data = 'KDDCup99/data_encoded/Test_encoded.csv'
    exchange_normal_and_dos(train_data)
    exchange_normal_and_dos(train_10p_data)
    exchange_normal_and_dos(test_data)


if __name__ == '__main__':
    # print("All data preprocess begin! ...\n")
    # # 为数据文件添加表头，去除'difficult_level'
    # preProcess_all()
    #
    # # 独热编码
    # one_hot_all()
    #
    # # 归一化
    # scale_all()
    #
    # print("All data preprocess finish!\n")
    #
    # # 查看数据文件的shape
    # print_all_data_info()

    # 查看编码后的数据文件,如果dos在123列,normal在124列,则运行下面的数据,交换两列数据
    exchange_all_nad()

