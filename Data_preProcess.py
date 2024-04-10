# python3.10
# coding:utf-8

import numpy as np
import csv
import pandas as pd

global staut_list
# real py file to preprocess data

'''
import pandas as pd
# 从CSV文件中读取数据
df = pd.read_csv('your_file.csv')
# 选择需要独热编码的列
columns_to_encode = df.columns[1:3]
# 对这些列进行独热编码,删除原来的列并将新列插入在原来的位置
df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
# 将独热编码后的数据集保存到新的CSV文件中
df_encoded.to_csv('your_file_encoded.csv', index=False)
在这段代码中，我们首先从CSV文件中读取数据。
然后，我们选择需要独热编码的列，这里假设是第2和第3列（在Python中，索引是从0开始的，所以这里的索引是1和2）。
最后，我们使用pandas的get_dummies方法对这些列进行独热编码。
注意，get_dummies方法的columns参数接受一个列名的列表，这个列表中的列将被独热编码。其他列将保持不变。
这段代码将打印出独热编码后的DataFrame。每个字符串属性值都变成了一个新的列，如果原来的属性值存在，则新的列的值为1，否则为0。
'''


def preProcess(source_data_path, preprocessed_data_path):
    source_file = source_data_path
    processed_file = preprocessed_data_path

    # 从CSV文件中读取数据
    df = pd.read_csv(source_file)

    # 选择需要独热编码的列
    columns_to_encode = [df.columns[i] for i in [1, 2, 3, 41]]

    # 对每一列进行独热编码，并插入回原始位置，然后删除原始列
    for col in columns_to_encode:
        dummies = pd.get_dummies(df[col], prefix=col)
        # 将独热编码后的数据转换为整数类型
        dummies = dummies.astype(int)
        df = pd.concat([df.loc[:, :col].drop(col, axis=1), dummies, df.loc[:, col:].drop(col, axis=1)], axis=1)

    # 将独热编码后的数据集保存到新的CSV文件中
    df.to_csv(processed_file, index=False)


def find_index(x, y):
    return [a for a in range(len(y)) if y[a] == x]


def handleProtocol(input):
    protoclo_list = ['tcp', 'udp', 'icmp']
    if input[1] in protoclo_list:
        return find_index(input[1], protoclo_list)[0]


def handleService(input):
    service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                    'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
                    'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin',
                    'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn',
                    'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private',
                    'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat',
                    'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11',
                    'Z39_50']
    if input[2] in service_list:
        return find_index(input[2], service_list)[0]


def handleFlag(input):
    flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    if input[3] in flag_list:
        return find_index(input[3], flag_list)[0]


def handleLabel(input):
    global staut_list
    # ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.', 'spy.', 'rootkit.']
    if input[41] in staut_list:
        return find_index(input[41], staut_list)[0]
    else:
        staut_list.append(input[41])
        return find_index(input[41], staut_list)[0]


if __name__ == '__main__':
    data_file = 'Data/KDDTrain+_20Percent_top200Item.csv'
    preprocessed_data = 'KDDTrain+_20Percent_preprocessed.csv'
    global staut_list
    staut_list = []
    preProcess(data_file, preprocessed_data)
    # print staut_list
    data = pd.read_csv(preprocessed_data)
    print(data)
