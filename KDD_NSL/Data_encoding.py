
# 数据编码处理
# def fileprocess(filename):
import numpy as np
fr = open('KDDTrain+_20Percent_top200Item.csv')  # 打开文件
arraylines = fr.readlines(1)  # 按行读取文件全部内容
print(arraylines)
arraylines[0].strip()
print(arraylines)
numberlines = len(arraylines)  # 总行数
print("numberlines = ", numberlines)

returnMatrix = np.zeros((numberlines, 6))  # 定义一个矩阵存储前六列
returnlabelVector = []  # 返回标签向量
index = 0  # 行索引值
for line in arraylines:
    line.strip()
    l = line.split(",")  # 将每一行按“,”切片
    print(l)
    print(len(l))


# def file2matrix(filename):
#     # 打开文件
#     fr = open(filename)
#     # 读取文件所有内容
#     arrayOLines = fr.readlines()
#     # 得到文件行数
#     numberOfLines = len(arrayOLines)
#     # 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
#     returnMat = np.zeros((numberOfLines, 4))
#     # 返回的分类标签向量
#     classLabelVector = []
#     # 行的索引值
#     index = 0
#     for line in arrayOLines:
#         # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
#         line = line.strip()
#         # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
#         listFromLine = line.split(' ')
#         # 打乱数据
#         # listFromLine = shuffle_columns(listFromLine)
#         # 将数据1-4列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
#         returnMat[index, :] = listFromLine[1:5]
#         # 进行分类
#         if listFromLine[-1] == '"setosa"':
#             classLabelVector.append(1)
#         elif listFromLine[-1] == '"versicolor"':
#             classLabelVector.append(2)
#         elif listFromLine[-1] == '"virginica"':
#             classLabelVector.append(3)
#         index += 1
#     return returnMat, classLabelVector
#
# # 示例数据
# filename = "KDD_NSL/KDDTrain+_20Percent.txt"
# data, label = file2matrix(filename)
# print(label)
#
# # 计算信息熵
# entropy = calculate_entropy(label)
# print("信息熵: ", entropy)