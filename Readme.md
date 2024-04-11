# Readme

## 01 Data_preprocess 数据预处理

`01Data_preprocess.py`文件对数据进行了预处理
1. 去除了第43列(难度等级，与学习无用)，给各列加上了列头，将22种攻击类型映射为5大类攻击
2. 对源数据中的非数字变量进行了独热编码， 将41维特征化为122维特征，1维`outputs`化为5维`label`具体为第2列`protocol_type`、第3列`service`、第4列`flag`、第42列`label`
3. 将其余38列数据归一化，归一化方法为最大最小值

## 测试实现CNN
