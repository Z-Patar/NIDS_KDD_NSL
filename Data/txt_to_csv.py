import csv

# 读取txt文件
txt_file_path = '../KDD_NSL/KDDTest+.txt'
csv_file_path = 'KDDTest+.csv'

with open(txt_file_path, 'r') as txt_file, open(csv_file_path, 'w', newline='') as csv_file:
    # 创建CSV写入器
    csv_writer = csv.writer(csv_file)

    # 使用CSV读取器逐行读取txt文件
    csv_reader = csv.reader(txt_file)

    # 将每一行的内容写入CSV文件
    for row in csv_reader:
        csv_writer.writerow(row)

print(f"Successfully converted {txt_file_path} to {csv_file_path}.")