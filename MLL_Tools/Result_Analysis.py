#-*-coding:utf-8-*-

# 对结果进行数据分析

import csv
import numpy as np

with open('real_3.csv', 'r', newline='') as real_labs_file:
    reader = csv.reader(real_labs_file)
    real_labs = np.array([[int(j) for j in i] for i in reader])
    # print(real_labs.shape)

with open('pos_3.csv', 'r', newline='') as pos_labs_file:
    reader = csv.reader(pos_labs_file)
    pos_labs = np.array([[int(j) for j in i] for i in reader])
    # print(pos_labs.shape)

with open('neg_3.csv', 'r', newline='') as neg_labs_file:
    reader = csv.reader(neg_labs_file)
    neg_labs = np.array([[int(j) for j in i] for i in reader])
    # print(neg_labs.shape)

ins_num, lab_num = real_labs.shape
total_labs = ins_num * lab_num

# 下一步先统计，如果正负两个相同的情况下，和正确也相同的数量/概率
pos_equal_neg = 0
pos_equal_neg_ = 0
for i in range(ins_num):
    for j in range(lab_num):
        if pos_labs[i][j] == neg_labs[i][j]:
            if pos_labs[i][j] == real_labs[i][j]:
                if neg_labs[i][j] == real_labs[i][j]:
                    pos_equal_neg += 1
            else:
                pos_equal_neg_ += 1
equal_3_ratio = pos_equal_neg / total_labs
equal_3_ratio_ = pos_equal_neg_ / total_labs
print("正负标记相同的情况下，和正确标记也相同的数量、概率: %d/%d %4.2f"
      %(pos_equal_neg, total_labs, equal_3_ratio))
print("正负标记相同的情况下，和正确标记不相同的数量、概率: %d/%d %4.2f"
      %(pos_equal_neg_, total_labs, equal_3_ratio_))
print('正负解码器标记相同的情况下，和正确标记也相同的概率 %4.2f，不相同的概率 %4.2f'
      % ((equal_3_ratio)/(equal_3_ratio + equal_3_ratio_), (equal_3_ratio_)/(equal_3_ratio+equal_3_ratio_)))

# 统计正负解码器不相同的情况下，正解码器标记和正确标记相同的数量/概率
pos_equal_real = 0
for i in range(ins_num):
    for j in range(lab_num):
        if pos_labs[i][j] != neg_labs[i][j] and pos_labs[i][j] == real_labs[i][j]:
            pos_equal_real += 1
equal_2_ratio_pos = pos_equal_real / total_labs
print('正负解码器解码标记不相同的情况下，正解码器解码标记和正确标记相同的数量、概率: %d/%d %4.2f'
      % (pos_equal_real, total_labs, equal_2_ratio_pos))

# 统计正负解码器不相同的情况下，负解码器标记和正确正确标记相同的数量/概率
neg_equal_real = 0
for i in range(ins_num):
    for j in range(lab_num):
        if pos_labs[i][j] != neg_labs[i][j] and neg_labs[i][j] == real_labs[i][j]:
            neg_equal_real += 1
equal_2_ratio_neg = neg_equal_real / total_labs
print('正负解码器解码标记不相同的情况下，负解码器解码标记和正确标记相同的数量、概率: %d/%d %4.2f'
      % (neg_equal_real, total_labs, equal_2_ratio_neg))

print('正负标记不相同的情况下，最终正解码器标记与正确标记相同的概率 %4.2f，负解码器标记与正确标记相同的概率 %4.2f'
      %((pos_equal_real)/(pos_equal_real+neg_equal_real),
        ((neg_equal_real)/(pos_equal_real+neg_equal_real))))

# 统计正负解码器标记的相同/不相同数量
equal_2 = 0
not_equal_2 = 0
for i in range(ins_num):
    for j in range(lab_num):
        if pos_labs[i][j] == neg_labs[i][j]:
            equal_2 += 1
        else:
            not_equal_2 += 1
print('正负解码器标记相同的个数和概率 %d %4.2f，正负解码器标记不相同的个数和概率 %d %4.2f'
      %(equal_2, equal_2/total_labs, not_equal_2, not_equal_2/total_labs))

