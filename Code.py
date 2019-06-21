#-*-coding:utf-8-*-

# 测试ball_tree和kd_tree的用法

# from sklearn.neighbors import KDTree, BallTree
#
# import numpy as np
# import pickle
#
# # rng = np.random.RandomState(0)
# # X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
# X = np.random.rand(10, 3)
# tree = BallTree(X, leaf_size=2)
# s = pickle.dumps(tree)
# tree_copy = pickle.loads(s)
# dist, ind = tree_copy.query(X[:1], k=3)
# print(ind)  # indices of 3 closest neighbors
# print(dist)
# a = [ind[j][q] for q in range(ind.shape[1]) for j in range(ind.shape[0])]
# print(a)
# b = list(ind[0, :])
# print(b)


# 获取路径

import os
import time

# 当前目录
# current_path = os.getcwd()
# print(current_path)
#
# # 上一级目录
# father_path = os.path.abspath(os.path.dirname(os.getcwd()))
# print(father_path)
#
# # 上上一级目录
# grandfather_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
# print(grandfather_path)
#
# a = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
# print(a)

#========================================$
# 读取tensor

# import torch
#
#
# a = [torch.LongTensor([i]).cuda()
#      if torch.cuda.is_available() else torch.LongTensor([i])
#      for i in range(10)]
# a.append(2)
# print(a)
# print(type(a[0]), type(a[-1]))
#
# b = [int(i.cpu().numpy()) if i != 2 else i for i in a]
# print(b)

#========================================#
# set集合的使用

# a = [[3, 4, 5, 6, 7, 7],
#      [3, 5, 6, 7, 7, 8],
#      [5, 6, 7, 7, 8, 2],
#      [3, 4, 4, 6, 7, 7],
#      [3, 4, 4, 6, 7, 7],
#      [3, 4, 4, 5, 6, 7],
#      [3, 4, 5, 5, 7, 7],
#      [3, 4, 5, 6, 7, 7]]
#
# b = []
# for i in a:
#     if 2 in i:  # 如果i中有2的话，去除2
#         i.remove(2)
#     i = sorted(set(i), reverse=False)  # 先去重，然后升序排列
#     b.append(i)
#
# print(b)


#=================================#
# 保存数据到Excel中

import os
import csv
import random

# lab_num = 6
# ins_num = 10
#
# real_labs = [[random.randint(0, lab_num) for _ in range(lab_num)] for _ in range(ins_num)]
# predict_labs = [[random.randint(0, lab_num) for _ in range(lab_num)] for _ in range(ins_num)]
#
# with open((os.getcwd() + '//MLL_Save_Csv//real_labs.csv'), 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     for list in real_labs:
#         print(list)
#         csv_writer.writerow(list)
#
# with open((os.getcwd() + '//MLL_Save_Csv//predict_labs.csv'), 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     for list in predict_labs:
#         print(list)
#         csv_writer.writerow(list)


import numpy as np

a = np.array([[0.3, 0.1, 0.4],
              [0.5, 0.2, 0.7]])
print(a.shape)

b = 1 - a
print(b)




