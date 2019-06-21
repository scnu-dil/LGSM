#-*-coding:utf-8-*-

import numpy as np

import os

import time

from sklearn.model_selection import train_test_split

import random

import matplotlib.pyplot as plt

from MLL_Tools.Evaluate import mll_evaluate

# 把输入数据集和目标数据集分为训练，验证和测试三个部分
def train_val_test_2(data_1, data_2, train_size=0.9, test_size=0.8, shuffle=True):
    # 这里要先给特征集最后一列加上一个序号，方便以后特征的排序
    features_indexes = [i for i in range(data_1.shape[0])]
    new_data_1 = np.column_stack((data_1, features_indexes))

    # 这里也要先给标记集最后一列加上一个序号，方便后续的标记寻找
    labels_indexes = [i for i in range(data_2.shape[0])]
    new_data_2 = np.column_stack((data_2, labels_indexes))


    train_data_x, val_test_data_x, train_data_y, val_test_data_y = train_test_split(
        new_data_1, new_data_2, train_size=train_size, test_size=1-train_size, shuffle=shuffle,
        random_state=2019
    )
    val_data_x, test_data_x, val_data_y, test_data_y = train_test_split(
        val_test_data_x, val_test_data_y, train_size=1-test_size,
        test_size=test_size, shuffle=True, random_state=2019
    )

    return train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y


# 可视化损失
def show_loss(x, y, picture_name, color='g'):
    lines = plt.plot(x, y, '.-')
    plt.setp(lines, color=color, linewidth=.5)
    plt.title('Train loss in every steps of epochs')
    plt.ylabel('loss')
    plt.xlabel('each steps of every epochs')

    # plt.show()
    # 当前项目路径
    project_path = os.getcwd()
    plt.savefig((project_path + '/MLL_Save_Pictures/' + picture_name + '.png'), dpi=100)
    print(project_path + '/MLL_Save_Pictures/' + picture_name + '.png')


# 把实际和标记序列和预测的标记序列进行转化
def real_predict_labs_change(real_pos_labs, predict_pos_labs, real_neg_labs, predict_neg_labs, lab_num):
    """
    :param real_pos_labs: 实际的正标记（正标记记为1，负标记记为0）
    :param predict_pos_labs: 预测的正标记（正标记记为1，负标记记为0）
    :param real_neg_labs: 实际的负标记（正标记记为1，负标记记为0）
    :param predict_neg_labs: 预测的负标记（正标记记为1，负标记记为0）
    :param lab_num:
    :return:
    """
    instance_num = len(real_pos_labs)

    # 减去3是对应前面的三个固定符（PAD, SOS, EOS）
    new_real_pos_labs = [[j - 3 for j in i] for i in real_pos_labs]
    real_pos_labs_0_1 = [[0 for _ in range(lab_num)] for _ in range(instance_num)]
    for index, i in enumerate(new_real_pos_labs):
        for j in i:
            real_pos_labs_0_1[index][j] = 1

    new_predict_pos_labs = []
    for i in predict_pos_labs:
        if 2 in i:
            i.remove(2)
        i = sorted(set([j - 3 for j in i]), reverse=False)
        new_predict_pos_labs.append(i)
    predict_pos_labs_0_1 = [[0 for _ in range(lab_num)] for _ in range(instance_num)]
    for index, i in enumerate(new_predict_pos_labs):
        for j in i:
            predict_pos_labs_0_1[index][j] = 1


    new_real_neg_labs = [[j - 3 for j in i] for i in real_neg_labs]
    real_neg_labs_0_1 = [[1 for _ in range(lab_num)] for _ in range(instance_num)]
    for index, i in enumerate(new_real_neg_labs):
        for j in i:
            real_neg_labs_0_1[index][j] = 0

    new_predict_neg_labs = []
    for i in predict_neg_labs:
        if 2 in i:
            i.remove(2)
        i = sorted(set([j -3 for j in i]), reverse=False)
        new_predict_neg_labs.append(i)
    predict_neg_labs_0_1 = [[1 for _ in range(lab_num)] for _ in range(instance_num)]
    for index, i in enumerate(new_predict_neg_labs):
        for j in i:
            predict_neg_labs_0_1[index][j] = 0

    return np.array(real_pos_labs_0_1), np.array(predict_pos_labs_0_1), \
           np.array(real_neg_labs_0_1), np.array(predict_neg_labs_0_1)


def evaluate(real_pos_labs, predict_pos_labs, real_neg_labs, predict_neg_labs):
    print('正标记！')
    subset_accuracy, average_precision, \
    hm, one_error, coverage, rank_loss = mll_evaluate(y_pred=predict_pos_labs, y_real=real_pos_labs)

    print('负标记！')
    subset_accuracy, average_precision, \
    hm, one_error, coverage, rank_loss = mll_evaluate(y_pred=predict_neg_labs, y_real=real_neg_labs)

real_pos_labs = [[8], [4], [4], [3], [5], [6], [8], [7]]
predict_pos_labs = [[8], [4], [3, 4], [5], [5], [6], [8], [7]]

real_neg_labs = [[3, 4, 5, 6, 7],
                 [3, 5, 6, 7, 8],
                 [3, 5, 6, 7, 8],
                 [4, 5, 6, 7, 8],
                 [3, 4, 6, 7, 8],
                 [3, 4, 5, 7, 8],
                 [3, 4, 5, 6, 7],
                 [3, 4, 5, 6, 8]]
predict_neg_labs = [[3, 4, 5, 6, 7, 7],
                    [3, 5, 6, 7, 7, 8],
                    [5, 6, 7, 7, 8, 2],
                    [3, 4, 4, 6, 7, 7],
                    [3, 4, 4, 6, 7, 7],
                    [3, 4, 4, 5, 6, 7],
                    [3, 4, 5, 5, 7, 7],
                    [3, 4, 5, 6, 7, 7]]

# real_pos_labs, predict_pos_labs, \
# real_neg_labs, predict_neg_labs = real_predict_labs_change(real_pos_labs, predict_pos_labs,
#                                                            real_neg_labs, predict_neg_labs, 6)
#
# evaluate(real_pos_labs, predict_pos_labs, real_neg_labs, predict_neg_labs)