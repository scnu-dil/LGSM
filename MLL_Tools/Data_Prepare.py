# 数据准备

#-*-coding:utf-8-*-

import os
import numpy
import numpy as np

from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree

import torch
from torch.autograd import Variable

from Parameters import Parameters
param = Parameters()


class DataPrepare(object):

    def __init__(self, mll_data_type:str, k_nearest_neighbor:int):
        self.mll_data_type = mll_data_type
        self.k_nearest_neighbor = k_nearest_neighbor
        self.PAD_token = 0
        self.USE_CUDA = torch.cuda.is_available()

    def get_mll_data(self):
        # 返回多标记数据的示例对象的特征集和标记集

        mll_data_set_path = os.getcwd() + '//MLL_Data//' + self.mll_data_type + '.csv'
        mll_data_set = numpy.loadtxt(open(mll_data_set_path, 'rb'), delimiter=',', skiprows=0)
        lab_num = 0

        if self.mll_data_type == 'yeast':
            lab_num = 14
        elif self.mll_data_type == 'scene':
            lab_num = 6
        elif self.mll_data_type == 'emotions':
            lab_num = 6
        elif self.mll_data_type == 'enron':
            lab_num = 53
        elif self.mll_data_type == 'image':
            lab_num = 5

        return mll_data_set[:, :-lab_num], mll_data_set[:, -lab_num:]

    def sort_label_order(self, sort_label, sort_mode:str):
        # 返回排列之后的示例对象标记集

        if sort_mode == "random":  # 随机就相当于原顺序
            return sort_label

        ins_num, lab_num = sort_label.shape
        sum_every_col = (sort_label.sum(axis=0)).tolist()
        index_lab_num = {}
        for i in range(0, len(sum_every_col)):
            index_lab_num[i] = sum_every_col[i]
        reverse = True if sort_mode == 'descend' else False  # 是按照标记出现的次数进行降序还是升序排列
        new_list = sorted(index_lab_num, key=lambda x: index_lab_num[x], reverse=reverse)
        new_lable = np.zeros((ins_num, lab_num))
        for i in range(0, len(new_list)):
            new_lable[:, i] = sort_label[:, new_list[i]]

        return new_lable

    def get_k_from_train_features(self, train_features, val_features, test_features):
        # 构建基于训练示例对象的特征集的KD树，查询KD树，返回和当前示例对象特征最相似的k个训练示例特征序号，按照紧密程度由小到大排序

        feature_num = train_features.shape[1] - 1  # 特征最后加入此对象的编号，方面编码器对特征的查找

        train_features_ball_tree = BallTree(train_features[:, :-1], metric='euclidean')

        # 查询
        train_k_nearest_neighbor = []
        for i in train_features:
            one_feature = i[:-1].reshape((1, feature_num))
            similarity_value, k_ins_list = train_features_ball_tree.query(
                X=one_feature, k=self.k_nearest_neighbor + 1)  # 在训练集中查询k+1个，因为自己本身和自己也相似
            temp_1 = list(reversed(k_ins_list[0, :]))  # 按照密切程度由大到小排列
            temp_2 = [int(train_features[j][-1]) for j in temp_1]  # 查询到对象的特征编号
            train_k_nearest_neighbor.append(temp_2)

        # 查询k个， 按相似值由小到大排列
        val_k_nearest_neighbor = []
        for index, i in enumerate(val_features):
            one_feature = i[:-1].reshape((1, feature_num))
            _, k_ins_list = train_features_ball_tree.query(X=one_feature, k=self.k_nearest_neighbor)
            temp_1 = list(reversed(k_ins_list[0, :]))
            temp_2 = [int(train_features[j][-1]) for j in temp_1]
            temp_2.append(int(val_features[index][-1]))
            val_k_nearest_neighbor.append(temp_2)

        # 查询k个，按相似值由小到大排列
        test_k_nearest_neighbor = []
        for index, i in enumerate(test_features):
            one_feature = i[:-1].reshape((1, feature_num))
            _, k_ins_list = train_features_ball_tree.query(X=one_feature, k=self.k_nearest_neighbor)
            temp_1 = list(reversed(k_ins_list[0, :]))
            temp_2 = [int(train_features[j][-1]) for j in temp_1]
            temp_2.append(int(test_features[index][-1]))
            test_k_nearest_neighbor.append(temp_2)

        return np.array(train_k_nearest_neighbor), np.array(val_k_nearest_neighbor), \
               np.array(test_k_nearest_neighbor)

    def get_prior_probability(self, train_chain, val_chain, test_chain, labs, k):
        """
        根据特征链，获取每个示例对象标记的正概率α和负覆盖率β
        :return:
        """
        chain_set = [train_chain, val_chain, test_chain]
        train_val_test_alpha = []
        train_val_test_beta = []
        for a_chain_set in chain_set:
            ins_num = a_chain_set.shape[0]
            lab_num = labs.shape[1]
            alpha = np.zeros((ins_num, lab_num))
            for index, i in enumerate(a_chain_set):
                temp = 0
                for j in range(0, len(i)-1):
                    temp += labs[i[j]]
                alpha[index] = temp / k

            train_val_test_alpha.append(alpha)
            train_val_test_beta.append(1 - alpha)

        return train_val_test_alpha, train_val_test_beta

    def build_feature_chain(self, chain_type:int, train_k_near_nei, val_k_near_nei, test_k_near_nei):
        """
        构建特征链，用于特征链编码的读入
            类型一 0：先按相似值由小到大排列，然后最后放本特征，最近邻按照紧密程度由小到大排序
            类型二 1：先读入相似值链，用本特征加入相似值特征链的空格中，最近邻和本对象交替
        :return: 构造好的特征链
        """
        if chain_type:  # 第二种类型的特征链
            new_train_k_near_nei = []
            for i in train_k_near_nei:
                this_feature = i[-1]  # 获取本对象序号
                temp = []
                for j in range(train_k_near_nei.shape[1] - 1):  # 交替排列最近邻序号和本对象序号
                    temp.append(i[j])
                    temp.append(this_feature)
                new_train_k_near_nei.append(temp)

            new_val_k_near_nei = []
            for i in val_k_near_nei:
                this_feature = i[-1]
                temp = []
                for j in range(val_k_near_nei.shape[1] - 1):
                    temp.append(i[j])
                    temp.append(this_feature)
                new_val_k_near_nei.append(temp)

            new_test_k_near_nei = []
            for i in test_k_near_nei:
                this_feature = i[-1]
                temp = []
                for j in range(test_k_near_nei.shape[1] - 1):
                    temp.append(i[j])
                    temp.append(this_feature)
                new_test_k_near_nei.append(temp)

            return new_train_k_near_nei, new_val_k_near_nei, new_test_k_near_nei
        else:
            return train_k_near_nei, val_k_near_nei, test_k_near_nei

    def str2list(self, target_str):
        target_str = target_str.lstrip('[').rstrip(']').replace(' ', '').split(',')
        target_list = [int(target_str[i]) for i in range(len(target_str))]
        return target_list

    def merge_features_chain_poslabs_neglabs(self, featuresChain_train, featuresChain_val,
                                             featuresChain_test, labels:int):
        """合并示例的特征最近邻特征链和它的正负标签, 分为训练，验证和测试"""
        feaChain_pos_neg_train = []
        for i in featuresChain_train:
            this_instance = i[-1]  # 最后一个就是当前对象序号
            poslabs_neglabs = labels[this_instance]  # 根据当前对象的序号找到他的标记
            poslabs = []
            neglabs = []
            _ = [poslabs.append(index+3) if j > 0 else neglabs.append(index + 3)
                 for index, j in enumerate(poslabs_neglabs.copy())]
            poslabs.append(param.EOS_TOKEN)  # 标记结束要加上序列终止符
            neglabs.append(param.EOS_TOKEN)
            temp = [str(i), str(poslabs), str(neglabs)]
            feaChain_pos_neg_train.append(temp)

        feaChain_pos_neg_val = []
        for i in featuresChain_val:
            this_instance = i[-1]
            poslabs_neglabs = labels[this_instance]
            poslabs = []
            neglabs = []
            _ = [poslabs.append(index + 3) if j > 0 else neglabs.append(index + 3)
                 for index, j in enumerate(poslabs_neglabs.copy())]
            poslabs.append(param.EOS_TOKEN)  # 标记结束要加上序列终止符
            neglabs.append(param.EOS_TOKEN)
            temp = [str(i), str(poslabs), str(neglabs)]
            feaChain_pos_neg_val.append(temp)

        feaChain_pos_neg_test = []
        for i in featuresChain_test:
            this_instance = i[-1]
            poslabs_neglabs = labels[this_instance]
            poslabs = []
            neglabs = []
            _ = [poslabs.append(index + 3) if j > 0 else neglabs.append(index + 3)
                 for index, j in enumerate(poslabs_neglabs.copy())]
            poslabs.append(param.EOS_TOKEN)  # 标记结束要加上序列终止符
            neglabs.append(param.EOS_TOKEN)
            temp = [str(i), str(poslabs), str(neglabs)]
            feaChain_pos_neg_test.append(temp)

        return feaChain_pos_neg_train, feaChain_pos_neg_val, feaChain_pos_neg_test

    def pad_seq(self, seq, max_length:int):
        seq += [self.PAD_token for i in range(max_length - len(seq))]
        return seq

    def batch_2_tensor(self, batch_data):
        """
        把批次大小的数据转化为tensor
        :return:
        """
        features_chain_sequence = []
        poslabs_sequence = []
        neglabs_sequence = []

        for i in range(len(batch_data[0])):
            features_chain_sequence.append(self.str2list(batch_data[0][i]))
            poslabs_sequence.append(self.str2list(batch_data[1][i]))
            neglabs_sequence.append(self.str2list(batch_data[2][i]))

        seq_pairs = sorted(zip(features_chain_sequence, poslabs_sequence, neglabs_sequence),
                           key=lambda p: len(p[0]), reverse=True)
        features_chain_sequence, poslabs_sequence, neglabs_sequence = zip(*seq_pairs)

        features_chain_lengths = [len(s) for s in features_chain_sequence]
        features_chain_padded = [self.pad_seq(s, max(features_chain_lengths))
                                 for s in features_chain_sequence]
        poslabs_lengths = [len(s) for s in poslabs_sequence]
        poslabs_padded = [self.pad_seq(s, max(poslabs_lengths))
                          for s in poslabs_sequence]
        neglabs_lengths = [len(s) for s in neglabs_sequence]
        neglabs_padded = [self.pad_seq(s, max(neglabs_lengths))
                         for s in neglabs_sequence]

        features_chain_variable = Variable(torch.LongTensor(features_chain_padded)).transpose(0, 1)
        poslabs_variable = Variable(torch.LongTensor(poslabs_padded)).transpose(0, 1)
        neglabs_variable = Variable(torch.LongTensor(neglabs_padded)).transpose(0, 1)

        if self.USE_CUDA:
            features_chain_variable = features_chain_variable.cuda()
            poslabs_variable = poslabs_variable.cuda()
            neglabs_variable = neglabs_variable.cuda()

        return features_chain_variable, features_chain_lengths, poslabs_variable, poslabs_lengths, \
               neglabs_variable, neglabs_lengths
