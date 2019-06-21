#-*-coding:utf-8-*-

# 多标记学习标签序列生成模型

import torch
from torch.utils.data import DataLoader

from MLL_Modules.FeaturesChainEncoder import FeaturesChainEncoder  # 特征链编码器
from MLL_Modules.PositiveLabelsDecoder import PositiveLabelsDecoder  # 正标记解码器
from MLL_Modules.NegativeLabelsDecoder import NegativeLabelsDecoder  # 负标记解码器
from MLL_Modules.Features2Lables import Features2Labels  # 特征链到标记的序列生成类

from MLL_Tools.Data_Prepare import DataPrepare  # 数据准备
from MLL_Tools.Others import train_val_test_2, show_loss, real_predict_labs_change, evaluate

import time
import csv
import os

from Parameters import Parameters  # 参数类和对象
param = Parameters()


class LSGM(object):  # 标记序列生成解码器

    def __init__(self, mll_data_type:str, label_order:str, k_nearest_neighbors:int,
                 features_chain_type:int, save_loss_fig=True):
        print('多标记学习数据类型:{}'.format(mll_data_type),
              '| 获取%1.0f个最近邻' % k_nearest_neighbors)

        self.mll_data_type = mll_data_type  # 多标记数据类型
        self.label_order = label_order  # 标记排序顺序
        self.k_nearest_neighbors = k_nearest_neighbors  # 标记最近邻个数
        self.features_chain_type = features_chain_type  # 特征链类型
        self.save_loss_fig = save_loss_fig  # 是否保存损失函数图像

        self.__get_data()  # 数据准备函数
        self.__define_encoder_decoder()  # 定义编码器-解码器函数

    def __get_data(self):
        data_obj = DataPrepare(self.mll_data_type, self.k_nearest_neighbors)
        self.data_obj = data_obj
        mll_features, mll_labels = data_obj.get_mll_data()  # 分别获取示例特征集和标记集
        mll_labels_new = data_obj.sort_label_order(mll_labels, self.label_order)  # 排列标记顺序
        self.instance_num, self.features_num = mll_features.shape
        self.labels_num = mll_labels.shape[1]
        self.features = mll_features.copy()

        # 分离训练，验证和测试的特征和标记
        mll_features_train, mll_labels_train, mll_features_val, mll_labels_val, \
        mll_features_test, mll_labels_test = train_val_test_2(mll_features, mll_labels_new)

        # 找每个示例对象特征的k个最近邻
        k_near_nei_train, \
        k_near_nei_val, \
        k_near_nei_test = data_obj.get_k_from_train_features(
            mll_features_train, mll_features_val, mll_features_test)

        # 得到每个示例k个最近邻的正概率α和贝塔
        train_val_test_alpha, train_val_test_beta = data_obj.get_prior_probability(
            k_near_nei_train, k_near_nei_val, k_near_nei_test, mll_labels_new, self.k_nearest_neighbors)

        # 组织成对应的特征链
        new_k_near_nei_train, \
        new_k_near_nei_val, \
        new_k_near_nei_test = data_obj.build_feature_chain(
            self.features_chain_type, k_near_nei_train, k_near_nei_val, k_near_nei_test)

        # 形成特征链-正标记-负标记数据，分为训练，验证和测试数据
        self.feaChain_pos_neg_train, \
        self.feaChain_pos_neg_val, \
        self.feaChain_pos_neg_test = data_obj.merge_features_chain_poslabs_neglabs(
            new_k_near_nei_train, new_k_near_nei_val, new_k_near_nei_test, mll_labels_new)

    def __define_encoder_decoder(self):
        self.encoder = FeaturesChainEncoder(
            self.instance_num, self.features_num, self.features).to(param.device)

        self.pos_lab_decoder = PositiveLabelsDecoder(
            self.labels_num + 3, self.features_num).to(param.device)

        self.neg_lab_decoder = NegativeLabelsDecoder(
            self.labels_num + 3, self.features_num).to(param.device)

        features_2_labs = Features2Labels(self.encoder, self.pos_lab_decoder, self.neg_lab_decoder)
        self.features_2_labs = features_2_labs

    def __save_loss_pictures_pos(self, x, y):
        picture_name = 'pos_lab' + '_' + \
                       self.mll_data_type + '_' + \
                       str(self.k_nearest_neighbors) + '_' + \
                       str(self.features_chain_type) + '_' + \
                       str(param.train_epochs) + '_' + \
                       str(param.batch_size) + '_' + \
                       str(self.features_2_labs.optimizer_name) + '_' + \
                       str(param.learning_rate) + '_' + \
                       str(time.strftime("%m_%d_%H_%M_%S", time.localtime()))
        show_loss(x, y, picture_name, color='g')

    def __save_loss_pictures_neg(self, x, y):
        picture_name = 'neg_lab' + '_' + \
                       self.mll_data_type + '_' + \
                       str(self.k_nearest_neighbors) + '_' + \
                       str(self.features_chain_type) + '_' + \
                       str(param.train_epochs) + '_' + \
                       str(param.batch_size) + '_' + \
                       str(self.features_2_labs.optimizer_name) + '_' + \
                       str(param.learning_rate) + '_' + \
                       str(time.strftime("%m_%d_%H_%M_%S", time.localtime()))
        show_loss(x, y, picture_name, color='r')

    def __save_labs_pos(self, real_labs, predict_labs):
        with open((os.getcwd() + '//MLL_Save_Csv//' + 'real_labs_pos_' +
                       self.mll_data_type + '_' + str(time.strftime("%M_%S", time.localtime()))
                       + '.csv'), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for list in real_labs:
                csv_writer.writerow(list)

        with open((os.getcwd() + '//MLL_Save_Csv//' + 'predict_labs_pos_' +
                       self.mll_data_type + '_' + str(time.strftime("%M_%S", time.localtime()))
                       + '.csv'), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for list in predict_labs:
                csv_writer.writerow(list)

    def __save_labs_neg(self, real_labs, predict_labs):
        with open((os.getcwd() + '//MLL_Save_Csv//' + 'real_labs_neg_' +
                       self.mll_data_type + '_'+ str(time.strftime("%M_%S", time.localtime()))
                       + '.csv'), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for list in real_labs:
                csv_writer.writerow(list)

        with open((os.getcwd() + '//MLL_Save_Csv//' + 'predict_labs_neg_' +
                       self.mll_data_type + '_' + str(time.strftime("%M_%S", time.localtime()))
                       + '.csv'), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for list in predict_labs:
                csv_writer.writerow(list)

    def train(self):
        train_data_loader = DataLoader(self.feaChain_pos_neg_train, param.batch_size, True, drop_last=False)

        pos_lab_show_y = []  # 收集损失
        neg_lab_show_y = []
        for epoch in range(param.train_epochs):
            for step, batch_data in enumerate(train_data_loader):
                feachain_vars, feachain_lens, poslabs_vars, poslabs_lens, neglabs_vars, neglabs_lens = \
                    self.data_obj.batch_2_tensor(batch_data)  # 构建特征链，正标记，负标记变量用于训练

                # # 特征链编码器-正标记解码器
                # pos_lab_loss = self.features_2_labs.features_chain_2_pos_labs_train(
                #     feachain_vars, poslabs_vars, poslabs_lens, self.labels_num)
                # pos_lab_show_y.append(round(pos_lab_loss, 4))

                # 特征链编码器-负标记解码器
                neg_lab_loss = self.features_2_labs.features_chain_2_neg_labs_train(
                    feachain_vars, neglabs_vars, neglabs_lens, self.labels_num)
                neg_lab_show_y.append(round(neg_lab_loss, 4))

                # 特征链编码器-正标记解码器
                pos_lab_loss = self.features_2_labs.features_chain_2_pos_labs_train(
                    feachain_vars, poslabs_vars, poslabs_lens, self.labels_num)
                pos_lab_show_y.append(round(pos_lab_loss, 4))

                print(
                    'Epoch: %2.0f' % epoch,
                    '| Step: %4.0f' % step,
                    '| Batch_size: %3.0f' % len(batch_data[0]),
                    '| Positive Labels Loss: %5.4f' % round(pos_lab_loss, 4),
                    '| Negative Labels Loss: %5.4f' % round(neg_lab_loss, 4)
                )

        pos_lab_show_x = [i for i in range(len(pos_lab_show_y))]
        neg_lab_show_x = [i for i in range(len(neg_lab_show_y))]
        if self.save_loss_fig:
            self.__save_loss_pictures_pos(pos_lab_show_x, pos_lab_show_y)
            self.__save_loss_pictures_neg(neg_lab_show_x, neg_lab_show_y)

    def evaluate(self, data):
        evaluate_data_loader = DataLoader(data, 1, True, drop_last=False)  # 一个一个预测新示例的标记

        all_predict_pos_labs = []
        all_predict_pos_labs_p = []
        all_real_pos_labs = []

        all_predict_neg_labs = []
        all_predict_neg_labs_p = []
        all_real_neg_labs = []

        for epoch in range(1):
            for one_instance in evaluate_data_loader:
                feachain_vars, feachain_lens, poslabs_vars, poslabs_lens, neglabs_vars, neglabs_lens = \
                self.data_obj.batch_2_tensor(one_instance)  # 构建特征链，正标记，负标记变量用于训练

                # 收集正确的标记
                all_real_pos_labs.append(poslabs_vars.transpose(0, 1).cpu().numpy().tolist()[0][:-1])
                all_real_neg_labs.append(neglabs_vars.transpose(0, 1).cpu().numpy().tolist()[0][:-1])

                # 预测的正标记
                decoder_pos_labs, decoder_pos_lab_p = self.features_2_labs.features_chain_2_pos_labs_val(
                    feachain_vars, self.labels_num)
                all_predict_pos_labs.append(decoder_pos_labs)
                all_predict_pos_labs_p.append(decoder_pos_lab_p)

                # 预测的负标记
                decoder_neg_labs, decoder_neg_lab_p = self.features_2_labs.features_chain_2_neg_labs_val(
                    feachain_vars, self.labels_num)
                all_predict_neg_labs.append(decoder_neg_labs)
                all_predict_neg_labs_p.append(decoder_neg_lab_p)

        # 标记变换
        real_pos_labs, predict_pos_labs, real_neg_labs, predict_neg_labs = real_predict_labs_change(
            all_real_pos_labs, all_predict_pos_labs, all_real_neg_labs, all_predict_neg_labs,
            self.labels_num)

        self.__save_labs_pos(real_pos_labs, predict_pos_labs)
        self.__save_labs_neg(real_neg_labs, predict_neg_labs)

        # 评估预测的标记
        evaluate(real_pos_labs, predict_pos_labs, real_neg_labs, predict_neg_labs)


def main():
    lsgm = LSGM('emotions', 'descend', 10, 1, True)
    lsgm.train()
    print('验证数据集，小数据集！')
    lsgm.evaluate(lsgm.feaChain_pos_neg_val)
    print('测试数据集，大数据集！')
    lsgm.evaluate(lsgm.feaChain_pos_neg_test)


if __name__ == '__main__':
    main()