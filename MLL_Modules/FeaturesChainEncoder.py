#-*-coding:utf-8-*-

# 特征链编码器

import torch
from torch import nn


class FeaturesChainEncoder(nn.Module):

    def __init__(self, input_instances_num, input_hidden_size, features=None,
                 rnn_layers=2, rnn_bidirectional=True, rnn_dropout=0):
        """
        :param input_instances_num: 训练集的示例个数
        :param input_hidden_size: 每个特征的维度\
        :param features:
        :param rnn_layers:
        :param rnn_bidirectional:
        :param rnn_dropout:
        """
        super(FeaturesChainEncoder, self).__init__()

        self.input_hidden_size = input_hidden_size

        self.Embedding_layer = nn.Embedding(input_instances_num, input_hidden_size)
        if features is not None:  # 载入训练的特征集
            self.Embedding_layer.weight.data.copy_(torch.from_numpy(features))
            self.Embedding_layer.weight.requires_grad = False  # 不更新词向量层

        self.RNN_layer_lstm = nn.LSTM(input_size=input_hidden_size,
                                      hidden_size=input_hidden_size,
                                      num_layers=rnn_layers,  # LSTM用两层结构尝试
                                      dropout=rnn_dropout,  # 当前量这么小，应该不需要dropout了
                                      bidirectional=rnn_bidirectional)

    def forward(self, input_sequences, first_state=None):
        train_instances_features = self.Embedding_layer(input_sequences)  # 从特征向量层找出每一个示例的特征

        outputs, state = self.RNN_layer_lstm(train_instances_features, first_state)  # 送入RNN中计算

        outputs = outputs[:, :, :self.input_hidden_size] + \
                  outputs[:, :, self.input_hidden_size:] # 正向和反向叠加

        return outputs, state