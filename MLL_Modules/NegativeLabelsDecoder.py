#-*-coding:utf-8-*-

# 负标记解码器

import torch
from torch import nn

from MLL_Modules.Attention import Attn


class NegativeLabelsDecoder(nn.Module):

    def __init__(self, target_labels_num, target_hidden_size, attention_model='dot', labels_embedding=None,
                 rnn_layers=2, rnn_dropout=0):
        """
        :param target_labels_num: 目标正标记的标记数
        :param target_hidden_size: 标记的维度，等于特征的维度
        :param attention_model:
        :param labels_embedding: 预训练好的标记词向量
        :param rnn_layers:
        :param rnn_dropout:
        """
        super(NegativeLabelsDecoder, self).__init__()

        self.target_labels_num = target_labels_num
        self.rnn_layers = rnn_layers
        self.target_hidden_size = target_hidden_size

        self.Embedding_layer = nn.Embedding(target_labels_num, target_hidden_size)
        if labels_embedding:  # 载入训练的特征集
            self.Embedding_layer.weight.data.copy_(torch.from_numpy(labels_embedding))
            self.Embedding_layer.weight.requires_grad = False  # 不更新词向量层

        self.Emedding_Drooput_layer = nn.Dropout(rnn_dropout)

        self.RNN_layer = nn.GRU(
            input_size=target_hidden_size,
            hidden_size=target_hidden_size,
            num_layers=rnn_layers,
            dropout=rnn_dropout)

        self.RNN_layer_lstm = nn.LSTM(
            input_size=target_hidden_size,
            hidden_size=target_hidden_size,
            num_layers=rnn_layers,
            dropout=rnn_dropout
        )

        self.Linear_concat = nn.Linear(target_hidden_size * 2, target_hidden_size)

        self.Linear_out = nn.Linear(target_hidden_size, target_labels_num)

        if attention_model:
            self.Attention_layer = Attn(attention_model, target_hidden_size)

        self.softmax_1 = nn.Softmax(dim=0)
        self.softmax_2 = nn.Softmax(dim=1)

    def forward(self, target_sequences, last_state, encoder_outputs):
        batch_size = target_sequences.size(0)  # 批次大小
        labels_embedding = self.Embedding_layer(target_sequences)  # 获取每一个标记的标记向量
        labels_embedded = self.Emedding_Drooput_layer(labels_embedding)  # 对词向量做dropout处理
        labels_embedded = labels_embedded.view(1, batch_size, self.target_hidden_size)  # 大小变换

        outputs, state = self.RNN_layer_lstm(labels_embedded, last_state)  # 将词向量和隐藏状态送入RNN中计算

        # 计算RNN隐藏层的输出和encoder_outputs的得分，即attention机制，
        attention_weights = self.Attention_layer(outputs, encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        # Luong 公式 5, 级联RNN输出和上下文向量context
        concat_input = torch.cat((outputs.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.Linear_concat(concat_input))

        # Luong 公式 6，生成当前字符的概率，没有使用softmax
        outputs = self.Linear_out(concat_output)

        outputs_1 = self.softmax_1(outputs)
        outputs_2 = self.softmax_2(outputs)

        return outputs, state ,outputs_2