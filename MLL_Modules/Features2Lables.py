#-*-coding:utf-8-*-

# 从特征链生成标记的模型

import random

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from Parameters import Parameters
param = Parameters()


class Features2Labels(object):

    def __init__(self, feature_chain_encoder, pos_lab_decoder, neg_lab_decoder):
        self.feature_chain_encoder = feature_chain_encoder
        self.pos_lab_decoder=pos_lab_decoder
        self.neg_lab_decoder = neg_lab_decoder
        self.encoder_optimizer, self.optimizer_name = \
            param.set_optimizer(feature_chain_encoder, param.learning_rate)
        self.pos_lab_decoder_optimizer, _ = param.set_optimizer(pos_lab_decoder, param.learning_rate)
        self.neg_lab_decoder_optimizer, _ = param.set_optimizer(neg_lab_decoder, param.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.use_gpu = torch.cuda.is_available()

    def sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)

        # seq_range = torch.range(0, max_len - 1).long()
        seq_range = torch.arange(0, max_len).long()

        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand

    def masked_cross_entropy(self, logits, target, length):
        length = Variable(torch.LongTensor(length)).cuda()

        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """

        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        # log_probs_flat = functional.log_softmax(logits_flat)
        log_probs_flat = F.log_softmax(logits_flat, dim=1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        mask = self.sequence_mask(sequence_length=length, max_len=target.size(1))
        losses = losses * mask.float()
        loss = losses.sum() / length.float().sum()
        return loss

    def features_chain_2_pos_labs_train(self, input_features_batches, target_labels_batches,
                                        target_labels_lengths, labels_num):
        batch_size = input_features_batches.shape[1]  # 获取批大小

        self.encoder_optimizer.zero_grad()  # 编码器梯度清零
        self.pos_lab_decoder_optimizer.zero_grad()  # 解码器梯度清零

        # 收集编码器的全部输出和最后的隐藏状态
        encoder_outputs, encoder_state = self.feature_chain_encoder(input_features_batches, None)

        decoder_input = Variable(torch.LongTensor([param.SOS_TOKEN] * batch_size))  # 解码器的初始输入是SOS开始符

        # 解码器的初始输入=编码器的最后隐藏状态
        decoder_hidden = (encoder_state[0][:self.pos_lab_decoder.rnn_layers],
                          encoder_state[1][:self.pos_lab_decoder.rnn_layers])

        max_target_length = max(target_labels_lengths)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size,
                                                   self.pos_lab_decoder.target_labels_num))

        if self.use_gpu:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()

        for t in range(max_target_length):
            decoder_output, decoder_hidden, outputs_softmax = self.pos_lab_decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output  # 收集解码器的全部输出

            # teacher_forcing_ratio = 0.5
            # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            use_teacher_forcing = False

            if use_teacher_forcing:
                top_values, top_indexes = decoder_output.data.topk(1)
                ni = [int(top_indexes[i][0].cpu().numpy()) for i in range(top_indexes.shape[0])]
                decoder_input = Variable(torch.LongTensor(ni))
                if self.use_gpu:
                    decoder_input = decoder_input.cuda()
            else:
                decoder_input = target_labels_batches[t]

        loss = self.masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_labels_batches.transpose(0, 1).contiguous(),
            target_labels_lengths
        )
        loss.backward()

        clip_value = 5.0
        _ = clip_grad_norm_(self.feature_chain_encoder.parameters(), clip_value)
        _ = clip_grad_norm_(self.pos_lab_decoder.parameters(), clip_value)

        self.encoder_optimizer.step()
        self.pos_lab_decoder_optimizer.step()

        return loss.item()

    def features_chain_2_neg_labs_train(self, input_features_batches, target_labels_batches,
                                        target_labels_lengths, labels_num):
        batch_size = input_features_batches.shape[1]  # 获取批大小

        self.encoder_optimizer.zero_grad()  # 编码器梯度清零
        self.neg_lab_decoder_optimizer.zero_grad()  # 解码器梯度清零

        # 收集编码器的全部输出和最后的隐藏状态
        encoder_outputs, encoder_state = self.feature_chain_encoder(input_features_batches, None)

        decoder_input = Variable(torch.LongTensor([param.SOS_TOKEN] * batch_size))  # 解码器的初始输入是SOS开始符

        # 解码器的初始输入=编码器的最后隐藏状态
        decoder_state = (encoder_state[0][:self.neg_lab_decoder.rnn_layers],
                          encoder_state[1][:self.neg_lab_decoder.rnn_layers])

        max_target_length = max(target_labels_lengths)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size,
                                                   self.neg_lab_decoder.target_labels_num))

        if self.use_gpu:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()

        for t in range(max_target_length):
            decoder_output, decoder_hidden, outputs_softmax = self.neg_lab_decoder(
                decoder_input, decoder_state, encoder_outputs)
            all_decoder_outputs[t] = decoder_output  # 收集解码器的全部输出

            # teacher_forcing_ratio = 0.5
            # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            use_teacher_forcing = False

            if use_teacher_forcing:
                top_values, top_indexes = decoder_output.data.topk(1)
                ni = [int(top_indexes[i][0].cpu().numpy()) for i in range(top_indexes.shape[0])]
                decoder_input = Variable(torch.LongTensor(ni))
                if self.use_gpu:
                    decoder_input = decoder_input.cuda()
            else:
                decoder_input = target_labels_batches[t]

        loss = self.masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_labels_batches.transpose(0, 1).contiguous(),
            target_labels_lengths
        )
        loss.backward()

        clip_value = 5.0
        _ = clip_grad_norm_(self.feature_chain_encoder.parameters(), clip_value)
        _ = clip_grad_norm_(self.neg_lab_decoder.parameters(), clip_value)

        self.encoder_optimizer.step()
        self.neg_lab_decoder_optimizer.step()

        return loss.item()

    def features_chain_2_pos_labs_val(self, input_features_batches, labels_num):
        # input_lengths = input_features_batches.shape[0]

        self.feature_chain_encoder.train(False)
        self.pos_lab_decoder.train(False)

        encoder_outputs, encoder_state = self.feature_chain_encoder(input_features_batches, None)

        decoder_input = Variable(torch.LongTensor([param.SOS_TOKEN]))
        decoder_state = (encoder_state[0][:self.pos_lab_decoder.rnn_layers],
                         encoder_state[1][:self.pos_lab_decoder.rnn_layers])

        if param.use_gpu:
            decoder_input = decoder_input.cuda()

        decoder_pos_labs = []
        decoder_pos_labs_p = []

        for i in range(labels_num):
            decoder_output, decoder_state, outputs_softmax = self.pos_lab_decoder(
                decoder_input, decoder_state, encoder_outputs)

            top_values, top_indexes = decoder_output.data.topk(1)
            ni = top_indexes[0][0]

            if ni == param.EOS_TOKEN:
                decoder_pos_labs.append(param.EOS_TOKEN)
                break
            else:
                decoder_pos_labs.append(ni)
                decoder_pos_labs_p.append(round(outputs_softmax[0][ni].cpu().detach().numpy().tolist(), 3))

            decoder_input = Variable(torch.LongTensor([ni]))
            if param.use_gpu:
                decoder_input = decoder_input.cuda()

        self.feature_chain_encoder.train(True)
        self.pos_lab_decoder.train(True)

        pos_labs = [int(i.cpu().numpy())
                    if i != param.EOS_TOKEN else i
                    for i in decoder_pos_labs]

        return pos_labs, decoder_pos_labs_p

    def features_chain_2_neg_labs_val(self, input_features_batches, labels_num):
        # input_lengths = input_features_batches.shape[0]

        self.feature_chain_encoder.train(False)
        self.neg_lab_decoder.train(False)

        encoder_outputs, encoder_state = self.feature_chain_encoder(input_features_batches, None)

        decoder_input = Variable(torch.LongTensor([param.SOS_TOKEN]))
        decoder_state = (encoder_state[0][:self.neg_lab_decoder.rnn_layers],
                         encoder_state[1][:self.neg_lab_decoder.rnn_layers])

        if param.use_gpu:
            decoder_input = decoder_input.cuda()

        decoder_neg_labs = []
        decoder_neg_labs_p = []

        for i in range(labels_num):
            decoder_output, decoder_state, outputs_softmax = self.neg_lab_decoder(
                decoder_input, decoder_state, encoder_outputs)

            top_values, top_indexes = decoder_output.data.topk(1)
            ni = top_indexes[0][0]

            if ni == param.EOS_TOKEN:
                decoder_neg_labs.append(param.EOS_TOKEN)
                break
            else:
                decoder_neg_labs.append(ni)
                decoder_neg_labs_p.append(round(outputs_softmax[0][ni].cpu().detach().numpy().tolist(), 3))

            decoder_input = Variable(torch.LongTensor([ni]))
            if param.use_gpu:
                decoder_input = decoder_input.cuda()

        self.feature_chain_encoder.train(True)
        self.neg_lab_decoder.train(True)

        neg_labs = [int(i.cpu().numpy())
                    if i != param.EOS_TOKEN else i
                    for i in decoder_neg_labs ]

        return neg_labs, decoder_neg_labs_p


# 原始训练代码
lab_num = 10
def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=lab_num):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder 这里是编码器批概念信息
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, bat, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)

        all_decoder_outputs[t] = decoder_output
        # 下一个输入使用的当前实际的目标，即没有使用教师强制训练机制，能更快收敛，但可能造成测试效果不好
        # decoder_input = target_batches[t]  # Next input is current target
        # print(len(decoder_input), decoder_input)
        # exit()

        #################### 这里改为交替使用教师强制训练机制 #################################
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing: # 如果使用教师强制训练机制
            topv, topi = decoder_output.data.topk(1)
            ni = [int(topi[i][0].cpu().numpy()) for i in range(topi.shape[0])] # 存在着优化的方法
            decoder_input = Variable(torch.LongTensor(ni)) # 下一个输入的词是当前预测出来的词
            if USE_CUDA: decoder_input = decoder_input.cuda()
        else: # 不使用教师强制训练机制
            decoder_input = target_batches[t]

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    # return loss.data[0], ec, dc
    return loss.item(), ec, dc
