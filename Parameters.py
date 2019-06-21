#-*-coding:utf-8-*-

# 参数类

import torch
from torch import optim

import re


class Parameters(object):

    def __init__(self):
        self.use_gpu = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_epochs = 20
        self.learning_rate = 0.0001  # 0.00001
        self.batch_size = 64  # 16, 32, 64

        self.PAD_TOKEN = 0
        self.SOS_TOKEN = 1
        self.EOS_TOKEN = 2


    def set_optimizer(self, object, learning_rate):
        self.optimizer = optim.RMSprop(object.parameters(), lr=learning_rate)
        self.optimizer_name = str(self.optimizer).split(' ')[0]

        return self.optimizer, self.optimizer_name
