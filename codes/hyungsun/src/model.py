import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
from config import *


class Cnn(nn.Module):
    """TODO(hyungsun): Let model classes have optimizer and loss function.
    """
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class RnnImdb(nn.Module):
    """TODO(hyungsun): Let model classes have optimizer and loss function.
    """
    def __init__(self):
        super(RnnImdb, self).__init__()
        config = ConfigManager(self.__class__.__name__).load()
        self.hidden_size = config['HIDDEN_SIZE']
        self.embed_size = config["EMBED_SIZE"]
        self.bi_direction = config['BI_DIRECTION']
        self.vocab_size = config["VOCAB_SIZE"]
        self.output_size = config["OUTPUT_SIZE"]
        if self.bi_direction:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, 1, batch_first=True, bidirectional=self.bi_direction)
        self.linear = nn.Linear(self.hidden_size * self.num_directions, self.output_size)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size))
        return hidden, cell

    def forward(self, inputs):
        embed = self.embed(inputs)
        hidden, cell = self.init_hidden(inputs.size(0))
        output, (_, _) = self.lstm(embed, (hidden, cell))
        output = self.linear(output)
        return output
