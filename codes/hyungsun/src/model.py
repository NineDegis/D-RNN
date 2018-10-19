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
    def __init__(self, pretrained):
        super(RnnImdb, self).__init__()
        config = ConfigManager(self.__class__.__name__).load()
        self.embed_size = int(config["EMBED_SIZE"])  # embed size -> 100
        self.hidden_size = int(config['HIDDEN_SIZE'])  # hidden_size -> 100
        self.output_size = int(config["OUTPUT_SIZE"])  # output_size -> 2
        self.batch_size = int(config["BATCH_SIZE"])  # batch size -> 1
        self.num_layers = int(config["NUM_LAYERS"])  # default -> 1

        self.embed = nn.Embedding.from_pretrained(pretrained)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, hidden, cell):
        # TODO(hyungsun): Make this properly.
        # embed = self.embed(torch.tensor([[1]]))
        out, (hidden, cell) = self.lstm(inputs, (hidden, cell))
        linear = self.linear(out.view(1, -1))  # 1->batch_size
        output = self.softmax(linear)
        return output, hidden, cell

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        cell = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        return hidden, cell


