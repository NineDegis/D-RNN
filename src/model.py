import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
from config import *


class CNN(nn.Module):
    """TODO(hyungsun): Let model classes have optimizer and loss function.
    """
    def __init__(self):
        super(CNN, self).__init__()
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


class RNN(nn.Module):
    """TODO(hyungsun): Let model classes have optimizer and loss function.
    """
    def __init__(self, pretrained):
        super(RNN, self).__init__()
        config = ConfigManager(self.__class__.__name__).load()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed_method = str(config["EMBED_METHOD"])
        self.vocab_size = int(config["VOCAB_SIZE"])
        self.embed_size = int(config["EMBED_SIZE"])
        self.hidden_size = int(config['HIDDEN_SIZE'])
        self.output_size = int(config["OUTPUT_SIZE"])
        self.batch_size = int(config["BATCH_SIZE"])

        print(self.embed_method)
        if self.embed_method == "TORCH":
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        hidden, cell = self.init_hidden()
        embed = self.embed(inputs).squeeze(2)
        out, (hidden, cell) = self.lstm(embed, (hidden, cell))
        linear = self.linear(out)
        output = self.softmax(linear[-1])
        return output, hidden, cell

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        cell = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        if torch.cuda.is_available():
            hidden = hidden.cuda()
            cell = cell.cuda()
            
        return hidden, cell
