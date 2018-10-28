import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
from config import ConfigRNN


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
    config = ConfigRNN.instance()

    def __init__(self, pretrained=None):
        super(RNN, self).__init__()
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')
        if pretrained == None:
            self.embed = nn.Embedding(self.config.VOCAB_SIZE, self.config.EMBED_SIZE)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained)
        self.lstm = nn.LSTM(self.config.EMBED_SIZE, self.config.HIDDEN_SIZE)
        self.linear = nn.Linear(self.config.HIDDEN_SIZE, self.config.OUTPUT_SIZE)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        hidden, cell = self.init_hidden()
        embed = self.embed(inputs).squeeze(2)
        out, (hidden, cell) = self.lstm(embed, (hidden, cell))
        linear = self.linear(out)
        output = self.softmax(linear[-1])
        return output, hidden, cell

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, self.config.BATCH_SIZE, self.config.HIDDEN_SIZE))
        cell = Variable(torch.zeros(1, self.config.BATCH_SIZE, self.config.HIDDEN_SIZE))
        if self.cuda_available:
            hidden = hidden.cuda()
            cell = cell.cuda()

        return hidden, cell
