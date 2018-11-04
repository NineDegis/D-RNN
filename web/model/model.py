import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
from config import ConfigRNN

POSITIVE = "POS"
NEGATIVE = "NEG"


class ReviewParser(nn.Module):
    config = ConfigRNN.instance()

    def __init__(self, pretrained=None):
        super().__init__()
        if pretrained is None:
            self.embed = nn.Embedding(self.config.VOCAB_SIZE, self.config.EMBED_SIZE)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained)
        self.lstm = nn.LSTM(self.config.EMBED_SIZE, self.config.HIDDEN_SIZE)
        self.linear = nn.Linear(self.config.HIDDEN_SIZE, 128)
        self.linear2 = nn.Linear(128, self.config.OUTPUT_SIZE)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        # 워드 인덱스 텐서를 원드 벡터 텐서로 변환.
        embedded = self.embed(inputs)

        output, (_, _) = self.lstm(embedded, (self.init_hidden()))

        linear = self.linear(output)
        linear = self.linear2(linear)

        output = self.softmax(linear[-1])
        print(output[0])
        prediction = output[0].max(0)[1]
        if prediction:
            return POSITIVE
        return NEGATIVE

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, self.config.BATCH_SIZE, self.config.HIDDEN_SIZE))
        cell = Variable(torch.zeros(1, self.config.BATCH_SIZE, self.config.HIDDEN_SIZE))
        if torch.cuda.is_available():
            hidden = hidden.cuda()
            cell = cell.cuda()

        return hidden, cell
