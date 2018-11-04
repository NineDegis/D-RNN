import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
from config import ConfigRNN
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(nn.Module):
    """TODO(hyungsun): Let model classes have optimizer and loss function.
    """
    config = ConfigRNN.instance()

    def __init__(self, pretrained=None):
        super(RNN, self).__init__()
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')
        if pretrained is None:
            self.embed = nn.Embedding(self.config.VOCAB_SIZE, self.config.EMBED_SIZE)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained)
        self.lstm = nn.LSTM(self.config.EMBED_SIZE, self.config.HIDDEN_SIZE)
        self.linear = nn.Linear(self.config.HIDDEN_SIZE, 128)
        self.linear2 = nn.Linear(128, self.config.OUTPUT_SIZE)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, target):
        # 문장 길이를 쉽게 뽑아내기 위해 전처리로 permute 와 squeeze를 한다.
        _inputs = inputs.permute(1, 0, 2).squeeze(2)
        input_lengths = torch.LongTensor([torch.max(_inputs[i, :].data.nonzero()) + 1 for i in range(_inputs.size(0))])
        input_lengths, sorted_idx = input_lengths.sort(0, descending=True)

        # 워드 인덱스 텐서가 문장 길이가 긴 순서대로 정렬된다.
        input_seq2idx = _inputs[sorted_idx]

        # 문장이 정렬된 순서에 맞게 Target 값도 정렬한다.
        sorted_target = target[sorted_idx]

        # 워드 인덱스 텐서를 원드 벡터 텐서로 변환.
        embeded = self.embed(input_seq2idx)

        # valid한 hidden을 뽑기 위해 워드 인덱스를 패킹한다.
        packed_input = pack_padded_sequence(embeded, input_lengths, batch_first=True)

        # packed_input 으로 뽑아낸 hidden은 padding 을 전부 스킵한다.
        packed_output, (hidden, cell) = self.lstm(packed_input, (self.init_hidden()))

        # hidden은 output 과 같기 때문에 packed_output을 다시 pad 해서 넣는 것이 아닌, hidden을 넣는다.
        linear = self.linear(hidden)
        linear = self.linear2(linear)

        # Soft max.
        output = self.softmax(linear.squeeze())

        return output, hidden, cell, sorted_target
    
    def init_hidden(self):
        hidden = Variable(torch.zeros(1, self.config.BATCH_SIZE, self.config.HIDDEN_SIZE))
        cell = Variable(torch.zeros(1, self.config.BATCH_SIZE, self.config.HIDDEN_SIZE))
        if self.cuda_available:
            hidden = hidden.cuda()
            cell = cell.cuda()

        return hidden, cell
