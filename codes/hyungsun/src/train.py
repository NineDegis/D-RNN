import torch.optim as optim
import time
from data import *
from model import *
import glob
import code
import random
from config import *
# from tensorBoardLogger import *

config = ConfigManager(RnnImdb.__class__.__name__).load()
acl_imdb = ACLIMDB(batch_size=1, word_embedding='CBOW', is_eval=False, test_mode=True)
pretrained = torch.FloatTensor(acl_imdb.data.embedding_model.wv.vectors)

lstm = RnnImdb(pretrained)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.001, weight_decay=0.0003)
optimizer.zero_grad()

def train_lstm_imdb(data, target):
    hidden, cell = lstm.init_hidden()
    for i in range(data.size()[0]):
        lstm_data = data[i].view(1,1,100)
        output, hidden, cell = lstm(lstm_data, hidden, cell)

    loss = loss_func(output, target)
    loss.backward()
    optimizer.step()

    return output, loss.item()

def main():
    print("Start training!!")
    num_files = 5
    for i in range(10000):
        sum_loss = 0
        for batch_idx, (data, target) in enumerate(acl_imdb.load()):
            input = torch.FloatTensor(data)

            if int(target[0]) >= 6:
                label = torch.tensor([1])  # positive
            else:
                label = torch.tensor([0])  # negative
#           input, label = input.to(torch.device("cuda")), label.to(torch.device("cuda"))
            output, loss = train_lstm_imdb(input, label)
            sum_loss += loss

            if i % 1 != 0:
                continue
#            if (batch_idx % 100 != 0):
#            if batch_idx != num_files * 2 - 1:
#                continue
            print("epoch:", i)
            print("batch idx:", batch_idx)
            print("label:", label)
            print("output:", output)
            print("mean loss:", sum_loss / (batch_idx + 1))
            print()

if __name__ == "__main__":
    main()
    print("End")

