import torch
import torch.nn as nn
from torch.autograd import Variable
import string
import random
import re
import time, math

num_epochs = 10000
print_every = 100
plot_every = 10
chunk_len = 200
embedding_size = 150
hidden_size = 100
batch_size =1
num_layers = 1
lr = 0.002

#Prepare characters (Only English)
all_characters = string.printable
n_characters = len(all_characters)

#Get text data
input_data = []
file = open("../data/shakespeare.txt", "r", encoding="utf-8")
file = file.read()
for line1 in file:
    input_data.append(line1.strip())

file_len = len(file)

#Functions for text processing
def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

#Character to tensor
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()

    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

#Chunk into input & label
def random_training_set():
    chunk = random_chunk()
    input = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return input, target

#Model
class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        self.encoder = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        out = self.encoder(input.view(batch_size, -1))
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.decoder(out.view(batch_size, -1))

        return out, hidden, cell

    def init_hidden(self):
        hidden = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        cell = Variable(torch.zeros(num_layers, batch_size, hidden_size))

        return hidden, cell

model = RNN(n_characters, embedding_size, hidden_size, n_characters, num_layers)

#Loss & Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

#predict
def test():
    start_str = "b"
    inp = char_tensor(start_str)
    hidden,cell = model.init_hidden()
    x = inp

    print(start_str,end="")
    for i in range(200):
        output,hidden,cell = model(x,hidden,cell)

        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]

        print(predicted_char,end="")

        x = char_tensor(predicted_char)


#train
for i in range(num_epochs):
    total = char_tensor(random_chunk())
    inp = total[:-1]    #input
    label = total[1:]   #target

    hidden, cell = model.init_hidden()

    loss = 0
    optimizer.zero_grad()


    for j in range(chunk_len - 1):
        x = inp[[j]]
        y_ = label[[j]]

        y, hidden, cell = model(x, hidden, cell)
        loss += loss_func(y, y_)

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print("\n", loss / chunk_len)
        test()
        print("\n\n")
