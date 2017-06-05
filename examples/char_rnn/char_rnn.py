# Special thanks to Kyle McDonald, this is based on his example
# https://gist.github.com/kylemcdonald/2d06dc736789f0b329e11d504e8dee9f
# Thanks to Laurent Dinh for examples of parameter saving/loading in PyTorch
from torch.autograd import Variable
import torch.nn as nn
import torch

import numpy as np
import time
import math
import os
import argparse

from pthbldr.core import TrainingLoop

use_cuda = torch.cuda.is_available()

# try to get deterministic runs
torch.manual_seed(1999)
random_state = np.random.RandomState(1999)

# from https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt
seq_length = 50
minibatch_size = 50
hidden_size = 128
n_epochs = 10
n_layers = 2
lr = 2e-3
input_filename = "tiny-shakespeare.txt"
with open(input_filename, "r") as f:
    text = f.read()

param_path = "params.npz"
final_param_path = "params_final.npz"

chars = set(text)
chars_len = len(chars)
char_to_index = {}
index_to_char = {}
for i, c in enumerate(chars):
    char_to_index[c] = i
    index_to_char[i] = c

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def chunks(l, n):
    #print(list(chunks(range(11), 3)))
    for i in range(0, len(l) - n, n):
        yield l[i:i + n]

def index_to_tensor(index):
    tensor = torch.zeros(1, 1).long()
    tensor[0,0] = index
    return Variable(tensor)


# convert all characters to indices
batches = [char_to_index[char] for char in text]

# chunk into sequences of length seq_length + 1
batches = list(chunks(batches, seq_length + 1))

# chunk sequences into batches
batches = list(chunks(batches, minibatch_size))

# convert batches to tensors and transpose
batches = [torch.LongTensor(batch).transpose_(0, 1) for batch in batches]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, batch_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.minibatch_size = minibatch_size

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.cells = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input)
        output, hidden = self.cells(input, hidden)
        output = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return output, hidden

    def create_hidden(self):
        # should this be small random instead of zeros
        # should this also be stored in the class rather than being passed around?
        return torch.zeros(self.n_layers, self.minibatch_size, self.hidden_size)


model = RNN(chars_len, hidden_size, chars_len, n_layers, minibatch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()
hidden = Variable(model.create_hidden())

if use_cuda:
    model = model.cuda()
    hidden = hidden.cuda()


def train_loop(itr, extra):
    batch_tensor = next(itr)
    if use_cuda:
        batch_tensor = batch_tensor.cuda()

    # reset the model
    model.zero_grad()

    # everything except the last
    input_variable = Variable(batch_tensor[:-1])

    # everything except the first, flattened
    target_variable = Variable(batch_tensor[1:].contiguous().view(-1))

    # prediction and calculate loss
    output, _ = model(input_variable, hidden)
    loss = loss_function(output, target_variable)

    # backprop and optimize
    loss.backward()
    optimizer.step()

    loss = loss.data[0]
    return [loss]


def valid_loop(itr, extra):
    batch_tensor = next(itr)
    if use_cuda:
        batch_tensor = batch_tensor.cuda()

    # reset the model
    model.zero_grad()

    # everything except the last
    input_variable = Variable(batch_tensor[:-1])

    # everything except the first, flattened
    target_variable = Variable(batch_tensor[1:].contiguous().view(-1))

    # prediction and calculate loss
    output, _ = model(input_variable, hidden)
    loss = loss_function(output, target_variable)

    # backprop and optimize
    #loss.backward()
    #optimizer.step()

    loss = loss.data[0]
    return [loss]


class minitr:
    def __init__(self, list_of_data, random_state):
        self.data = list_of_data
        self.idx = 0

    def next(self):
        if self.idx >= (len(self.data) - 1):
            self.reset()
            raise StopIteration("End of epoch")
        d = self.data[self.idx]
        self.idx += 1
        return d

    def  __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random_state.shuffle(self.data)


train_len = int(.9 * len(batches))
random_state = np.random.RandomState(2177)
train_itr = minitr(batches[:train_len], random_state)
valid_itr = minitr(batches[train_len:], random_state)

checkpoint_dict = {}

TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=n_epochs,
                  checkpoint_every_n_seconds=60 * 60,
                  checkpoint_every_n_epochs=n_epochs // 10,
                  checkpoint_dict=checkpoint_dict,
                  skip_minimums=True,
                  skip_most_recents=False)
results = TL.run()
