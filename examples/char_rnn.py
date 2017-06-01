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

parser = argparse.ArgumentParser(description="PyTorch char-rnn")
parser.add_argument("--mode", "-m", type=int, default=0,
                    help="0 is evaluate only, 1 is train")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# try to get deterministic runs
torch.manual_seed(1999)
random_state = np.random.RandomState(1999)

# from https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt
seq_length = 50
minibatch_size = 50
hidden_size = 128
epoch_count = 10
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

# each batch is (sequence_length + 1) x batch_size
print(batches[0].size())

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

print_every = 1

model = RNN(chars_len, hidden_size, chars_len, n_layers, minibatch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()
hidden = Variable(model.create_hidden())

if use_cuda:
    model = model.cuda()
    hidden = hidden.cuda()

def train():
    if os.path.exists(param_path):
        print("Parameters found at {}... loading".format(param_path))
        params_val = np.load(param_path)

        for key_, param in model.named_parameters():
            param.data = torch.Tensor(params_val[key_])


    start = time.time()
    all_losses = []

    format_string = \
    """
    Duration: {duration}
    Epoch: {epoch}/{epoch_count}
    Batch: {batch}/{batch_count}, {batch_rate:.2f}/s
    Loss: {loss:.2f}
    """
    try:
        for epoch in range(1, epoch_count + 1):
            d = {key_: val_.data.numpy() for (key_, val_) in model.named_parameters()}
            with open(param_path, "w") as f:
                np.savez(f, **d)

            random_state.shuffle(batches)
            for batch, batch_tensor in enumerate(batches):
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
                all_losses.append(loss)

                if print_every > 0 and batch % print_every == 0:
                    batch_count = len(batches)
                    batch_rate = ((batch_count * (epoch - 1)) + batch) / (time.time() - start)
                    print(format_string.format(duration=time_since(start),
                                               epoch=epoch,
                                               epoch_count=epoch_count,
                                               batch=batch,
                                               batch_count=batch_count,
                                               batch_rate=batch_rate,
                                               loss=loss))

    except KeyboardInterrupt:
        pass

    # final save
    d = {key_: val_.data.numpy() for (key_, val_) in model.named_parameters()}
    with open(final_param_path, "w") as f:
        np.savez(f, **d)


def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    if os.path.exists(final_param_path):
        print("Final parameters found at {}... loading".format(final_param_path))
        params_val = np.load(final_param_path)
        for key_, param in model.named_parameters():
            param.data = torch.Tensor(params_val[key_])
    else:
        raise ValueError("Training was not finalized, no file found at {}. Run with -m 1 first to train a model".format(final_param_path))

    model.minibatch_size = 1
    hidden = Variable(model.create_hidden(), volatile=True)

    if use_cuda:
        hidden = hidden.cuda()

    prime_tensors = [index_to_tensor(char_to_index[char]) for char in prime_str]

    if use_cuda:
        prime_tensors = [tensor.cuda() for tensor in prime_tensors]

    for prime_tensor in prime_tensors[-2:]:
        _, hidden = model(prime_tensor, hidden)

    inp = prime_tensors[-1]
    predicted = prime_str
    for p in range(predict_len):
        if use_cuda:
            inp = inp.cuda()

        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()

        # use numpy - torch output non-deterministic even with seeds
        def rn(x):
            return x / (np.sum(x) + .0001 * np.sum(x))
        s = random_state.multinomial(1, rn(output_dist.numpy()))
        top_i = int(np.where(s == 1)[0])

        # not deterministic even with seed set
        #top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = index_to_char[top_i]
        predicted += predicted_char
        inp = index_to_tensor(char_to_index[predicted_char])
    return predicted


if args.mode == 0:
    print(evaluate('Th', 500, temperature=0.8))
    from IPython import embed; embed(); raise ValueError()
elif args.mode == 1:
    train()
