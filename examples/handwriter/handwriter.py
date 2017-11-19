# Special thanks to Kyle McDonald, this is based on his example
# https://gist.github.com/kylemcdonald/2d06dc736789f0b329e11d504e8dee9f
# Thanks to Laurent Dinh for examples of parameter saving/loading in PyTorch
from torch.autograd import Variable
import torch.nn as nn
import torch as th
import torch.nn.functional as F

import numpy as np
import time
import math
import os
import argparse

import operator
from pthbldr import floatX, intX
from pthbldr import TrainingLoop
from pthbldr import create_checkpoint_dict

from extras import fetch_iamondb, list_iterator

iamondb = fetch_iamondb()
X = iamondb["data"]
y = iamondb["target"]
vocabulary = iamondb["vocabulary"]
vocabulary_size = iamondb["vocabulary_size"]
pen_trace = np.array([x.astype(floatX) for x in X])
chars = np.array([yy.astype(floatX) for yy in y])

minibatch_size = 50
n_epochs = 100  # Used way at the bottom in the training loop!
cut_len = 300  # Used way at the bottom in the training loop!
learning_rate = 1E-4
random_state = np.random.RandomState(1999)

train_itr = list_iterator([pen_trace, chars], minibatch_size, axis=1, stop_index=10000,
                          make_mask=True)
valid_itr = list_iterator([pen_trace, chars], minibatch_size, axis=1, start_index=10000,
                          make_mask=True)

mb, mb_mask, c_mb, c_mb_mask = next(train_itr)
train_itr.reset()

n_hid = 400
n_att_components = 10
n_components = 20
n_chars = vocabulary_size
n_out = 3
n_in = n_chars

use_cuda = th.cuda.is_available()

# try to get deterministic runs
th.manual_seed(1999)
random_state = np.random.RandomState(1999)

class GLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(GLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x_orig = x
        last_axis = x.size(-1)
        x = x.view(-1, last_axis)
        l_o = self.linear(x)
        return l_o.view(*list(x_orig.size())[:-1] + [self.output_size])

# from J Howard
# https://discuss.pytorch.org/t/tip-using-keras-compatible-tensor-dot-product-and-broadcasting-ops/595
def unit_prefix(x, n=1):
    for i in range(n): x = x.unsqueeze(0)
    return x

def align(x, y, start_dim=2):
    xd, yd = x.dim(), y.dim()
    if xd > yd: y = unit_prefix(y, xd - yd)
    elif yd > xd: x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if   ys[td]==1: ys[td] = xs[td]
        elif xs[td]==1: xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)

def aligned_op(x,y,f):
    x, y = align(x, y, 0)
    return f(x, y)

def badd(x, y): return aligned_op(x, y, operator.add)
def bsub(x, y): return aligned_op(x, y, operator.sub)
def bmul(x, y): return aligned_op(x, y, operator.mul)
def bdiv(x, y): return aligned_op(x, y, operator.truediv)


class GaussianAttention(nn.Module):
    def __init__(self, c_input_size, input_size, n_components, minibatch_size,
                 cell_type="GRU"):
        super(GaussianAttention, self).__init__()
        self.c_input_size = c_input_size
        self.input_size = input_size
        self.n_components = n_components
        self.minibatch_size = minibatch_size
        self.att_a = GLinear(self.c_input_size, self.n_components)
        self.att_b = GLinear(self.c_input_size, self.n_components)
        self.att_k = GLinear(self.c_input_size, self.n_components)
        self.cell_type = cell_type
        if cell_type == "GRU":
            self.gru1 = nn.GRUCell(self.input_size, self.input_size)
        else:
            raise ValueError("Unsupported cell_type={}".format(cell_type))

    # 3D at every timestep
    def calc_phi(self, k_t, a_t, b_t, u_c):
        a_t = a_t[:, :, None]
        b_t = b_t[:, :, None]
        k_t = k_t[:, :, None]
        ss1 = bsub(k_t, u_c) ** 2
        ss2 = bmul(-b_t, ss1)
        ss3 = bmul(a_t, ss2.exp_())
        ss4 = ss3.sum(dim=1)[:, 0, :]
        return ss4

    def forward(self, c_inp, inp, gru_init, att_init):
        ts = inp.size(0)
        cts = c_inp.size(0)
        minibatch_size = inp.size(1)
        hiddens = Variable(th.zeros(ts, minibatch_size, self.input_size))
        att_k = Variable(th.zeros(ts, minibatch_size, self.n_components))
        att_w = Variable(th.zeros(ts, minibatch_size, self.input_size))
        #hiddens = []
        #att_k = []
        #att_w = []

        k_tm1 = inits[-1]
        h_tm1 = inits[0]

        u = Variable(th.FloatTensor(th.arange(0, cts)))[None, None, :]
        for i in range(ts):
            inp_t = inp[i]
            a_t = self.att_a(inp_t).exp_()
            b_t = self.att_b(inp_t).exp_()
            att_k_o = self.att_k(inp_t).exp_()
            k_t = k_tm1.expand_as(att_k_o) + att_k_o
            ss4 = self.calc_phi(k_t, a_t, b_t, u)
            ss5 = ss4[:, :, None]
            ss6 = bmul(ss5, c_inp.permute(1, 0, 2))
            w_t = ss6.sum(dim=1)[:, 0, :]

            h_t = self.gru1(w_t, h_tm1)

            att_w[i] = att_w[i] + w_t
            att_k[i] = att_k[i] + k_t
            hiddens[i] = hiddens[i] + h_t

            #att_w.append(w_t)
            #att_k.append(k_t)
            #hiddens.append(h_t)
            h_tm1 = h_t
            k_tm1 = k_t
            w_tm1 = w_t
        return hiddens, att_k, att_w

    def create_inits(self):
        if self.cell_type == "GRU":
            h_i = th.zeros(self.minibatch_size, self.input_size)
        k_i = th.zeros(self.minibatch_size, self.n_components)
        return [h_i, k_i]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.minibatch_size = minibatch_size
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_chars = n_chars
        self.n_att_components = n_att_components
        self.n_components = n_components
        self.n_density = 1 + 6 * self.n_components
        self.minibatch_size = minibatch_size

        self.linp = nn.Embedding(self.n_chars, self.n_hid)
        self.lproj = GLinear(self.n_out, self.n_hid)
        self.att_l1 = GaussianAttention(self.n_hid, self.n_hid,
                                        self.n_att_components,
                                        self.minibatch_size)
        self.l2 = nn.GRU(self.n_hid, self.n_hid)
        self.l3 = nn.GRU(self.n_hid, self.n_hid)
        self.loutp = nn.Linear(self.n_hid, self.n_out)

    def forward(self, c_inp, inp, inits):
        l1_o = self.linp(c_inp)
        lproj_o = self.lproj(inp)
        att_h, att_k, att_w = self.att_l1(l1_o, lproj_o, inits[-2], inits[-1])
        l2_o, l2_h = self.l2(att_h, inits[0][None])
        l3_o, l3_h = self.l3(l2_o, inits[1][None])
        from IPython import embed; embed(); raise ValueError()
        output = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return output + [l2_h, l3_h]

    def create_inits(self):
        l1_h = th.zeros(self.minibatch_size, self.n_hid)
        l2_h = th.zeros(self.minibatch_size, self.n_hid)
        return [l1_h, l2_h] + self.att_l1.create_inits()

model = Model()
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()
inits = [Variable(i) for i in model.create_inits()]

cinp_mb = c_mb.argmax(axis=-1).astype("int64")
cinp_mb_v = Variable(th.LongTensor(cinp_mb))
inp_mb = mb.astype(floatX)
inp_mb_v = Variable(th.FloatTensor(inp_mb))
model(cinp_mb_v, inp_mb_v, inits)

if use_cuda:
    model = model.cuda()
    inits = [i.cuda() for i in inits]

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


class minitr(object):
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


train_len = int(.05 * len(batches))
random_state = np.random.RandomState(2177)
train_itr = minitr(batches[:train_len], random_state)
valid_itr = minitr(batches[train_len:], random_state)


checkpoint_dict, model, optimizer = create_checkpoint_dict(model, optimizer)

TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=n_epochs,
                  checkpoint_every_n_seconds=60 * 60,
                  checkpoint_every_n_epochs=1,
                  checkpoint_dict=checkpoint_dict,
                  skip_minimums=True,
                  skip_most_recents=False)
results = TL.run()
