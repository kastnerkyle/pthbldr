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
n_epochs = 3  # Used way at the bottom in the training loop!
cut_len = 300  # Used way at the bottom in the training loop!
learning_rate = 1E-5
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
bias = 0.
n_chars = vocabulary_size
n_out = 3
n_in = n_chars

use_cuda = True

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

# need to write this...
def th_normal(shp, scale=0.01):
    if scale >= 1 or scale <= 0:
        print("WARNING: excessive scale {} detected! Function should be called as th_normal((shp0, shp1)), notice parenthesis!")
    return scale * th.randn(*shp)


class GGRUFork(nn.Module):
     def __init__(self, input_size, hidden_size):
         super(GGRUFork, self).__init__()
         self.W = nn.Parameter(th_normal((input_size, 3 * hidden_size)))
         self.b = nn.Parameter(th_normal((3 * hidden_size,)))
         self.input_size = input_size
         self.hidden_size = hidden_size

     def forward(self, inp):
         proj = th.mm(inp, self.W) + self.b[None].expand(inp.size(0), self.b.size(0))
         return proj


class GGRUCell(nn.Module):
    # https://discuss.pytorch.org/t/how-to-define-a-new-layer-with-autograd/351
    def __init__(self, hidden_size, minibatch_size):
        super(GGRUCell, self).__init__()
        self.Wur = nn.Parameter(th_normal((hidden_size, 2 * hidden_size)))
        self.U = nn.Parameter(th_normal((hidden_size, hidden_size)))
        self.hidden_size = hidden_size
        self.minibatch_size = minibatch_size

    def _slice(self, p, d):
        return p[..., d * self.hidden_size:(d + 1) * self.hidden_size]

    def forward(self, inp, previous_state, mask=None):
        state_inp = self._slice(inp, 0)
        gate_inp = inp[..., self.hidden_size:]
        gates = th.sigmoid(th.mm(previous_state, self.Wur) + gate_inp)
        update = gates[..., :self.hidden_size]
        reset = gates[..., self.hidden_size:]

        p = th.mm(state_inp * reset, self.U)
        next_state = th.tanh(p + state_inp)
        next_state_f = (next_state + update) + (previous_state + (1. - update))
        if mask is not None:
            # next_state = mask[:, None] * next_state + (1. - mask[:, None]) * previous_state
            raise ValueError("NYI")
        return next_state_f

    def create_inits(self):
        h_i = th.zeros(self.minibatch_size, self.hidden_size)
        if use_cuda:
            h_i = h_i.cuda()
        return h_i


class GaussianAttention(nn.Module):
    def __init__(self, c_input_size, input_size, hidden_size, n_components, minibatch_size,
                 cell_type="GRU"):
        super(GaussianAttention, self).__init__()
        self.c_input_size = c_input_size
        self.input_size = input_size
        self.n_components = n_components
        self.minibatch_size = minibatch_size
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.att_a = GLinear(self.c_input_size, self.n_components)
        self.att_b = GLinear(self.c_input_size, self.n_components)
        self.att_k = GLinear(self.c_input_size, self.n_components)
        if cell_type == "GRU":
            #self.gru1 = nn.GRUCell(self.input_size, self.hidden_size)
            self.inp_fork = GLinear(self.input_size, self.hidden_size)
            self.fork = GGRUFork(self.input_size, self.hidden_size)
            self.cell = GGRUCell(self.hidden_size, self.minibatch_size)
        else:
            raise ValueError("Unsupported cell_type={}".format(cell_type))

    # 3D at every timestep
    def calc_phi(self, k_t, a_t, b_t, u_c):
        a_t = a_t[:, :, None]
        b_t = b_t[:, :, None]
        k_t = k_t[:, :, None]
        k_t = k_t.expand(k_t.size(0), k_t.size(1), u_c.size(2))
        u_c = u_c.expand(k_t.size(0), k_t.size(1), u_c.size(2))

        a_t = a_t.expand(a_t.size(0), a_t.size(1), u_c.size(2))
        b_t = b_t.expand(b_t.size(0), b_t.size(1), u_c.size(2))

        ss1 = (k_t - u_c) ** 2
        ss2 = -b_t * ss1
        ss3 = a_t * ss2.exp_()
        ss4 = ss3.sum(dim=1)[:, 0, :]
        return ss4

    def forward(self, c_inp, inp, gru_init, att_init):
        ts = inp.size(0)
        cts = c_inp.size(0)
        minibatch_size = inp.size(1)
        hiddens = Variable(th.zeros(ts, minibatch_size, self.input_size))
        att_k = Variable(th.zeros(ts, minibatch_size, self.n_components))
        att_w = Variable(th.zeros(ts, minibatch_size, self.input_size))
        if use_cuda:
            hiddens = hiddens.cuda()
            att_k = att_k.cuda()
            att_w = att_w.cuda()
        #hiddens = []
        #att_k = []
        #att_w = []

        k_tm1 = att_init
        h_tm1 = gru_init
        # input needs to be projected to hidden size and merge with cell...
        # otherwise this is junk

        u = Variable(th.FloatTensor(th.arange(0, cts)))[None, None, :]
        if use_cuda:
            u = u.cuda()

        for i in range(ts):
            inp_t = inp[i]
            a_t = self.att_a(inp_t).exp_()
            b_t = self.att_b(inp_t).exp_()
            att_k_o = self.att_k(inp_t).exp_()
            k_t = k_tm1.expand_as(att_k_o) + att_k_o
            ss4 = self.calc_phi(k_t, a_t, b_t, u)
            ss5 = ss4[:, :, None]
            ss6 = ss5.expand(ss5.size(0), ss5.size(1), c_inp.size(2)) * c_inp.permute(1, 0, 2)
            w_t = ss6.sum(dim=1)[:, 0, :]

            inp_t = self.inp_fork(inp_t)
            f_t = self.fork(w_t + inp_t)
            h_t = self.cell(f_t, h_tm1)

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
            h_i = self.cell.create_inits()
        k_i = th.zeros(self.minibatch_size, self.n_components)
        if use_cuda:
            h_i = h_i.cuda()
            k_i = k_i.cuda()
        return [h_i, k_i]


def softmax(inp, eps=1E-6, axis=1):
    # https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
    input_size = inp.size()

    trans_input = inp.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    #ee = th.exp(input_2d - th.max(input_2d).expand_as(input_2d))
    #soft_max_2d = ee / th.sum(ee + eps).expand_as(ee)
    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)

# https://discuss.pytorch.org/t/gradient-clipping/2836/14
def clip_gradient(model, clip):
    """Clip the gradient"""
    if clip is None:
        return
    totalnorm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.data = p.grad.data.clamp(-clip, clip)


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
        # 1 for beroulli
        # 1 * n_outs * n_components for mean
        # 1 * n_outs * n_components for var
        # 1 * n_components for membership
        # 1 * n_components for corr - note this will be different for hidh dimensional outputs... :|
        # self.n_density = 1 + 6 * self.n_components
        self.n_density = 1 + ((1 + 1) * (n_out - 1) * n_components) + ((1 + 1) * n_components)
        self.minibatch_size = minibatch_size

        self.linp = nn.Embedding(self.n_chars, self.n_hid)
        # n_in is the number of chars, n_out is 3 (pen, X, y)
        self.lproj = GLinear(self.n_out, self.n_hid)
        self.att_l1 = GaussianAttention(self.n_hid, self.n_hid, self.n_hid,
                                        self.n_att_components,
                                        self.minibatch_size)
        self.proj_to_l2 = GGRUFork(self.n_hid, self.n_hid)
        self.proj_to_l3 = GGRUFork(self.n_hid, self.n_hid)
        self.att_to_l2 = GGRUFork(self.n_hid, self.n_hid)
        self.att_to_l3 = GGRUFork(self.n_hid, self.n_hid)
        self.l1_to_l2 = GGRUFork(self.n_hid, self.n_hid)
        self.l1_to_l3 = GGRUFork(self.n_hid, self.n_hid)
        self.l2_to_l3 = GGRUFork(self.n_hid, self.n_hid)
        self.l2 = GGRUCell(self.n_hid, self.minibatch_size)
        self.l3 = GGRUCell(self.n_hid, self.minibatch_size)
        self.loutp1 = GLinear(self.n_hid, self.n_density)
        self.loutp2 = GLinear(self.n_hid, self.n_density)
        self.loutp3 = GLinear(self.n_hid, self.n_density)
        self.poutp = GLinear(self.n_density, self.n_density)

    def _slice_outs(self, outs, corr_scale=0.99):
        k = self.n_components
        mu = outs[..., 0:2*k]
        sigma = outs[..., 2*k:4*k]
        corr = outs[..., 4*k:5*k]
        coeff = outs[..., 5*k:6*k]
        binary = outs[..., 6*k:]
        #binary = th.sigmoid(binary)
        sigma = F.softplus(sigma) + 1E-4
        #sigma = th.exp(sigma.clamp(-10., 3.)) + 1E-4
        # constant offset of 1 to set starting corr to 0?
        # scale it
        corr = th.tanh(corr) * corr_scale
        coeff = softmax(coeff)
        mu = mu.contiguous().view(mu.size()[:-1] + (2, self.n_components))
        sigma = sigma.contiguous().view(sigma.size()[:-1] + (2, self.n_components))
        return mu, sigma, corr, coeff, binary

    def forward(self, c_inp, inp,
                att_gru_init, att_k_init, dec_gru1_init, dec_gru2_init):
        l1_o = self.linp(c_inp)
        lproj_o = self.lproj(inp)
        ts = inp.size(0)
        # inits[0] = att_gru_init
        # inits[1] = att_k_init
        # inits[2] = dec_gru1_init
        # inits[3] = dec_gru2_init

        att_h, att_k, att_w = self.att_l1(l1_o, lproj_o, att_gru_init, att_k_init)
        proj_tm1 = lproj_o[0]
        h2_tm1 = dec_gru1_init
        h3_tm1 = dec_gru2_init

        hiddens = [Variable(th.zeros(ts, minibatch_size, self.n_hid)) for i in range(3)]
        if use_cuda:
            hiddens = [h.cuda() for h in hiddens]

        for i in range(ts):
            proj_t = lproj_o[i]
            h1_t = att_h[i]
            w_t = att_w[i]
            k_t = att_k[i]

            inp_f_l2 = self.proj_to_l2(proj_t)
            inp_f_l3 = self.proj_to_l3(proj_t)

            att_f_l2 = self.att_to_l2(w_t)
            att_f_l3 = self.att_to_l3(w_t)

            l1_f_l2 = self.l1_to_l2(h1_t)
            l1_f_l3 = self.l1_to_l3(h1_t)

            h2_t = self.l2(inp_f_l2 + att_f_l2 + l1_f_l2, h2_tm1)

            l2_f_l3 = self.l2_to_l3(h2_t)

            h3_t = self.l3(inp_f_l3 + att_f_l3 + l1_f_l3 + l2_f_l3, h3_tm1)

            h2_tm1 = h2_t
            h3_tm1 = h3_t

            # adding hiddens over activation
            hiddens[0][i] = hiddens[0][i] + self.att_l1.cell._slice(h1_t, 0)
            hiddens[1][i] = hiddens[1][i] + self.l2._slice(h2_t, 0)
            hiddens[2][i] = hiddens[2][i] + self.l3._slice(h3_t, 0)
        output = self.loutp1(hiddens[0]) + self.loutp2(hiddens[1]) + self.loutp3(hiddens[2])
        poutput = self.poutp(output)
        mu, sigma, corr, coeff, binary = self._slice_outs(poutput)
        return [mu, sigma, corr, coeff, binary] + hiddens + [att_w, att_k]

    def create_inits(self):
        l2_h = th.zeros(self.minibatch_size, self.n_hid)
        l3_h = th.zeros(self.minibatch_size, self.n_hid)
        if use_cuda:
             l2_h = l2_h.cuda()
             l3_h = l3_h.cuda()
        return self.att_l1.create_inits() + [l2_h, l3_h]

# https://github.com/pytorch/pytorch/issues/2591
def logsumexp(inputs, dim=None):
    return (inputs - F.log_softmax(inputs)).sum(dim=dim)

# https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/18
class BernoulliAndBivariateGMM(nn.Module):
    def __init__(self, epsilon=1E-6, exp_clamp_lim=20):
        super(BernoulliAndBivariateGMM, self).__init__()
        self.epsilon = epsilon
        self.exp_clamp_lim = exp_clamp_lim

    def forward(self, true, mu, sigma, corr, coeff, binary):
        # true is L, B, 3
        # mu is output of linear
        # sigma is output of softplus / exp
        # corr is output of tanh
        # coeff is output of softmax
        # binary is output of linear (logit)
        mu1 = mu[:, :, 0, :]
        mu2 = mu[:, :, 1, :]
        sigma1 = sigma[:, :, 0, :]
        sigma2 = sigma[:, :, 1, :]
        t0 = true[:, :, 0][:, :, None]
        t1 = true[:, :, 1][:, :, None].expand_as(mu1)
        t2 = true[:, :, 2][:, :, None].expand_as(mu2)

        normalizer = 1. / (2. * 3.14159 * sigma1 * sigma2 * th.sqrt(1. - corr ** 2))
        Z1 = ((t1 - mu1) / sigma1) ** 2
        Z2 = ((t2 - mu2) / sigma2) ** 2
        # expansion of Z12
        pp1 = 2 * corr * t1 * t2
        pp2 = -2 * corr * mu1 * t2
        pp3 = -2 * corr * mu2 * t1
        pp4 = 2 * corr * mu1 * mu2
        denom = 2 * (1. - corr ** 2) / (sigma1 * sigma2)
        Z12 = pp1 * denom + pp2 * denom + pp3 * denom + pp4 * denom

        part = th.exp(-Z1) * th.exp(-Z2)
        zm = Z12.max().expand_as(Z12)
        tt = th.exp(Z12 - Z12.max().expand_as(Z12))
        full = part * tt / zm

        gprob = normalizer * full
        fgprob = gprob + self.epsilon
        fcoeff = coeff + self.epsilon
        fgprob = fgprob / fgprob.sum(dim=2).expand_as(gprob)
        fcoeff = fcoeff / fcoeff.sum(dim=2).expand_as(coeff)

        """
        # original
        # naturally, this is blowing UP!
        Z1 = (((t1 - mu1) / sigma1) ** 2) + (((t2 - mu2) / sigma2) ** 2)
        Z = Z1 - (2. * (corr * (t1 - mu1) * (t2 - mu2)) / (sigma1 * sigma2))
        inner2 = 0.5 * (1. / (1. - corr ** 2 + self.epsilon))
        cost = -(inner1 + (inner2 * Z))
        """
        # Thanks to DWF https://gist.github.com/dwf/b2e1d8d575cb9e7365f302c90d909893
        a, t = binary, t0
        c_b = th.sum(t * F.softplus(-a) + (1. - t) * F.softplus(a), dim=2)
        #c_b = th.sum(t0 * th.log(binary) + (1. - t0) * th.log(1. - binary), dim=2)


        nll1 = -th.log((fcoeff * fgprob).sum(dim=2)[:, :, 0])
        nll2 = c_b
        nll = nll1 - nll2
        return nll

model = Model()

optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = BernoulliAndBivariateGMM()

"""
# test the model
import time
cinp_mb = c_mb.argmax(axis=-1).astype("int64")
cinp_mb_v = Variable(th.LongTensor(cinp_mb))
inp_mb = mb.astype(floatX)
cuts = int(len(inp_mb) / float(cut_len)) + 1
# if it is exact, need 1 less
if (len(inp_mb) % cut_len) == 0:
    cuts = cuts - 1

use_cuda = True
if use_cuda:
    model = model.cuda()
    cinp_mb_v = cinp_mb_v.cuda()

for e in range(10):
    start = time.time()
    for cut_i in range(cuts):
        inits = [Variable(i) for i in model.create_inits()]
        inp_mb_v = Variable(th.FloatTensor(inp_mb[cut_i * cut_len:(cut_i + 1) * cut_len]))

        if use_cuda:
            inits = [i.cuda() for i in inits]
            inp_mb_v = inp_mb_v.cuda()

        o = model(cinp_mb_v, inp_mb_v, inits)
        l_full = loss_function(inp_mb_v,o[0], o[1], o[2], o[3], o[4])
        # sum / mask once adding mask
        l = l_full.mean()
        print("TBPTT step {}".format(cut_i))
        print("loss {}".format(l))
        print("did loss")
        l.backward()
        print("did backward")
        clip_gradient(model, 10.)
        print("did clip")
        optimizer.step()
        print("did step")
    end = time.time()
    print("total time {}".format(end - start))
from IPython import embed; embed(); raise ValueError()
"""

if use_cuda:
    model = model.cuda()

def train_loop(itr, extra):
    mb, mb_mask, c_mb, c_mb_mask = next(itr)

    cinp_mb = c_mb.argmax(axis=-1).astype("int64")
    inp_mb = mb.astype(floatX)
    cuts = int(len(inp_mb) / float(cut_len)) + 1
    # if it is exact, need 1 less
    if (len(inp_mb) % cut_len) == 0:
        cuts = cuts - 1

    # reset the model
    model.zero_grad()

    total_count = 0
    total_loss = 0
    inits = [Variable(i) for i in model.create_inits()]
    att_gru_init = inits[0]
    att_k_init = inits[1]
    dec_gru1_init = inits[2]
    dec_gru2_init = inits[3]
    for cut_i in range(cuts):
        model.zero_grad()
        cinp_mb_v = Variable(th.LongTensor(cinp_mb))
        inp_mb_sub = inp_mb[cut_i * cut_len:(cut_i + 1) * cut_len]
        inp_mb_v = Variable(th.FloatTensor(inp_mb_sub))

        if use_cuda:
            inp_mb_v = inp_mb_v.cuda()
            cinp_mb_v = cinp_mb_v.cuda()
            att_gru_init = att_gru_init.cuda()
            att_k_init = att_k_init.cuda()
            dec_gru1_init = dec_gru1_init.cuda()
            dec_gru2_init = dec_gru2_init.cuda()

        o = model(cinp_mb_v, inp_mb_v,
                  att_gru_init, att_k_init, dec_gru1_init, dec_gru2_init)
        mu, sigma, corr, coeff, binary = o[:5]

        att_w, att_k = o[-2:]
        hiddens = o[5:-2]
        mu = mu[1:]
        sigma = sigma[1:]
        corr = corr[1:]
        coeff = coeff[1:]
        binary = binary[1:]
        target = inp_mb_v[:-1]
        l_full = loss_function(target, mu, sigma, corr, coeff, binary)
        # sum / mask once adding mask
        l = l_full.mean()
        #l.backward()
        # ???? why retain_variables... TBPTT?
        l.backward(retain_variables=False)
        optimizer.step()
        clip_gradient(model, 10.)
        total_count += len(inp_mb_sub)
        total_loss = total_loss + l.cpu().data[0] * len(inp_mb_sub)

        # setup next inits for TBPTT
        #att_k_init = Variable(att_k[-1])
        #att_gru_init = Variable(hiddens[0][-1])
        #dec_gru1_init = Variable(hiddens[1][-1])
        #dec_gru2_init = Variable(hiddens[2][-1])
        att_k_init = Variable(att_k[-1].cpu().data)
        att_gru_init = Variable(hiddens[0][-1].cpu().data)
        dec_gru1_init = Variable(hiddens[1][-1].cpu().data)
        dec_gru2_init = Variable(hiddens[2][-1].cpu().data)
        print("Part:", total_loss / float(total_count))
    print("Final:", total_loss / float(total_count))
    return [total_loss / float(total_count)]

def valid_loop(itr, extra):
    mb, mb_mask, c_mb, c_mb_mask = next(itr)

    cinp_mb = c_mb.argmax(axis=-1).astype("int64")
    inp_mb = mb.astype(floatX)
    cuts = int(len(inp_mb) / float(cut_len)) + 1
    # if it is exact, need 1 less
    if (len(inp_mb) % cut_len) == 0:
        cuts = cuts - 1

    # reset the model
    model.zero_grad()

    total_loss = 0
    total_count = 0
    inits = [Variable(i) for i in model.create_inits()]
    att_gru_init = inits[0]
    att_k_init = inits[1]
    dec_gru1_init = inits[2]
    dec_gru2_init = inits[3]
    for cut_i in range(cuts):
        cinp_mb_v = Variable(th.LongTensor(cinp_mb))
        inp_mb_sub = inp_mb[cut_i * cut_len:(cut_i + 1) * cut_len]
        inp_mb_v = Variable(th.FloatTensor(inp_mb_sub))

        if use_cuda:
            inp_mb_v = inp_mb_v.cuda()
            cinp_mb_v = cinp_mb_v.cuda()
            att_gru_init = att_gru_init.cuda()
            att_k_init = att_k_init.cuda()
            dec_gru1_init = dec_gru1_init.cuda()
            dec_gru2_init = dec_gru2_init.cuda()

        o = model(cinp_mb_v, inp_mb_v,
                  att_gru_init, att_k_init, dec_gru1_init, dec_gru2_init)
        mu, sigma, corr, coeff, binary = o[:5]
        att_w, att_k = o[-2:]
        hiddens = o[5:-2]
        l_full = loss_function(inp_mb_v, mu, sigma, corr, coeff, binary)
        # sum / mask once adding mask
        l = l_full.mean()
        #l.backward()
        # ???? why retain_variables... TBPTT?
        total_count += len(inp_mb_sub)
        total_loss = total_loss + l.cpu().data[0] * len(inp_mb_sub)

        # setup next inits for TBPTT
        #att_k_init = Variable(att_k[-1])
        #att_gru_init = Variable(hiddens[0][-1])
        #dec_gru1_init = Variable(hiddens[1][-1])
        #dec_gru2_init = Variable(hiddens[2][-1])
        att_k_init = att_k[-1]
        att_gru_init = hiddens[0][-1]
        dec_gru1_init = hiddens[1][-1]
        dec_gru2_init = hiddens[2][-1]
        #print("Part:", total_loss / float(total_count))
    #print("Final:", total_loss / float(total_count))
    return [total_loss / float(total_count)]


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
