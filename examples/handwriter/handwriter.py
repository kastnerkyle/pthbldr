# Special thanks to Kyle McDonald, this is based on his example
# https://gist.github.com/kylemcdonald/2d06dc736789f0b329e11d504e8dee9f
# Thanks to Laurent Dinh for examples of parameter saving/loading in PyTorch
from torch.autograd import Variable
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torch.nn.modules.module import _addindent

import numpy as np
import time
import math
import os
import argparse

import operator
from pthbldr import floatX, intX
from pthbldr import TrainingLoop
from pthbldr import create_checkpoint_dict
from pthbldr import get_cuda, set_cuda

from extras import fetch_iamondb, list_iterator, rsync_fetch

iamondb = rsync_fetch(fetch_iamondb, "leto01")
X = iamondb["data"]
y = iamondb["target"]
vocabulary = iamondb["vocabulary"]
vocabulary_size = iamondb["vocabulary_size"]
pen_trace = np.array([x.astype(floatX) for x in X])
chars = np.array([yy.astype(floatX) for yy in y])

set_cuda(True)

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
bias = 0.
n_chars = vocabulary_size
n_out = 3
n_in = n_chars

# try to get deterministic runs
th.manual_seed(1999)
random_state = np.random.RandomState(1999)

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    global_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            nn.modules.container.Container,
            nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'
        global_params += params
    tmpstr = tmpstr + ')'
    tmpstr += "\n"
    tmpstr += "Total parameters={0:.4f} M".format(global_params / 1E6)
    return tmpstr

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
            next_state_f = mask[:, None].expand_as(next_state_f) * next_state_f + (1. - mask[:, None].expand_as(previous_state)) * previous_state
        return next_state_f

    def create_inits(self):
        h_i = th.zeros(self.minibatch_size, self.hidden_size)
        if get_cuda():
            h_i = h_i.cuda()
        return h_i


# TODO: Change logic to cell based... oy
class GaussianAttentionCell(nn.Module):
    def __init__(self, c_input_size, input_size, hidden_size, n_components, minibatch_size,
                 cell_type="GRU"):
        super(GaussianAttentionCell, self).__init__()
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

    def forward(self, c_inp, inp_t, gru_h_tm1, att_k_tm1, c_mask=None, inp_mask=None):
        cts = c_inp.size(0)
        minibatch_size = c_inp.size(1)

        k_tm1 = att_k_tm1
        h_tm1 = gru_h_tm1
        # input needs to be projected to hidden size and merge with cell...
        # otherwise this is junk

        u = Variable(th.FloatTensor(th.arange(0, cts)))[None, None, :]
        if get_cuda():
            u = u.cuda()
            k_tm1 = att_k_tm1.cuda()
            h_tm1 = gru_h_tm1.cuda()

        # c_mask here...
        a_t = self.att_a(inp_t).exp_()
        b_t = self.att_b(inp_t).exp_()
        att_k_o = self.att_k(inp_t).exp_()
        k_t = k_tm1.expand_as(att_k_o) + att_k_o
        ss4 = self.calc_phi(k_t, a_t, b_t, u)
        ss5 = ss4[:, :, None]
        ss6 = ss5.expand(ss5.size(0), ss5.size(1), c_inp.size(2)) * c_inp.permute(1, 0, 2)
        w_t = ss6.sum(dim=1)[:, 0, :]

        finp_t = self.inp_fork(inp_t)
        f_t = self.fork(w_t + finp_t)
        h_t = self.cell(f_t, h_tm1, inp_mask)
        return h_t, k_t, w_t

    def create_inits(self):
        if self.cell_type == "GRU":
            h_i = self.cell.create_inits()
        k_i = th.zeros(self.minibatch_size, self.n_components)
        if get_cuda():
            h_i = h_i.cuda()
            k_i = k_i.cuda()
        return [h_i, k_i]

def log_softmax(inp, axis=-1):
    # https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
    input_size = inp.size()

    trans_input = inp.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    log_softmax_2d = F.log_softmax(input_2d)

    log_softmax_nd = log_softmax_2d.view(*trans_size)
    tt = log_softmax_nd.transpose(axis, len(input_size)-1)
    return tt

def softmax(inp, eps=1E-6, axis=-1):
    # https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
    if axis != -1:
        raise ValueError("NYI")
    input_size = inp.size()

    trans_input = inp.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    ee = th.exp(input_2d - th.max(input_2d).expand_as(input_2d))
    softmax_2d = ee / (th.sum(ee, dim=-1).expand_as(ee) + eps)
    #softmax_2d = F.softmax(input_2d)
    #esoftmax_2d = softmax_2d
    #softmax_2d = esoftmax_2d / esoftmax_2d.sum(dim=-1).expand_as(esoftmax_2d) + eps

    softmax_nd = softmax_2d.view(*trans_size)
    tt = softmax_nd.transpose(axis, len(input_size)-1)
    return tt


# https://discuss.pytorch.org/t/gradient-clipping/2836/14
def clip_gradient(model, clip):
    """Clip the gradient"""
    if clip is None:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.data = p.grad.data.clamp(-clip, clip)


def norm_gradient(model, rescale):
    """Norm the gradient"""
    if rescale is None:
        return
    grad_norm = None
    for p in model.parameters():
        if p.grad is None:
            continue
        if (p.grad != p.grad).float().sum().data[0] > 0:
            print("WARNING: NaN grad, replacing by {}".format(rescale))
            p.grad[p.grad != p.grad] = rescale
        if grad_norm is None:
            grad_norm = (p.grad.data ** 2).sum()
        else:
            grad_norm += (p.grad.data ** 2).sum()
    grad_norm = np.sqrt(grad_norm)
    scaling_num = rescale
    scaling_den = max([rescale, grad_norm])
    scaling = float(scaling_num) / scaling_den

    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.data = scaling * p.grad.data


# MUST PASS ALL THESE IN
class Model(nn.Module):
    def __init__(self, minibatch_size, n_in, n_hid, n_out, n_chars,
                 n_att_components, n_components, bias=0.):
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
        self.bias_coeff = bias
        self.bias_sigma = bias

        self.linp = nn.Embedding(self.n_chars, self.n_hid)
        # n_in is the number of chars, n_out is 3 (pen, X, y)
        self.lproj = GLinear(self.n_out, self.n_hid)
        self.att_l1 = GaussianAttentionCell(self.n_hid, self.n_hid, self.n_hid,
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
        for m in self.modules():
            if isinstance(m, GLinear):
                sz = m.linear.weight.size()
                m.linear.weight.data = th_normal((sz[0], sz[1]))
                m.linear.bias.data.zero_()
            """
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            """

    def _slice_outs(self, outs, eps=1E-5):
        k = self.n_components
        mu = outs[..., 0:2*k]
        sigma = outs[..., 2*k:4*k]
        corr = outs[..., 4*k:5*k]
        log_coeff = outs[..., 5*k:6*k]
        log_binary = outs[..., 6*k:]
        corr = th.tanh(corr)
        #binary = th.sigmoid(binary)
        #binary = (binary + eps) * (1. - 2 * eps)
        sigma = th.exp(sigma - self.bias_sigma) + 1E-4
        #log_coeff = log_softmax(log_coeff)
        #coeff = softmax(coeff * (1. + self.bias_coeff))
        mu = mu.contiguous().view(mu.size()[:-1] + (2, self.n_components))
        sigma = sigma.contiguous().view(sigma.size()[:-1] + (2, self.n_components))
        return mu, sigma, corr, log_coeff, log_binary
        """
        #binary = th.sigmoid(binary)
        #sigma = th.exp(sigma.clamp(-3., 3.) - self.bias_sigma).clamp(1E-3, 10.)
        #sigma = F.softplus(sigma - self.bias_sigma) + 1E-4
        #sigma = th.exp(sigma.clamp(-10., 3.)) + 1E-4
        sigma = th.exp(sigma.clamp(-10., 10.) - self.bias_sigma).clamp(1E-3, 10.)
        #) constant offset of 1 to set starting corr to 0?
        # scale it
        corr = th.tanh(corr)
        log_coeff = F.log_softmax(lcoeff)
        #coeff = softmax(coeff * (1. + self.bias_coeff))
        mu = mu.contiguous().view(mu.size()[:-1] + (2, self.n_components))
        sigma = sigma.contiguous().view(sigma.size()[:-1] + (2, self.n_components))
        return mu, sigma, corr, coeff, binary
        """

    def forward(self, c_inp, inp, mask_inp,
                att_gru_init, att_k_init, dec_gru1_init, dec_gru2_init):
        lproj_o = self.lproj(inp)
        l1_o = self.linp(c_inp)
        #lproj_o = self.lproj(inp)
        ts = inp.size(0)
        # inits[0] = att_gru_init
        # inits[1] = att_k_init
        # inits[2] = dec_gru1_init
        # inits[3] = dec_gru2_init

        hiddens = [Variable(th.zeros(ts, self.minibatch_size, self.n_hid)) for i in range(3)]
        att_w = Variable(th.zeros(ts, self.minibatch_size, self.n_hid))
        att_k = Variable(th.zeros(ts, self.minibatch_size, self.n_att_components))
        if get_cuda():
            hiddens = [h.cuda() for h in hiddens]
            att_w = att_w.cuda()
            att_k = att_k.cuda()
        att_inits = self.att_l1.create_inits()
        att_gru_h_tm1 = Variable(att_inits[0])
        att_k_tm1 = Variable(att_inits[1])


        for i in range(ts):
            #proj_tm1 = lproj_o[0]
            h2_tm1 = dec_gru1_init
            h3_tm1 = dec_gru2_init

            proj_t = lproj_o[i]
            mask_t = mask_inp[i]
            att_h_t, att_k_t, att_w_t = self.att_l1(l1_o, proj_t, att_gru_h_tm1, att_k_tm1, inp_mask=mask_t)

            h1_t = att_h_t
            w_t = att_w_t
            k_t = att_k_t

            inp_f_l2 = self.proj_to_l2(proj_t)
            inp_f_l3 = self.proj_to_l3(proj_t)

            att_f_l2 = self.att_to_l2(w_t)
            att_f_l3 = self.att_to_l3(w_t)

            l1_f_l2 = self.l1_to_l2(h1_t)
            l1_f_l3 = self.l1_to_l3(h1_t)

            h2_t = self.l2(inp_f_l2 + att_f_l2 + l1_f_l2, h2_tm1, mask=mask_t)

            l2_f_l3 = self.l2_to_l3(h2_t)

            h3_t = self.l3(inp_f_l3 + att_f_l3 + l1_f_l3 + l2_f_l3, h3_tm1, mask=mask_t)

            att_gru_h_tm1 = att_h_t
            att_k_tm1 = att_k_t
            h2_tm1 = h2_t
            h3_tm1 = h3_t

            # adding hiddens over activation
            hiddens[0][i] = hiddens[0][i] + self.att_l1.cell._slice(h1_t, 0)
            hiddens[1][i] = hiddens[1][i] + self.l2._slice(h2_t, 0)
            hiddens[2][i] = hiddens[2][i] + self.l3._slice(h3_t, 0)
            att_w[i] = att_w[i] + att_w_t
            att_k[i] = att_k[i] + att_k_t

        output = self.loutp1(hiddens[0]) + self.loutp2(hiddens[1]) + self.loutp3(hiddens[2])
        poutput = self.poutp(output)
        mu, sigma, corr, log_coeff, log_binary = self._slice_outs(poutput)
        return [mu, sigma, corr, log_coeff, log_binary] + hiddens + [att_w, att_k]

    def create_inits(self):
        l2_h = th.zeros(self.minibatch_size, self.n_hid)
        l3_h = th.zeros(self.minibatch_size, self.n_hid)
        if get_cuda():
             l2_h = l2_h.cuda()
             l3_h = l3_h.cuda()
        return self.att_l1.create_inits() + [l2_h, l3_h]

# from A d B
# https://github.com/adbrebs/handwriting/blob/master/model.py
def logsumexp(inputs, dim=None):
    max_i = inputs.max(dim=dim)[0]
    z = th.log(th.sum(th.exp(inputs - max_i.expand_as(inputs)), dim=dim)) + max_i
    return z


# https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/18
class BernoulliAndBivariateGMM(nn.Module):
    def __init__(self):
        super(BernoulliAndBivariateGMM, self).__init__()

    def forward(self, true, mu, sigma, corr, log_coeff, log_binary):
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

        eps = 1E-5

        binary = th.sigmoid(log_binary)
        binary = (binary + eps) * (1. - 2 * eps)
        coeff = softmax(log_coeff, eps=eps)

        i = th.log(binary)
        t = t0
        max_val = (-i).clamp(min=0)
        nc_b = i - i * t + max_val + ((-max_val).exp() + (-i - max_val).exp()).log()
        c_b = -nc_b

        buff = 1. - corr ** 2 + eps
        std_x = (t1 - mu1) / sigma1
        std_y = (t2 - mu2) / sigma2

        pi = 3.14159
        z = std_x ** 2 + (std_y ** 2 - 2. * corr * std_x * std_y)
        cost = - z / (2. * buff) - 0.5 * th.log(buff) - th.log(sigma1) - th.log(sigma2) - np.log(2. * pi)

        nll = -logsumexp(th.log(coeff) + cost, dim=2) - c_b
        return nll
        #from IPython import embed; embed(); raise ValueError()

        """
        normalizer = 1. / (2. * 3.14159 * sigma1 * sigma2 * th.sqrt(1. - corr ** 2))
        """

        """
        Z12 = 2 * corr * (t1 - mu1) / sigma1 * (t2 - mu2) / sigma2
        Z12 = Z12 * 1. / (2. * (1. - corr ** 2))
        """
        """
        # expansion of Z12?
        pp1 = 2. * corr * t1 * t2
        pp2 = -2. * corr * mu1 * t2
        pp3 = -2. * corr * t1 * mu2
        pp4 = 2. * corr * mu1 * mu2
        denom = 2. * (1. - corr ** 2) / (sigma1 * sigma2)
        Z12 = pp1 * denom + pp2 * denom + pp3 * denom + pp4 * denom
        """

        """
        inner11 = (0.5 * th.log(1. - corr ** 2 + self.epsilon))
        inner12 = th.log(2. * 3.14159 * sigma1) + th.log(2. * 3.14159 * sigma2) # + th.log(2. * 3.14159)

        inner1 = inner11 + inner12

        Z = (((t1 - mu1) / sigma1)**2) + (((t2 - mu2) / sigma2) **2)
        p1 = 2 * corr * t1 * t2
        p2 = -2 * corr * mu1 * t2
        p3 = -2 * corr * t1 * mu2
        p4 = 2 * corr * mu1 * mu2
        denom = 1. / (sigma1 * sigma2)
        Z -= denom * p1 + denom * p2 + denom * p3 + denom * p4
        #Z -= (2. * (corr * (t1 - mu1) * (t2 - mu2)) / (sigma1 * sigma2))
        inner2 = 0.5 * (1. / (1. - corr ** 2 + self.epsilon))
        log_gprob = -(inner1 + (inner2 * Z))
        """

        """
        # expansion of Z12?
        pp1 = 2. * corr * t1 * t2
        pp2 = -2. * corr * mu1 * t2
        pp3 = -2. * corr * t1 * mu2
        pp4 = 2. * corr * mu1 * mu2
        denom = 1. / ((2. * (1. - corr ** 2)) * (sigma1 * sigma2))

        Z12 = pp1 * denom + pp2 * denom + pp3 * denom + pp4 * denom

        Z1 = ((t1 - mu1) / sigma1) ** 2
        Z2 = ((t2 - mu2) / sigma2) ** 2

        Z = Z1 + Z2 - Z12
        log_gprob = th.log(normalizer) + Z
        """
        """
        if Z.sum() < 1:
            log_gprob = th.log(normalizer * th.exp(Z) + 1E-6)
        else:
            log_gprob = th.log(normalizer) + Z
        # this should stop NaN in log_grob
        p = th.log(normalizer) + Z
        log_gprob = (Z > 1).float() * p

        p = th.log(normalizer * th.exp(Z) + 1E-6)
        log_gprob += (Z <= 1).float() * th.log(normalizer * th.exp(Z) + 1E-6)
        """

        #log_gprob = (Z > 1).float() * (th.log(normalizer) + Z) + (Z <= 1).float() * th.log(normalizer * th.exp(Z) + 1E-6)
        #print("log_gprob: {}:{}".format(log_gprob.cpu().min().data[0], log_gprob.cpu().max().data[0]))
        #fcoeff = coeff + self.epsilon
        #fcoeff = fcoeff / fcoeff.sum(dim=2).expand_as(coeff)
        #fgprob = th.exp(log_gprob) + self.epsilon
        #fgprob = fgprob / fgprob.sum(dim=2).expand_as(fgprob)
        #log_gprob = th.log(fgprob)
        #print("post_log_gprob: {}:{}".format(log_gprob.cpu().min().data[0], log_gprob.cpu().max().data[0]))

        """
        # original
        # naturally, this is blowing UP!
        Z1 = (((t1 - mu1) / sigma1) ** 2) + (((t2 - mu2) / sigma2) ** 2)
        Z = Z1 - (2. * (corr * (t1 - mu1) * (t2 - mu2)) / (sigma1 * sigma2))
        inner2 = 0.5 * (1. / (1. - corr ** 2 + self.epsilon))
        cost = -(inner1 + (inner2 * Z))
        """

        """
        # Thanks to DWF https://gist.github.com/dwf/b2e1d8d575cb9e7365f302c90d909893
        a, t = log_binary, t0
        c_b = -1. * th.sum(t * F.softplus(-a) + (1. - t) * F.softplus(a), dim=2)
        """

        """
        # alternate version from BCE_with_logits
        # from F.binary_cross_entropy_with_logits
        i = th.log(binary)
        t = t0
        max_val = (-i).clamp(min=0)
        nc_b = i - i * t + max_val + ((-max_val).exp() + (-i - max_val).exp()).log()
        c_b = -nc_b
        """

        #l_fcoeff = logsumexp(coeff, dim=2)[:, :, 0].sum()
        #l_log_gprob = logsumexp(log_gprob, dim=2)[:, :, 0].sum()

        """
        print("Z1 {}, Z2 {}, Z12 {}".format(Z1.sum(), Z2.sum(), Z12.sum()))
        print("log normalizer {}, Z {}".format(th.log(normalizer).sum(), Z.sum()))
        print("normalizer {}, expZ {}".format(normalizer.sum(), th.exp(Z).sum()))
        print("l_fcoeff {}, l_log_gprob {}, l_c_b {}".format(l_fcoeff, l_log_gprob, c_b.sum()))
        """
        """
        ll1 = logsumexp(log_coeff, dim=2) + logsumexp(log_gprob, dim=2)[:, :, 0] #logsumexp(log_coeff + log_gprob, dim=2)[:, :, 0]
        ll2 = c_b
        nll = -ll1 - ll2
        return nll
        """

model = Model(minibatch_size, n_in, n_hid, n_out, n_chars,
              n_att_components, n_components)
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = BernoulliAndBivariateGMM()

if get_cuda():
    model = model.cuda()

"""
min1s = []
max1s = []
min2s = []
max2s = []
try:
    while True:
        mb, mb_mask, c_mb, c_mb_mask = next(train_itr)
        # normalize x, y deltas
        fixed = []
        for ii in range(mb.shape[1]):
            min1 = mb[:, ii, 1].min()
            max1 = mb[:, ii, 1].max()
            min2 = mb[:, ii, 2].min()
            max2 = mb[:, ii, 2].max()
            min1s.append(min1)
            max1s.append(max1)
            min2s.append(min2)
            max2s.append(max2)
except:
    min1s = np.array(min1s)
    max1s = np.array(max1s)
    min2s = np.array(min2s)
    max2s = np.array(max2s)
    print("min1 median {}".format(np.median(min1s)))
    print("max1 median {}".format(np.median(max1s)))
    print("min2 median {}".format(np.median(min2s)))
    print("max2 median {}".format(np.median(max2s)))
    from IPython import embed; embed(); raise ValueError()
"""

train_itt = 0
train_nan_itt = []
train_nan_chunk = []

valid_itt = 0
valid_nan_itt = []
valid_nan_chunk = []
def loop(itr, extra):
    mb, mb_mask, c_mb, c_mb_mask = next(itr)
    """
    # normalize x, y deltas
    min1 median -4.84999990463
    max1 median 24.25
    min2 median -7.0
    max2 median 16.25
    fixed = []
    for ii in range(mb.shape[1]):
        min1 = -4.8499999
        max1 = 24.25
        min2 = -7.0
        max2 = 16.25
        sub = mb[:, ii]
        sub[:, 1] = (sub[:, 1] - min1) / (max1 - min1) - 0.5
        sub[:, 2] = (sub[:, 2] - min2) / (max2 - min2) - 0.5
        fixed.append(sub)
    fixed = np.array(fixed)
    fixed = fixed.transpose(1, 0, 2)
    mb = fixed
    """

    global train_itt
    global train_nan_itt
    global train_nan_chunk
    global valid_itt
    global valid_nan_itt
    global valid_nan_chunk
    if extra["train"]:
        train_itt += 1

        valid_itt = 0
        valid_nan_itt = []
        valid_nan_chunk = []
    else:
        valid_itt += 1
        if len(train_nan_itt) > 0:
            raise ValueError("NaN detected in training")

        train_itt = 0
        train_nan_itt = []
        train_nan_chunk = []

    cinp_mb = c_mb.argmax(axis=-1).astype("int64")
    inp_mb = mb.astype(floatX)
    cuts = int(len(inp_mb) / float(cut_len)) + 1
    # if it is exact, need 1 less
    if (len(inp_mb) % cut_len) == 0:
        cuts = cuts - 1

    cuts_a = [(cut_i * cut_len, (cut_i + 1) * cut_len) for cut_i in range(cuts)]
    safe_cuts_a = []
    for cut_a in cuts_a:
        cut_s, cut_e = cut_a
        if len(inp_mb[cut_s:cut_e]) > 10:
            safe_cuts_a.append(cut_a)
        else:
            # extend by 10
            last_cut_s, last_cut_e = safe_cuts_a[-1]
            safe_cuts_a.pop()
            safe_cuts_a.append((last_cut_s, cut_e))

    # reset the model
    model.zero_grad()

    total_count = 0
    total_loss = 0
    inits = [Variable(i) for i in model.create_inits()]
    att_gru_init = inits[0]
    att_k_init = inits[1]
    dec_gru1_init = inits[2]
    dec_gru2_init = inits[3]
    for cut_a in safe_cuts_a:
        cut_s, cut_e = cut_a
        model.zero_grad()
        cinp_mb_v = Variable(th.LongTensor(cinp_mb))
        inp_mb_sub = inp_mb[cut_s:cut_e]
        mask_mb_sub = mb_mask[cut_s:cut_e]
        inp_mb_v = Variable(th.FloatTensor(inp_mb_sub))
        mask_mb_v = Variable(th.FloatTensor(mask_mb_sub))

        if get_cuda():
            inp_mb_v = inp_mb_v.cuda()
            cinp_mb_v = cinp_mb_v.cuda()
            mask_mb_v = mask_mb_v.cuda()
            att_gru_init = att_gru_init.cuda()
            att_k_init = att_k_init.cuda()
            dec_gru1_init = dec_gru1_init.cuda()
            dec_gru2_init = dec_gru2_init.cuda()

        o = model(cinp_mb_v, inp_mb_v, mask_mb_v,
                  att_gru_init, att_k_init, dec_gru1_init, dec_gru2_init)
        mu, sigma, corr, log_coeff, log_binary = o[:5]

        att_w, att_k = o[-2:]
        hiddens = o[5:-2]
        assert len(mu) > 2
        mu = mu[:-1]
        sigma = sigma[:-1]
        corr = corr[:-1]
        log_coeff = log_coeff[:-1]
        log_binary = log_binary[:-1]
        # target is 1:
        target = inp_mb_v[1:]
        l_full = loss_function(target, mu, sigma, corr, log_coeff, log_binary)
        """
        if train_itt == 161 and cut_a == (400, 500):
            print("The bad mama, at cut {}?".format(cut_a))
            from IPython import embed; embed()
        """
        if (l_full != l_full).float().sum().data[0] > 0:
            print("NaN detected, added to log")
            if extra["train"]:
                train_nan_itt.append(train_itt)
                train_nan_chunk.append(cut_a)
                print("train mb logged {}".format(train_nan_itt))
                print("train chunk logged {}".format(train_nan_chunk))
                from IPython import embed; embed();
                raise ValueError("NaN in train")
            else:
                valid_nan_itt.append(valid_itt)
                valid_nan_chunk.append(cut_a)
                print("valid mb logged {}".format(valid_nan_itt))
                print("valid chunk logged {}".format(valid_nan_chunk))
            if total_count == 0:
                total_count = 1
            return [total_loss / float(total_count)]

        l = ((l_full * mask_mb_v[1:]) / mask_mb_v[1:].sum().expand_as(l_full)).sum()
        if extra["train"]:
            l.backward(retain_variables=False)
            #th.nn.utils.clip_grad_norm(net.parameters(), max_norm)
            norm_gradient(model, 10.)
            optimizer.step()
            #clip_gradient(model, 100.)
        total_count += len(inp_mb_sub)
        total_loss = total_loss + l.cpu().data[0] * len(inp_mb_sub)

        att_k_init = Variable(att_k[-1].cpu().data)
        att_gru_init = Variable(hiddens[0][-1].cpu().data)
        dec_gru1_init = Variable(hiddens[1][-1].cpu().data)
        dec_gru2_init = Variable(hiddens[2][-1].cpu().data)
    return [total_loss / float(total_count)]


checkpoint_dict, model, optimizer = create_checkpoint_dict(model, optimizer,
        magic_reload=True, force_match="handwriter")
print(torch_summarize(model))

TL = TrainingLoop(loop, train_itr,
                  loop, valid_itr,
                  n_epochs=n_epochs,
                  checkpoint_every_n_seconds=60 * 60 * 4,
                  checkpoint_every_n_epochs=10,
                  checkpoint_delay=5,
                  checkpoint_dict=checkpoint_dict,
                  skip_minimums=True,
                  skip_most_recents=False)
results = TL.run()
