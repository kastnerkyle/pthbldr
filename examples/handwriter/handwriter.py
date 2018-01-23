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
n_epochs = 30  # Used way at the bottom in the training loop!
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
    def __init__(self, input_size, output_size, random_state=None):
        super(GLinear, self).__init__()
        if random_state is None:
            raise ValueError()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        # linear stores it "backwards"...
        self.linear.weight.data = th_normal((output_size, input_size),
                                             random_state=random_state)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x_orig = x
        last_axis = x.size(-1)
        x = x.view(-1, last_axis)
        l_o = self.linear(x)
        return l_o.view(*list(x_orig.size())[:-1] + [self.output_size])


def th_normal(shp, scale=0.08, random_state=None):
    if random_state is None:
        raise ValueError("Must pass np.random.RandomState(seed_integer) object!")
    if scale > 1 or scale <= 0:
        print("WARNING: excessive scale {} detected! Function should be called as th_normal((shp0, shp1), random_state=random_state), notice parenthesis!")

    return th.FloatTensor(scale * random_state.randn(*shp))


def th_zeros(shp):
    return th.FloatTensor(0. * np.zeros(shp))


class GGRUFork(nn.Module):
     def __init__(self, input_size, hidden_size, random_state):
         super(GGRUFork, self).__init__()
         self.W = nn.Parameter(th_normal((input_size, 3 * hidden_size),
                                         random_state=random_state))
         self.b = nn.Parameter(th_zeros((3 * hidden_size,)))
         self.input_size = input_size
         self.hidden_size = hidden_size

     def forward(self, inp):
         proj = th.mm(inp, self.W) + self.b[None].expand(inp.size(0), self.b.size(0))
         return proj


class GGRUCell(nn.Module):
    # https://discuss.pytorch.org/t/how-to-define-a-new-layer-with-autograd/351
    def __init__(self, hidden_size, minibatch_size, random_state):
        super(GGRUCell, self).__init__()
        self.Wur = nn.Parameter(th_normal((hidden_size, 2 * hidden_size),
                                           random_state=random_state))
        self.U = nn.Parameter(th_normal((hidden_size, hidden_size),
                                         random_state=random_state))
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
                 cell_type="GRU", default_step=1., random_state=None):
        super(GaussianAttentionCell, self).__init__()
        if random_state is None:
            raise ValueError("Must pass random_state!")
        self.c_input_size = c_input_size
        self.input_size = input_size
        self.n_components = n_components
        self.minibatch_size = minibatch_size
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.default_step = default_step
        self.att_a = GLinear(self.c_input_size, self.n_components, random_state=random_state)
        self.att_b = GLinear(self.c_input_size, self.n_components, random_state=random_state)
        self.att_k = GLinear(self.c_input_size, self.n_components, random_state=random_state)
        if cell_type == "GRU":
            self.inp_fork = GLinear(self.input_size, self.hidden_size, random_state=random_state)
            self.fork = GGRUFork(2 * self.input_size, self.hidden_size, random_state=random_state)
            self.cell = GGRUCell(self.hidden_size, self.minibatch_size, random_state=random_state)
        else:
            raise ValueError("Unsupported cell_type={}".format(cell_type))

    # 3D at every timestep
    def calc_phi(self, k_t, a_t, b_t, u_c):
        # add training 1 dim
        a_t = a_t[:, :, None]
        b_t = b_t[:, :, None]
        k_t = k_t[:, :, None]

        # go from minibatch_size, n_att_components, 1 to minibatch_size, n_att_components, c_inp_seq_len
        k_t = k_t.expand(k_t.size(0), k_t.size(1), u_c.size(2))

        # go from c_inp_seq_len to minibatch_size, n_att_components, c_inp_seq_len
        u_c = u_c.expand(k_t.size(0), k_t.size(1), u_c.size(2))

        """
        # quick check to be sure numpy matches th "broadcasting"
        aaaa = np.random.randn(50, 10, 1) * 0
        bbbb = np.linspace(0, 46-1, 46)
        cccc = aaaa - bbbb
        tttt = k_t - u_c
        """
        # square error term, shape minibatch_size, n_att_components, c_inp_seq_len
        ss1 = (k_t - u_c) ** 2
        b_t = b_t.expand(b_t.size(0), b_t.size(1), ss1.size(2))
        ss2 = -b_t * ss1
        a_t = a_t.expand(a_t.size(0), a_t.size(1), ss2.size(2))

        # still minibatch_size, n_att_components, c_inp_seq_len
        ss3 = a_t * th.exp(ss2)
        ss4 = ss3.sum(dim=1)
        return ss4

    def forward(self, c_inp, inp_t, gru_h_tm1, att_k_tm1, c_mask=None, inp_mask=None):
        cts = c_inp.size(0)
        minibatch_size = c_inp.size(1)

        k_tm1 = att_k_tm1
        h_tm1 = gru_h_tm1
        # input needs to be projected to hidden size and merge with cell...
        # otherwise this is junk

        u = Variable(th.FloatTensor(np.linspace(0, cts - 1, cts)))[None, None, :]
        if get_cuda():
            u = u.cuda()
            k_tm1 = att_k_tm1.cuda()
            h_tm1 = gru_h_tm1.cuda()

        # c_mask here?
        a_t = self.att_a(inp_t)
        b_t = self.att_b(inp_t)
        att_k_t = self.att_k(inp_t)

        #a_t, b_t, att_k_t all shape (minibatch_size, n_att_components)
        a_t_exp = th.exp(a_t)
        b_t_exp = th.exp(b_t)
        att_k_t_exp = th.exp(att_k_t)
        k_t = k_tm1.expand_as(att_k_t) + self.default_step * att_k_t_exp

        # phi has shape minibatch_size, 1, c_inp_seq_len
        phi = self.calc_phi(k_t, a_t_exp, b_t_exp, u)
        # minibatch_size, c_inp_seq_len, embed_dim
        c = c_inp.permute(1, 0, 2)
        """
        # sanity check shapes for proper equivalent to np.dot
        aaaa = np.random.randn(50, 1, 46)
        bbbb = np.random.randn(50, 46, 400)
        r = np.matmul(aaaa, bbbb)
        # r has shape ms, 1, embed_dim
        # since aaaa and bbbb are > 2d, treated as stack of matrices, matrix dims on last 2 axes
        # this means 50, 1, 46 x 50, 46, 400 is 50 reps of 1, 46 x 46, 400
        # leaving shape 50, 1, 400
        # equivalent to dot for 1 matrix is is (aaaa[0][:, :, None] * bbbb[0][None, :, :]).sum(axis=-2)
        # so for all 50, (aaaa[:, :, :, None] * bbbb[:, None, :, :]).sum(axis=-2)
        # ((aaaa[:, :, :, None] * bbbb[:, None, :, :]).sum(axis=-2) == r).all()
        _a = Variable(th.FloatTensor(aaaa))
        _b = Variable(th.FloatTensor(bbbb))
        e_a = _a[:, :, :, None].expand(_a.size(0), _a.size(1), _a.size(2), _b.size(2))
        e_b = _b[:, None, :, :].expand(_b.size(0), _a.size(1), _b.size(1), _b.size(2))
        # In [17]: np.sum(((e_a * e_b).sum(dim=-2)[:, :, 0].data.numpy() - r) ** 2)
        # Out[17]: 1.6481219193765024e-08
        """
        # equivalent to comb = th.matmul(phi, c), for backwards compat
        e_phi = phi[:, :, :, None].expand(phi.size(0), phi.size(1), phi.size(2), c.size(2))
        e_c = c[:, None, :, :].expand(c.size(0), phi.size(1), c.size(1), c.size(2))
        comb = (e_phi * e_c).sum(dim=-2)[:, :, 0]
        # comb has shape minibatch_size, 1, embed_size
        # w_t has shape minibatch_size, embed_size
        w_t = comb[:, 0, :]

        finp_t = self.inp_fork(inp_t)
        f_t = self.fork(th.cat([finp_t, w_t], 1))
        h_t = self.cell(f_t, h_tm1, inp_mask)
        # slice out the empty 1 dim, leaving shape minibatch_size, c_inp_seq_len
        phi_t = phi[:, 0]
        return h_t, k_t, phi_t, w_t

    def create_inits(self):
        if self.cell_type == "GRU":
            h_i = self.cell.create_inits()
        k_i = th.zeros(self.minibatch_size, self.n_components)
        if get_cuda():
            h_i = h_i.cuda()
            k_i = k_i.cuda()
        return [h_i, k_i]


def log_softmax(inp, eps=1E-6, axis=-1):
    # https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
    if axis != -1:
        raise ValueError("NYI")
    input_size = inp.size()

    trans_input = inp.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    # backwards compat ... https://github.com/pytorch/pytorch/issues/1546
    def _log_softmax(input_):
        max_vals, max_pos = th.max(input_, 1)
        input_ = input_ - max_vals.expand_as(input_)
        input_exp = th.exp(input_)
        norm_vals = input_exp.sum(1)
        norm_vals = th.log(norm_vals)
        # subtract sum
        input_ = input_ - norm_vals.expand_as(input_)
        return input_

    log_softmax_2d = _log_softmax(input_2d)
    log_softmax_nd = log_softmax_2d.view(*trans_size)
    tt = log_softmax_nd.transpose(axis, len(input_size) - 1)
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
    for k, p in model.named_parameters():
        if p.grad is None:
            continue
        if (p.grad != p.grad).float().sum().data[0] > 0:
            r = 0.
            print("WARNING: NaN grad, replacing by {}".format(r))
            p.grad[p.grad != p.grad] = r
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

    def forward(self, true, mu, sigma, corr, lin_coeff, lin_binary):
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

        eps = 1E-3

        log_coeff = log_softmax(lin_coeff, eps=eps)

        a, t = lin_binary, t0
        c_b = -1. * th.sum(t * F.softplus(-a) + (1. - t) * F.softplus(a), dim=2)

        pi = 3.14159
        buff = 1. - corr ** 2 + eps
        std_x = (t1 - mu1) / sigma1
        std_y = (t2 - mu2) / sigma2

        z = std_x ** 2 + std_y ** 2 - 2. * corr * std_x * std_y

        log_gprob = - z / (2. * buff) - 0.5 * th.log(buff) - th.log(sigma1) - th.log(sigma2) - np.log(2. * pi)
        nll = -logsumexp(log_coeff + log_gprob, dim=2) - c_b
        return nll


# MUST PASS ALL THESE IN
class Model(nn.Module):
    def __init__(self, minibatch_size, n_in, n_hid, n_out, n_chars,
                 n_att_components, n_components, bias=0., att_step=.01,
                 random_seed=2177):
        super(Model, self).__init__()
        self.minibatch_size = minibatch_size
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_chars = n_chars
        self.att_step = att_step
        self.n_att_components = n_att_components
        self.n_components = n_components
        self.random_state = np.random.RandomState(random_seed)
        # 1 for beroulli
        # 1 * n_outs * n_components for mean
        # 1 * n_outs * n_components for var
        # 1 * n_components for membership
        # 1 * n_components for corr - note this will be different for high dimensional outputs... :|
        # self.n_density = 1 + 6 * self.n_components
        self.n_density = 1 + ((1 + 1) * (n_out - 1) * n_components) + ((1 + 1) * n_components)
        self.minibatch_size = minibatch_size
        self.bias_coeff = bias
        self.bias_sigma = bias

        # random state or at least control seeding...
        self.linp = nn.Embedding(self.n_chars, self.n_hid)
        self.linp.weight.data = th_normal((self.n_chars, self.n_hid),
                                           scale=1.0,
                                           random_state=random_state)
        # n_in is the number of chars, n_out is 3 (pen, X, y)
        self.lproj = GLinear(self.n_out, self.n_hid, random_state=self.random_state)
        self.att_l1 = GaussianAttentionCell(self.n_hid, self.n_hid, self.n_hid,
                                            self.n_att_components,
                                            self.minibatch_size,
                                            default_step=att_step,
                                            random_state=self.random_state)
        self.proj_to_l2 = GGRUFork(self.n_hid, self.n_hid, random_state=self.random_state)
        self.proj_to_l3 = GGRUFork(self.n_hid, self.n_hid, random_state=self.random_state)
        self.att_to_l2 = GGRUFork(self.n_hid, self.n_hid, random_state=self.random_state)
        self.att_to_l3 = GGRUFork(self.n_hid, self.n_hid, random_state=self.random_state)
        self.l1_to_l2 = GGRUFork(self.n_hid, self.n_hid, random_state=self.random_state)
        self.l1_to_l3 = GGRUFork(self.n_hid, self.n_hid, random_state=self.random_state)
        self.l2_to_l3 = GGRUFork(self.n_hid, self.n_hid, random_state=self.random_state)
        self.l2 = GGRUCell(self.n_hid, self.minibatch_size, random_state=self.random_state)
        self.l3 = GGRUCell(self.n_hid, self.minibatch_size, random_state=self.random_state)
        self.loutp1 = GLinear(self.n_hid, self.n_density, random_state=self.random_state)
        self.loutp2 = GLinear(self.n_hid, self.n_density, random_state=self.random_state)
        self.loutp3 = GLinear(self.n_hid, self.n_density, random_state=self.random_state)
        self.poutp = GLinear(self.n_density, self.n_density, random_state=self.random_state)

        for m in self.modules():
            # Handle in respective classes
            pass
            """
            if isinstance(m, GLinear):
                sz = m.linear.weight.size()
                m.linear.weight.data = th_normal((sz[0], sz[1]))
                m.linear.bias.data.zero_()
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
        lin_coeff = outs[..., 5*k:6*k]
        lin_binary = outs[..., 6*k:]
        corr = th.tanh(corr)
        sigma = F.softplus(sigma - self.bias_sigma) + 1E-4
        mu = mu.contiguous().view(mu.size()[:-1] + (2, self.n_components))
        sigma = sigma.contiguous().view(sigma.size()[:-1] + (2, self.n_components))
        return mu, sigma, corr, lin_coeff, lin_binary

    def forward(self, c_inp, inp, mask_inp,
                att_gru_init, att_k_init, dec_gru1_init, dec_gru2_init):
        lproj_o = self.lproj(inp)
        l1_o = self.linp(c_inp)
        ts = inp.size(0)

        hiddens = [Variable(th.zeros(ts, self.minibatch_size, self.n_hid)) for i in range(3)]
        att_w = Variable(th.zeros(ts, self.minibatch_size, self.n_hid))
        att_k = Variable(th.zeros(ts, self.minibatch_size, self.n_att_components))
        att_phi = Variable(th.zeros(ts, self.minibatch_size, len(c_inp)))

        att_inits = self.att_l1.create_inits()
        hh = att_inits[0].cpu().numpy()
        kk = att_inits[1].cpu().numpy()

        aghtm1 = th.FloatTensor(hh + att_gru_init.cpu().data.numpy())
        aktm1 = th.FloatTensor(kk + att_k_init.cpu().data.numpy())
        att_gru_h_tm1 = Variable(aghtm1)
        att_k_tm1 = Variable(aktm1)

        if get_cuda():
            hiddens = [h.cuda() for h in hiddens]
            att_w = att_w.cuda()
            att_k = att_k.cuda()
            att_phi = att_phi.cuda()
            att_k_tm1 = att_k_tm1.cuda()
            att_gru_h_tm1 = att_gru_h_tm1.cuda()

        for i in range(ts):
            proj_t = lproj_o[i]
            mask_t = mask_inp[i]
            h2_tm1 = dec_gru1_init
            h3_tm1 = dec_gru2_init

            att_h_t, att_k_t, att_phi_t, att_w_t = self.att_l1(l1_o, proj_t, att_gru_h_tm1, att_k_tm1, inp_mask=mask_t)

            h1_t = att_h_t
            w_t = att_w_t
            phi_t = att_phi_t
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
            att_phi[i] = att_phi[i] + att_phi_t

        output = self.loutp1(hiddens[0]) + self.loutp2(hiddens[1]) + self.loutp3(hiddens[2])
        poutput = self.poutp(output)
        mu, sigma, corr, lin_coeff, lin_binary = self._slice_outs(poutput)
        return [mu, sigma, corr, lin_coeff, lin_binary] + hiddens + [att_w, att_k, att_phi]

    def create_inits(self):
        l2_h = th.zeros(self.minibatch_size, self.n_hid)
        l3_h = th.zeros(self.minibatch_size, self.n_hid)
        if get_cuda():
             l2_h = l2_h.cuda()
             l3_h = l3_h.cuda()
        return self.att_l1.create_inits() + [l2_h, l3_h]


model = Model(minibatch_size, n_in, n_hid, n_out, n_chars,
              n_att_components, n_components)
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = BernoulliAndBivariateGMM()

if get_cuda():
    model = model.cuda()

# loop *must* be after loading...
#checkpoint_dict, model, optimizer = create_checkpoint_dict(model, optimizer,
#    magic_reload=True, force_match="handwriter")
checkpoint_dict, model, optimizer = create_checkpoint_dict(model, optimizer)
print(torch_summarize(model))

def loop(itr, extra):
    mb, mb_mask, c_mb, c_mb_mask = next(itr)
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
    for s_n, cut_a in enumerate(safe_cuts_a):
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
        mu, sigma, corr, lin_coeff, lin_binary = o[:5]

        att_w, att_k, att_phi = o[-3:]

        hiddens = o[5:-3]
        assert len(mu) > 2
        mu = mu[:-1]
        sigma = sigma[:-1]
        corr = corr[:-1]
        lin_coeff = lin_coeff[:-1]
        lin_binary = lin_binary[:-1]
        # target is 1:
        target = inp_mb_v[1:]
        l_full = loss_function(target, mu, sigma, corr, lin_coeff, lin_binary)
        l = ((l_full * mask_mb_v[1:]) / mask_mb_v[1:].sum().expand_as(l_full)).sum()
        if extra["train"]:
            l.backward()
            norm_gradient(model, 10.)
            optimizer.step()
        total_count += len(inp_mb_sub)
        total_loss = total_loss + l.cpu().data[0] * len(inp_mb_sub)

        att_k_init = Variable(th.FloatTensor(att_k[-1].cpu().data.numpy().copy()))
        att_gru_init = Variable(th.FloatTensor(hiddens[0][-1].cpu().data.numpy().copy()))
        dec_gru1_init = Variable(th.FloatTensor(hiddens[1][-1].cpu().data.numpy().copy()))
        dec_gru2_init = Variable(th.FloatTensor(hiddens[2][-1].cpu().data.numpy().copy()))
    return [total_loss / float(total_count)]


TL = TrainingLoop(loop, train_itr,
                  loop, valid_itr,
                  n_epochs=n_epochs,
                  checkpoint_every_n_seconds=60 * 60 * 4,
                  checkpoint_every_n_epochs=5,
                  checkpoint_delay=3,
                  checkpoint_dict=checkpoint_dict,
                  skip_minimums=True,
                  skip_most_recents=False)
results = TL.run()
