import matplotlib
matplotlib.use("Agg")
import numpy as np
floatX = "float32"

import copy
from pthbldr import fetch_checkpoint_dict
from pthbldr import get_cuda, set_cuda
from extras import fetch_iamondb, list_iterator, rsync_fetch, plot_lines_iamondb_example

iamondb = rsync_fetch(fetch_iamondb, "leto01")

minibatch_size = 50
cut_len = 300
which_example = 0

X = iamondb["data"]
y = iamondb["target"]
vocabulary = iamondb["vocabulary"]
vocabulary_size = iamondb["vocabulary_size"]
pen_trace = np.array([x.astype(floatX) for x in X])
chars = np.array([yy.astype(floatX) for yy in y])
train_itr = list_iterator([pen_trace, chars], minibatch_size, axis=1, stop_index=10000,
                          make_mask=True)
valid_itr = list_iterator([pen_trace, chars], minibatch_size, axis=1, start_index=10000,
                          make_mask=True)

mb, mb_mask, c_mb, c_mb_mask = next(train_itr)
train_itr.reset()

orig_mb = mb[:, which_example].copy()
#samp_mb[:, 0] = orig_mb[:len(samp_mb), 0]

def get_text(cond):
    inv_map = {v: k for k, v in iamondb['vocabulary'].items()}
    return "".join([inv_map[c] for c in cond.flatten()])

text = get_text(c_mb[:, which_example].argmax(axis=-1))
plot_lines_iamondb_example(orig_mb, title=text, save_name="tru")

#from IPython import embed; embed(); raise ValueError()

checkpoint_dict, model, optimizer = fetch_checkpoint_dict(["handwriter"])
set_cuda(True)
loss_function = BernoulliAndBivariateGMM()

# based on
# https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
def sigmoid(x):
    gtz = x > 0
    gtz_z = np.exp(gtz * -x)
    gtz_z = gtz * (1. / (1 + gtz_z))

    ltz = x <= 0
    ltz_z = np.exp(ltz * x)
    ltz_z = ltz * (ltz_z / (1. + ltz_z))
    return (gtz_z + ltz_z)

def softmax(X, axis=-1):
    # https://nolanbconaway.github.io/blog/2017/softmax-numpy
    # make X at least 2d
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()
    return p


def numpy_sample_bernoulli_and_bivariate_gmm(mu, sigma, corr, coeff, binary,
                                             random_state, epsilon=1E-5,
                                             use_map=False):
    # only handles one example at a time
    # renormalize coeff to assure sums to 1...
    coeff = coeff / (coeff.sum(axis=-1, keepdims=True) + 1E-3)
    idx = np.array([np.argmax(random_state.multinomial(1, coeff[i, :])) for i in range(len(coeff))])

    mu_i = mu[np.arange(len(mu)), :, idx]
    sigma_i = sigma[np.arange(len(sigma)), :, idx]
    corr_i = corr[np.arange(len(corr)), idx]

    mu_x = mu_i[:, 0]
    mu_y = mu_i[:, 1]
    sigma_x = sigma_i[:, 0]
    sigma_y = sigma_i[:, 1]
    if use_map:
        s_b = binary > 0.5
        s_x = mu_x[:, None]
        s_y = mu_y[:, None]
        s = np.concatenate([s_b, s_x, s_y], axis=-1)
        return s
    else:
        z = random_state.randn(*mu_i.shape)
        un = random_state.rand(*binary.shape)
        s_b = un < binary

        s_x = (mu_x + sigma_x * z[:, 0])[:, None]
        s_y = mu_y + sigma_y * (
            (z[:, 0] * corr_i) + (z[:, 1] * np.sqrt(1. - corr_i ** 2)))
        s_y = s_y[:, None]
        s = np.concatenate([s_b, s_x, s_y], axis=-1)
        return s


def sample_sequence(itr):
    mb, mb_mask, c_mb, c_mb_mask = next(train_itr)

    cinp_mb = c_mb.argmax(axis=-1).astype("int64")
    inp_mb = mb.astype(floatX)
    cuts = int(len(inp_mb) / float(cut_len)) + 1
    # if it is exact, need 1 less
    if (len(inp_mb) % cut_len) == 0:
        cuts = cuts - 1

    # reset the model
    model.zero_grad()

    total_count = 0
    inits = [Variable(i) for i in model.create_inits()]

    att_gru_init = inits[0]
    att_k_init = inits[1]
    dec_gru1_init = inits[2]
    dec_gru2_init = inits[3]
    mu_results = []
    sigma_results = []
    corr_results = []
    log_coeff_results = []
    log_binary_results = []
    att_w_results = []
    att_k_results = []
    for cut_i in range(cuts):
        cinp_mb_v = Variable(th.LongTensor(cinp_mb))
        inp_mb_sub = inp_mb[cut_i * cut_len:(cut_i + 1) * cut_len]
        mask_mb_sub = mb_mask[cut_i * cut_len:(cut_i + 1) * cut_len]
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

        mu = mu[:-1]
        sigma = sigma[:-1]
        corr = corr[:-1]
        log_coeff = log_coeff[:-1]
        log_binary = log_binary[:-1]

        mu_results.append(mu.cpu().data.numpy())
        sigma_results.append(sigma.cpu().data.numpy())
        corr_results.append(corr.cpu().data.numpy())
        log_coeff_results.append(log_coeff.cpu().data.numpy())
        log_binary_results.append(log_binary.cpu().data.numpy())

        att_w_results.append(att_w.cpu().data.numpy())
        att_k_results.append(att_k.cpu().data.numpy())

        hiddens = o[5:-2]
        att_k_init = Variable(att_k[-1].cpu().data)
        att_gru_init = Variable(hiddens[0][-1].cpu().data)
        dec_gru1_init = Variable(hiddens[1][-1].cpu().data)
        dec_gru2_init = Variable(hiddens[2][-1].cpu().data)
    att_w_results = np.concatenate(att_w_results)
    att_k_results = np.concatenate(att_k_results)
    mu_results = np.concatenate(mu_results)
    sigma_results = np.concatenate(sigma_results)
    corr_results = np.concatenate(corr_results)
    log_coeff_results = np.concatenate(log_coeff_results)
    log_binary_results = np.concatenate(log_binary_results)

    def plot_attention(att_k_i, save_name):
        import matplotlib.pyplot as plt
        plt.plot(att_k_i.mean(axis=-1))
        plt.savefig(save_name)
        plt.close()


    plot_attention(att_k_results[:, which_example, :], save_name="att")
    plot_attention(att_k_results[:, which_example, :], save_name="att2")

    binary_results = sigmoid(log_binary_results)
    coeff_results = softmax(log_coeff_results)
    from IPython import embed; embed(); raise ValueError()


    mu_i = mu_results[:, which_example]
    sigma_i = sigma_results[:, which_example]
    corr_i = corr_results[:, which_example]
    coeff_i = coeff_results[:, which_example]
    binary_i = binary_results[:, which_example]

    random_state = np.random.RandomState(2177)

    s = numpy_sample_bernoulli_and_bivariate_gmm(mu_i, sigma_i, corr_i, coeff_i,
                                                 binary_i, random_state, use_map=True)
    samp_mb = s.copy()
    orig_mb = mb[:, which_example].copy()
    #samp_mb[:, 0] = orig_mb[:len(samp_mb), 0]

    def get_text(cond):
        inv_map = {v: k for k, v in iamondb['vocabulary'].items()}
        return "".join([inv_map[c] for c in cond.flatten()])

    text = get_text(cinp_mb[:, which_example])

    plot_lines_iamondb_example(samp_mb, title=text, save_name="samp")
    plot_lines_iamondb_example(orig_mb, title=text, save_name="gt")
    att_k_i = att_k_results[:, which_example].mean(axis=-1)

    from IPython import embed; embed(); raise ValueError()

sample_sequence(train_itr)
