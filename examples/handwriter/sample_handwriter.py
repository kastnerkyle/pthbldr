from pthbldr import fetch_checkpoint_dict
from pthbldr import get_cuda, set_cuda
from extras import fetch_iamondb, list_iterator

use_cuda = True
global use_cuda

checkpoint_dict, model, optimizer = fetch_checkpoint_dict(["handwriter"])
use_cuda = True
iamondb = fetch_iamondb()
minibatch_size = 50
cut_len = 300

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

def sample_sequence(itr):
    global use_cuda
    mb, mb_mask, c_mb, c_mb_mask = next(train_itr)
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
        total_count += len(inp_mb_sub)
        total_loss = total_loss + l.cpu().data[0] * len(inp_mb_sub)

        att_k_init = Variable(att_k[-1].cpu().data)
        att_gru_init = Variable(hiddens[0][-1].cpu().data)
        dec_gru1_init = Variable(hiddens[1][-1].cpu().data)
        dec_gru2_init = Variable(hiddens[2][-1].cpu().data)
    print(total_loss / float(total_count))
    from IPython import embed; embed(); raise ValueError()

sample_sequence(train_itr)
