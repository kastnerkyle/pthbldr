from nose.tools import assert_raises

import numpy as np
from pthbldr.utils import minibatch_kmedians
from pthbldr.utils import beamsearch


def test_kmedians():
    random_state = np.random.RandomState(1999)
    Xa = random_state.randn(200, 2)
    Xb = .25 * random_state.randn(200, 2) + np.array((5, 3))
    X = np.vstack((Xa, Xb))
    ind = np.arange(len(X))
    random_state.shuffle(ind)
    X = X[ind]
    M1 = minibatch_kmedians(X, n_iter=1, random_state=random_state)
    M2 = minibatch_kmedians(X, M1, n_iter=1000, random_state=random_state)

# must be globally importable?
pseudo_lm = {}
sentences = ["the cow went fast", "that horse went far"]
order = 1
vocab = [ord(c) for se in sentences for c in se]
# init counts
for va in vocab:
    k = va
    pseudo_lm[k] = {}
    for vb in vocab:
        pseudo_lm[k][vb] = 0

# add counts
for se in sentences:
    s = 0
    for n in range(len(se[:-order])):
        k = ord(se[s:s + order])
        v = ord(se[s + order])
        s = s + 1
        pseudo_lm[k][v] += 1

    # normalize
    s = 0
    for n in range(len(se[:-order])):
        k = ord(se[s:s + order])
        tot = 0
        for ki in pseudo_lm[k].keys():
            tot += pseudo_lm[k][ki]
        for ki in pseudo_lm[k].keys():
            pseudo_lm[k][ki] /= float(tot)
        s = s + 1

def prob_func(prefix):
    history = prefix[-1]
    k = history
    dist_lookup = pseudo_lm[k]
    # list of prob, key pairs back
    dist = [(v, k) for k, v in dist_lookup.items()]
    return dist


def test_beamsearch():
    beam_width = 5
    n_letters = 50
    random_state = np.random.RandomState(2177)
    for stochastic in (True, False):
        start_token = ord("t")
        end_token = ord("e")
        b = beamsearch(prob_func, beam_width, start_token=start_token,
                       end_token=end_token,
                       clip_len=n_letters,
                       stochastic=stochastic,
                       random_state=random_state)
        # Map to chars
        all_r = []
        for r in b:
            char_r0 = "".join([chr(c) for c in r[0]])
            all_r.append((char_r0, r[1], r[2]))
        # ordered from best (0) to worst
        assert all_r[0][1] >= all_r[1][1]

        start_token = ord("t")
        end_token = [ord("a"), ord("s")]
        b = beamsearch(prob_func, beam_width, start_token=start_token,
                       end_token=end_token,
                       clip_len=n_letters,
                       stochastic=stochastic,
                       random_state=random_state)
        # Map to chars
        all_r = []
        for r in b:
            char_r0 = "".join([chr(c) for c in r[0]])
            all_r.append((char_r0, r[1], r[2]))
        # ordered from best (0) to worst
        assert all_r[0][1] >= all_r[1][1]

        start_token = ord("t")
        end_token = [ord("c"), ord("a"), None]
        b = beamsearch(prob_func, beam_width, start_token=start_token,
                       end_token=end_token,
                       clip_len=n_letters,
                       stochastic=stochastic,
                       random_state=random_state)
        # Map to chars
        all_r = []
        for r in b:
            char_r0 = "".join([chr(c) for c in r[0]])
            all_r.append((char_r0, r[1], r[2]))
        # ordered from best (0) to worst
        assert all_r[0][1] >= all_r[1][1]

        # test that it still works with non-global (won't use multiprocessing)
        def inner_prob_func(prefix):
            history = prefix[-1]
            k = history
            dist_lookup = pseudo_lm[k]
            # list of prob, key pairs back
            dist = [(v, k) for k, v in dist_lookup.items()]
            return dist

        start_token = ord("t")
        end_token = [ord("c"), ord("a"), None]
        b = beamsearch(inner_prob_func, beam_width, start_token=start_token,
                       end_token=end_token,
                       clip_len=n_letters,
                       stochastic=stochastic,
                       random_state=random_state)
        # Map to chars
        all_r = []
        for r in b:
            char_r0 = "".join([chr(c) for c in r[0]])
            all_r.append((char_r0, r[1], r[2]))
        # ordered from best (0) to worst
        assert all_r[0][1] >= all_r[1][1]

        def fail():
            # be sure that non-importable function and a timeout raises errors
            start_token = ord("t")
            end_token = [ord("c"), ord("a"), None]
            b = beamsearch(inner_prob_func, beam_width, start_token=start_token,
                           end_token=end_token,
                           clip_len=n_letters,
                           beam_timeout=5,
                           stochastic=stochastic,
                           random_state=random_state)
        assert_raises(ValueError, fail)
