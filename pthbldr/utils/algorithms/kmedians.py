# Author: Kyle Kastner
# Thanks to LD for mathematical guidance
# License: BSD 3-Clause
# See pseudocode for minibatch kmeans
# https://algorithmicthoughts.wordpress.com/2013/07/26/machine-learning-mini-batch-k-means/
# Unprincipled and hacky recentering to median at the end of function
import numpy as np
from scipy.cluster.vq import vq


def minibatch_kmedians(X, M=None, n_components=10, n_iter=100,
                       minibatch_size=100, random_state=None,
                       init_type="exhaustive_data",
                       init_values=None,
                       shuffle_minibatches=True,
                       verbose=False):
    """
    Example usage:
        random_state = np.random.RandomState(1999)
        Xa = random_state.randn(200, 2)
        Xb = .25 * random_state.randn(200, 2) + np.array((5, 3))
        X = np.vstack((Xa, Xb))
        ind = np.arange(len(X))
        random_state.shuffle(ind)
        X = X[ind]
        M1 = minibatch_kmedians(X, n_iter=1, random_state=random_state)
        M2 = minibatch_kmedians(X, M1, n_iter=1000, random_state=random_state)
    """
    n_clusters = n_components
    if M is not None:
        assert M.shape[0] == n_components
        assert M.shape[1] == X.shape[1]
    if random_state is None:
        random_state = np.random.RandomState(random_state)
    elif not hasattr(random_state, 'shuffle'):
        # Assume integer passed
        random_state = np.random.RandomState(int(random_state))
    if M is None:
        if init_type == "data":
            ind = np.arange(len(X)).astype('int32')
            random_state.shuffle(ind)
            M = X[ind[:n_clusters]]
            if init_values != None:
                M[:len(init_values)] = init_values
        elif init_type == "exhaustive_data":
            ind = np.arange(len(X)).astype('int32')
            random_state.shuffle(ind)
            M = X[ind[:n_clusters]]
            if init_values != None:
                M[:len(init_values)] = init_values
            clean = False
            pass_itr = 0
            bail_at = 10
            npi = n_clusters
            while not clean and pass_itr < bail_at:
                if verbose:
                    print("Picking centers, iter {}".format(pass_itr))
                pass_itr += 1
                clean = True
                for n, Mi in enumerate(M):
                    repeats = (Mi == M).all(axis=1)
                    # self-match means always 1 will match
                    if sum(repeats) > 1:
                        clean = False
                        repeat_idx = np.where(repeats > 0)[0]
                        # don't resample backwards
                        repeat_idx = repeat_idx[repeat_idx > n]
                        if len(repeat_idx) == 0:
                            continue
                        else:
                            npi_e = npi + len(repeat_idx)
                            if npi_e >= len(ind):
                                npi_e = len(ind)
                            M[repeat_idx[:npi_e - npi]] = X[ind[npi:npi_e]]
                            npi = npi_e
                            if npi_e == len(ind):
                                break

            if not clean and pass_itr >= bail_at:
                original_size = len(M)
                for n, Mi in enumerate(M):
                    repeats = (Mi == M).all(axis=1)
                    # self-match means always 1 will match
                    if sum(repeats) > 1:
                        repeat_idx = np.where(repeats > 0)[0]
                        # remove forward repeat clusters
                        repeat_idx = repeat_idx[repeat_idx > n]
                        if len(repeat_idx) == 0:
                            continue
                        else:
                            to_keep = np.array(sorted(list(set(np.arange(len(M))) - set(repeat_idx))))
                            M = M[to_keep]

                final_size = len(M)
                if verbose and n_iter == 0:
                    print("WARNING: Collapsed repeated cluster centers, from size {} to {}".format(original_size, final_size))
        else:
            raise ValueError("Unknown init_type {}".format(init_type))
    if n_iter == 0:
        return M
    elif n_iter < 0:
        raise ValueError("n_iter {} should be >= 0!".format(n_iter))

    center_counts = np.zeros(n_clusters)
    pts = list(np.arange(0, len(X), minibatch_size)) + [len(X)]
    if len(pts) == 1:
        # minibatch size > dataset size case
        pts = [0, None]
    minibatch_indices = zip(pts[:-1], pts[1:])
    if shuffle_minibatches:
        random_state.shuffle(minibatch_indices)
    for i in range(n_iter):
        if verbose:
            print("iter {}".format(i))
        if shuffle_minibatches:
            random_state.shuffle(minibatch_indices)
        for n, (mb_s, mb_e) in enumerate(minibatch_indices):
            Xi = X[mb_s:mb_e]
            # Broadcasted Manhattan distance
            # Could be made faster with einsum perhaps
            centers = np.abs(Xi[:, None, :] - M[None]).sum(
                axis=-1).argmin(axis=1)

            def count_update(c):
                center_counts[c] += 1
            [count_update(c) for c in centers]
            scaled_lr = 1. / center_counts[centers]
            Mi = M[centers]
            scaled_lr = scaled_lr[:, None]
            # Gradient of abs
            diff_error = (Xi - Mi)
            sq_error = diff_error ** 2
            g_error = (diff_error / np.sqrt(sq_error + 1E-9))
            Mi = Mi - scaled_lr * g_error
            M[centers] = Mi

    # Reassign centers to nearest datapoint
    mem, _ = vq(M, X)
    M = X[mem]

    # remove repeats
    original_size = len(M)
    for n, Mi in enumerate(M):
        repeats = (Mi == M).all(axis=1)
        # self-match means always 1 will match
        if sum(repeats) > 1:
            repeat_idx = np.where(repeats > 0)[0]
            # remove forward repeat clusters
            repeat_idx = repeat_idx[repeat_idx > n]
            if len(repeat_idx) == 0:
                continue
            else:
                to_keep = np.array(sorted(list(set(np.arange(len(M))) - set(repeat_idx))))
                M = M[to_keep]

    final_size = len(M)
    if verbose:
        print("WARNING: Collapsed repeated cluster centers, from size {} to {}".format(original_size, final_size))
    return M
