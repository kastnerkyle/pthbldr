#!/usr/bin/env python
import numpy as np
from scipy.cluster.vq import vq
import os
import cPickle as pickle
import copy
import collections

from pthbldr.datasets import pitches_and_durations_to_pretty_midi
from pthbldr.datasets import list_of_array_iterator
from pthbldr.datasets import fetch_bach_chorales_music21
from pthbldr.utils import minibatch_kmedians, beamsearch

#mu = fetch_lakh_midi_music21(subset="pop")
#mu = fetch_haralick_midi_music21(subset="mozart_piano")
#mu = fetch_symbtr_music21()
#mu = fetch_wikifonia_music21()

#n_epochs = 500
#n_epochs = 2350
#n_epochs = 3000

mu = fetch_bach_chorales_music21()
order = mu["list_of_data_pitch"][0].shape[-1]

random_state = np.random.RandomState(1999)

lp = mu["list_of_data_pitch"]
ld = mu["list_of_data_duration"]
lql = mu["list_of_data_quarter_length"]

pitch_clusters = 32000
dur_clusters = 10000
minibatch_size = 75
n_iter = 0
voice_type = "woodwinds"
#key = None
subset = None
key = "minor"
#subset = 100
train_cache = True
from_scratch = False
joint_order = 7
default_quarter_length = 55
hook_check = False
# random-ish ext to check for plagiarism
ext = 3 * joint_order + joint_order - 1

remove_files = ["samples/samples/" + fi for fi in os.listdir("samples/samples") if fi.endswith(".mid")]
for rf in remove_files:
    print("Removing pre-existing file {}".format(rf))
    os.remove(rf)

# prune to only major/minor
if key != None:
    keep_lp = []
    keep_ld = []
    keep_lql = []
    lk = mu["list_of_data_key"]
    for n in range(len(lp)):
        if subset != None and len(keep_lp) > subset:
            break

        if key in lk[n]:
            keep_lp.append(lp[n])
            keep_ld.append(ld[n])
            keep_lql.append(lql[n])
    lp = copy.deepcopy(keep_lp)
    ld = copy.deepcopy(keep_ld)
    lql = copy.deepcopy(keep_lql)

def backfill(list_of_arr, empty=-1):
    final = []
    for arr in list_of_arr:
        for ic in range(arr.shape[1]):
            last = arr[arr[:, ic] != empty, ic][0]
            for ir in range(arr.shape[0]):
                if arr[ir, ic] == empty:
                    arr[ir, ic] = last
                else:
                    last = arr[ir, ic]
        # the first element must be defined for all 4!
        is_gt = arr > 0.
        all_true = np.prod(is_gt, axis=1)
        try:
            first_all = np.where(all_true > 0)[0][0]
            final.append(arr[first_all:])
        except:
            arr = arr * 0. - 10000
            final.append(arr[first_all:])
    return final

blp = backfill(lp)
prune = [i for i in range(len(blp)) if (blp[i] < -1000).any()]
lp = [lpi for i, lpi in enumerate(lp) if i not in prune]
ld = [ldi for i, ldi in enumerate(ld) if i not in prune]
lql = [lqli for i, lqli in enumerate(lql) if i not in prune]
# now that it is pruned try again
blp = backfill(lp)
prune = [i for i in range(len(blp)) if (blp[i] < -1000).any()]
assert len(prune) == 0

llp = [lpi[:, 0] for lpi in blp]
allowed_bottom_notes = np.unique(np.concatenate(llp, axis=0))
all_blp = np.concatenate(blp, axis=0)

vlp = [np.array((lpi[:, 1] - lpi[:, 0], lpi[:, 2] - lpi[:, 1], lpi[:, 3] - lpi[:, 2])).T for lpi in blp]

vlps = np.concatenate(vlp, axis=0)
dd1 = collections.Counter(vlps[:, 0].ravel())
dd2 = collections.Counter(vlps[:, 1].ravel())
dd3 = collections.Counter(vlps[:, 2].ravel())
allowed_intervals1 = sorted([did[0] for did in dd1.most_common()[:36]])
allowed_intervals2 = sorted([did[0] for did in dd2.most_common()[:36]])
allowed_intervals3 = sorted([did[0] for did in dd3.most_common()[:36]])
allowed_values = [allowed_bottom_notes, allowed_intervals1, allowed_intervals2, allowed_intervals3]

dlp = [blpi[1:] - blpi[:-1] for blpi in blp]
lp = [np.concatenate((lpi[0][None], dlpi), axis=0) for lpi, dlpi in zip(lp, dlp)]

mu["pitch_list"] = [u for u in np.unique(np.concatenate(lp, axis=0))]

n_dur = len(mu["duration_list"])
n_pitches = len(mu["pitch_list"])
pitch_oh_size = len(mu["pitch_list"])
dur_oh_size = len(mu["duration_list"])


# 100+ potential seeds, reused from train...
si = max((0., .9 - 100. / len(lp)))

train_itr = list_of_array_iterator([lp, ld], minibatch_size,
                                   list_of_extra_info=[lql],
                                   #stop_index=si,
                                   randomize=True, random_state=random_state)

valid_itr = list_of_array_iterator([lp, ld], minibatch_size,
                                   list_of_extra_info=[lql],
                                   start_index=si,
                                   randomize=True, random_state=random_state)

# make sure there is at least 1 minibatch worth!
try:
    next(valid_itr)
    valid_itr.reset()
except StopIteration:
    raise ValueError("Need to increase number of seeds!")

r = next(train_itr)
pitch_mb, pitch_mask, dur_mb, dur_mask = r[:4]
train_itr.reset()


def oh_3d(a, oh_size):
    return (np.arange(oh_size + 1) == a[:, :, None]).astype(int)


def get_codebook(list_of_arr, n_components, n_iter, oh_size):
    # make eos symbol all 0s
    eos = 0 * list_of_arr[0][-1][None]
    list_of_arr = [np.concatenate((la, eos), axis=0) for la in list_of_arr]
    j = np.vstack(list_of_arr)
    oh_j = oh_3d(j, oh_size=oh_size)
    shp = oh_j.shape
    oh_j2 = oh_j.reshape(-1, shp[1] * shp[2])
    oh_eos = oh_3d(eos, oh_size=oh_size)
    oh_eos2 = oh_eos.reshape(-1, shp[1] * shp[2])

    codebook = minibatch_kmedians(oh_j2, n_components=n_components,
                                  n_iter=n_iter,
                                  init_values=oh_eos2,
                                  random_state=random_state, verbose=True)
    return codebook


def quantize(list_of_arr, codebook, oh_size):
    # make eos symbol all 0s
    eos = 0 * list_of_arr[0][-1][None]
    list_of_arr = [np.concatenate((la, eos), axis=0) for la in list_of_arr]
    quantized_arr = []
    list_of_codes = []
    for arr in list_of_arr:
        oh_a = oh_3d(arr, oh_size)
        shp = oh_a.shape
        oh_a2 = oh_a.reshape(-1, shp[1] * shp[2])

        codes, _ = vq(oh_a2, codebook)
        list_of_codes.append(codes)

        q_oh_a = codebook[codes]
        q_oh_a = q_oh_a.reshape(-1, shp[1], shp[2]).argmax(axis=-1)
        quantized_arr.append(q_oh_a)
    return quantized_arr, list_of_codes


def codebook_lookup(list_of_code_arr, codebook, last_shape=4):
    reconstructed_arr = []
    for arr in list_of_code_arr:
        pitch_slices = []
        oh_codes = codebook[arr]
        pitch_size = codebook.shape[1] // last_shape
        boundaries = np.arange(1, last_shape + 1, 1) * pitch_size
        for i in range(len(oh_codes)):
            pitch_slice = np.where(oh_codes[i] == 1)[0]
            for n, bo in enumerate(boundaries):
                if len(pitch_slice) <= n:
                    pitch_slice = np.insert(pitch_slice, len(pitch_slice), 0)
                elif pitch_slice[n] >= bo:
                    pitch_slice = np.insert(pitch_slice, n, 0)
                else:
                    pass
            pitch_slices.append(pitch_slice % pitch_size)
        new_arr = np.array(pitch_slices).astype("float32")
        reconstructed_arr.append(new_arr)
    return reconstructed_arr


def fixup_dur_list(dur_list):
    new = []
    dl = mu["duration_list"]

    for ldi in dur_list:
        ldi = ldi.copy()
        dur_where = []
        for n, dli in enumerate(dl):
            dur_where.append(np.where(ldi == dli))

        for n, dw in enumerate(dur_where):
            ldi[dw] = n
        new.append(ldi)
    return new


def unfixup_dur_list(dur_list, hack=True):
    new = []
    dl = mu["duration_list"]
    for ldi in dur_list:
        ldi = ldi.copy().astype("float32")
        dur_where = []
        for n, dli in enumerate(dl):
            dur_where.append(np.where(ldi == n))

        if hack:
            for n, dw in enumerate(dur_where[:-2]):
                ldi[dw] = dl[n] if n <= 1 else dl[n + 2]
            new.append(ldi)
        else:
            for n, dw in enumerate(dur_where):
                ldi[dw] = dl[n]
            new.append(ldi)
    return new


def fixup_pitch_list(pitch_list):
    new = []
    pl = mu["pitch_list"]

    for lpi in pitch_list:
        lpi = lpi.copy()
        pitch_where = []
        for n, pli in enumerate(pl):
            pitch_where.append(np.where(lpi == pli))

        for n, pw in enumerate(pitch_where):
            lpi[pw] = n
        new.append(lpi)
    return new


def unfixup_pitch_list(pitch_list):
    new = []
    pl = mu["pitch_list"]
    for lpi in pitch_list:
        lpi = lpi.copy().astype("float32")
        pitch_where = []
        for n, pli in enumerate(pl):
            pitch_where.append(np.where(lpi == n))

        for n, pw in enumerate(pitch_where):
            lpi[pw] = pl[n]
        new.append(lpi)
    return new


ld = fixup_dur_list(ld)
lp = fixup_pitch_list(lp)

if from_scratch or not os.path.exists("dur_codebook.npy"):
    #if subset != None or key != None:
    #    raise ValueError("subset and key should be None when building the codebooks!")
    dur_codebook = get_codebook(ld, n_components=dur_clusters, n_iter=n_iter, oh_size=dur_oh_size)
    np.save("dur_codebook.npy", dur_codebook)
else:
    dur_codebook = np.load("dur_codebook.npy")


if from_scratch or not os.path.exists("pitch_codebook.npy"):
    #if subset != None or key != None:
    #    raise ValueError("subset and key should be None when building the codebooks!")
    pitch_codebook = get_codebook(lp, n_components=pitch_clusters, n_iter=n_iter, oh_size=pitch_oh_size)
    np.save("pitch_codebook.npy", pitch_codebook)
else:
    pitch_codebook = np.load("pitch_codebook.npy")


def pre_d(dmb, quantize_it=True):
    list_of_dur = [dmb[:, i, :] for i in range(dmb.shape[1])]
    o_list_of_dur = list_of_dur
    list_of_dur = fixup_dur_list(list_of_dur)

    if quantize:
        q_list_of_dur, q_list_of_dur_codes = quantize(list_of_dur, dur_codebook, dur_oh_size)
        o_q_list_of_dur = q_list_of_dur
        q_list_of_dur = unfixup_dur_list(q_list_of_dur, hack=False)
    else:
        q_list_of_dur = list_of_dur
        # garbage
        q_list_of_dur_codes = list_of_dur
        q_list_of_dur = unfixup_dur_list(q_list_of_dur, hack=False)

    q_list_of_dur = [qld[:, None, :] for qld in q_list_of_dur]
    q_list_of_dur_codes = [qldc[:, None, None] for qldc in q_list_of_dur_codes]

    q_dur_mb = np.concatenate(q_list_of_dur, axis=1)
    q_code_mb = np.concatenate(q_list_of_dur_codes, axis=1).astype("float32")
    return q_dur_mb, q_code_mb


def pre_p(pmb, quantize_it=True):
    list_of_pitch = [pmb[:, i, :] for i in range(pmb.shape[1])]
    o_list_of_pitch = list_of_pitch
    list_of_pitch = fixup_pitch_list(list_of_pitch)

    if quantize:
        q_list_of_pitch, q_list_of_pitch_codes = quantize(list_of_pitch, pitch_codebook, pitch_oh_size)
        o_q_list_of_pitch = q_list_of_pitch
    else:
        q_list_of_pitch = list_of_pitch
        # garbage
        q_list_of_pitch_codes = list_of_pitch
    q_list_of_pitch = unfixup_pitch_list(q_list_of_pitch)

    q_list_of_pitch = [qlp[:, None, :] for qlp in q_list_of_pitch]
    q_list_of_pitch_codes = [qlpc[:, None, None] for qlpc in q_list_of_pitch_codes]
    q_pitch_mb = np.concatenate(q_list_of_pitch, axis=1)
    q_code_mb = np.concatenate(q_list_of_pitch_codes, axis=1).astype("float32")
    return q_pitch_mb, q_code_mb


def accumulate(mb, counter_dict, order):
    counter_dict = copy.deepcopy(counter_dict)
    for mi in range(mb.shape[1]):
        si = order
        for ni in range(len(mb) - order - 1):
            se = si - order
            ee = si
            prefix = tuple(mb[se:ee, mi].ravel())
            next_i = mb[ee, mi].ravel()[0]
            if prefix not in counter_dict.keys():
                counter_dict[prefix] = {}

            if next_i not in counter_dict[prefix].keys():
                counter_dict[prefix][next_i] = 1
            else:
                counter_dict[prefix][next_i] += 1
            si += 1
    return counter_dict


def normalize(counter_dict):
    counter_dict = copy.deepcopy(counter_dict)
    for k in counter_dict.keys():
        sub_d = copy.deepcopy(counter_dict[k])

        tot = 0.
        for sk in sub_d.keys():
            tot += sub_d[sk]

        for sk in sub_d.keys():
            sub_d[sk] /= float(tot)

        counter_dict[k] = sub_d
    return counter_dict


if train_cache:
    print("Using train cache...")
    p_total_frequency = np.load("pitch_cache.npy").item()
    d_total_frequency = np.load("dur_cache.npy").item()
    print("Finished loading train cache...")
else:
    p_total_frequency = {}
    d_total_frequency = {}

    for r in train_itr:
        pitch_mb, pitch_mask, dur_mb, dur_mask = r[:4]
        q_pitch_mb, q_pitch_code_mb = pre_p(pitch_mb)
        q_dur_mb, q_dur_code_mb = pre_d(dur_mb)
        p_frequency = copy.deepcopy(p_total_frequency)
        d_frequency = copy.deepcopy(d_total_frequency)
        p_frequency = accumulate(q_pitch_code_mb, p_frequency, joint_order)
        d_frequency = accumulate(q_dur_code_mb, d_frequency, joint_order)
        p_total_frequency.update(p_frequency)
        d_total_frequency.update(d_frequency)

    p_total_frequency = normalize(p_total_frequency)
    d_total_frequency = normalize(d_total_frequency)
    np.save("pitch_cache.npy", p_total_frequency)
    np.save("dur_cache.npy", d_total_frequency)


def prune_invalid_keys(dist_lookup, prefix):
    dlk = sorted(list(dist_lookup.keys()))
    the_notes = codebook_lookup([[int(k) for k in dlk]], pitch_codebook, 4)[0]
    the_prefix = codebook_lookup([prefix], pitch_codebook, 4)[0]
    paths = [np.concatenate((the_prefix, notes[None]), axis=0) for notes in the_notes]
    allowed_keys = []
    for n, p in enumerate(paths):
        up = np.array(unfixup_pitch_list(p))
        up = np.cumsum(up, axis=0)
        up_l = up[-1]
        safe = True
        if up_l[0] not in allowed_values[0]:
            safe = False

        up_vlp = up_l[1:] - up_l[:-1]
        for i in range(len(up_vlp)):
            if up_vlp[i] not in allowed_values[i + 1]:
                safe = False
        if safe:
            allowed_keys.append(n)
    if len(allowed_keys) > 0:
        return {dlk[k]: dist_lookup[dlk[k]] for k in allowed_keys}
    else:
        return {}


def rolloff_lookup(lookup_dict, lookup_key, full_prefix, hook=False):
    """ roll off lookups n, n-1, n-2, n-3, down to random choice at 0 """
    lk = lookup_key
    ld = lookup_dict
    keylen = 0
    try:
        dist_lookup = lookup_dict[lk]
        keylen = len(lk)
        if hook:
            dist_lookup = prune_invalid_keys(dist_lookup, full_prefix)
            if len(dist_lookup.keys()) == 0:
                # need to map the whole thing back from symbol -> notes
                # then see if each and every interval is valid
                # check it here
                raise KeyError("nope.jpg")
    except KeyError:
        found = False
        for oi in range(1, len(lookup_key) - 1):
            sub_keys = [ki for ki in lookup_dict.keys() if lk[oi:] == ki[oi:]]
            if len(sub_keys) > 0:
               ii = 0
               random_state.shuffle(sub_keys)
               dist_lookup = lookup_dict[sub_keys[0]]
               if hook:
                   dist_lookup = prune_invalid_keys(dist_lookup, full_prefix)
                   if len(dist_lookup.keys()) == 0:
                       # check it here
                       continue
               found = True
               keylen = len(sub_keys[0])
               """
               for ii in range(len(sub_keys)):
                   dist_lookup = lookup_dict[sub_keys[ii]]
                   if hook:
                       dist_lookup = prune_invalid_keys(dist_lookup, full_prefix)
                       if len(dist_lookup.keys()) == 0:
                           # check it here
                           continue
                   found = True
                   keylen = len(sub_keys[ii])
                """
            if found:
                break
        if not found:
            # failover case
            # sub_keys = sorted(list(set(lookup_key)))
            # dist_lookup = {sk: 1. / len(sub_keys) for sk in sub_keys}
            sub_keys = sorted(list(full_prefix))
            dist_lookup = collections.Counter(sub_keys)
            """
            if hook:
                prune_invalid_keys(dist_lookup, full_prefix)
                # check it here too? Or just let it slide...
                pass
            """
    #if hook:
    #    from IPython import embed; embed(); raise ValueError()
    return dist_lookup


temperature = .25
def prob_func(prefix):
    history = prefix[-joint_order:]
    pitch_prefix = [p[0] for p in prefix]
    pitch_history = [h[0] for h in history]
    dur_prefix = [p[1] for p in prefix]
    dur_history = [h[1] for h in history]
    p_lu = tuple(pitch_history)
    d_lu = tuple(dur_history)
    pitch_dist = rolloff_lookup(p_total_frequency, p_lu, pitch_prefix, hook_check)
    dur_dist = rolloff_lookup(d_total_frequency, d_lu, dur_prefix, False)
    dist = []
    # model as p(x, y) = p(x) * p(y)
    for pk in pitch_dist.keys():
        for dk in dur_dist.keys():
            dist.append((pitch_dist[pk] * dur_dist[dk], (pk, dk)))

    probs_f = [d[0] for d in dist]
    probs_f = [pf - max(probs_f) for pf in probs_f]
    probs_f = [np.exp(pf / temperature) for pf in probs_f]
    epf = sum(probs_f)
    probs_f = [pf / epf for pf in probs_f]
    assert len(probs_f) == len(dist)
    dist = [(pf, d[1]) for pf, d in zip(probs_f, dist)]
    return dist


i = 0
for a in valid_itr:
    pitch_mb, pitch_mask, dur_mb, dur_mask = a[:4]
    pitch_mb, _ = pre_p(pitch_mb, quantize_it=False)
    dur_mb, _ = pre_d(dur_mb, quantize_it=False)

    pitch_mb = np.cumsum(pitch_mb, axis=0)
    pitch_mb = pitch_mb[ext:]
    dur_mb = dur_mb[ext:]
    pitches_and_durations_to_pretty_midi(pitch_mb, dur_mb,
                                         save_dir="samples/samples",
                                         name_tag="test_sample_{}.mid",
                                         #list_of_quarter_length=[int(.5 * qpm) for qpm in qpms],
                                         default_quarter_length=default_quarter_length,
                                         voice_params=voice_type,
                                         add_to_name=i * pitch_mb.shape[1])
    i += 1
valid_itr.reset()

i = 0
for a in valid_itr:
    pitch_mb, pitch_mask, dur_mb, dur_mask = a[:4]
    pitch_mb, _ = pre_p(pitch_mb, quantize_it=True)
    dur_mb, _ = pre_d(dur_mb, quantize_it=True)

    pitch_mb = np.cumsum(pitch_mb, axis=0)
    pitch_mb = pitch_mb[ext:]
    dur_mb = dur_mb[ext:]
    pitches_and_durations_to_pretty_midi(pitch_mb, dur_mb,
                                         save_dir="samples/samples",
                                         name_tag="test_quantized_sample_{}.mid",
                                         #list_of_quarter_length=[int(.5 * qpm) for qpm in qpms],
                                         default_quarter_length=default_quarter_length,
                                         voice_params=voice_type,
                                         add_to_name=i * pitch_mb.shape[1])
    i += 1
valid_itr.reset()

i = 0
for a in valid_itr:
    print("Beginning minibatch {}".format(i))
    n_pitch_mb, n_pitch_mask, n_dur_mb, n_dur_mask = a[:4]

    q_pitch_mb, q_pitch_code_mb = pre_p(n_pitch_mb)
    o_q_pitch_mb = copy.deepcopy(q_pitch_mb)

    q_dur_mb, q_dur_code_mb = pre_d(n_dur_mb)
    o_q_dur_mb = copy.deepcopy(q_dur_mb)

    qpms = r[-1]

    failed = []
    final_beams = []
    for mbi in range(minibatch_size):
        start_pitch_token = [int(qp) for qp in list(q_pitch_code_mb[:joint_order, mbi, 0])]
        start_dur_token = [int(dp) for dp in list(q_dur_code_mb[:joint_order, mbi, 0])]
        start_token = [tuple([qp, dp]) for qp, dp in zip(start_pitch_token, start_dur_token)]
        # eos for pitch, don't care what dur
        end_token = [(0, None)]
        stochastic = True
        diversity_score = "set"
        beam_width = 15
        clip = 60
        timeout = 20
        debug = False
        #timeout = None
        #debug = True
        verbose = True
        random_state = np.random.RandomState(90210)
        b = beamsearch(prob_func, beam_width,
                       start_token=start_token,
                       end_token=end_token,
                       clip_len=clip,
                       diversity_score=diversity_score,
                       stochastic=stochastic,
                       random_state=random_state,
                       verbose=verbose,
                       beam_timeout=timeout,
                       debug=debug)

        if len(b) > 0:
            final_beams.append(b)
        else:
            final_beams.append(b)
            failed.append(mbi)
            print("Sequence {} failed beamsearch timeout".format(mbi))
            """
            failed.append(i * q_pitch_mb.shape[1] + mbi)
            print("Sequence {} failed beamsearch timeout".format(mbi))
            final_pitches.append(q_pitch_mb[:joint_order + 1, mbi])
            final_durs.append(q_dur_mb[:joint_order + 1, mbi])
            """

    for ni in range(len(final_beams)):
        final_pitches = []
        final_durs = []

        if ni in failed:
            """
            # delete all failures here
            all_files = [fi for fi in os.listdir("samples/samples") if fi.endswith(".mid")]
            remove_files = []
            for failed_i in failed:
                to_remove = ["samples/samples/" + fi for fi in all_files if "sample_{}.mid".format(failed_i) in fi]
                remove_files.extend(to_remove)

            remove_files = sorted(list(set(remove_files)))
            for rf in remove_files:
                print("Removing copycat {}".format(rf))
                os.remove(rf)
            """
            # delete here?
            continue

        b = final_beams[ni]
        # for all beams, take the sequence (p[0]) and the respective type (ip[0] for pitch, ip[1] for dur)
        # last number (4) for reconstruction to actual data (used 4 voices)

        # top 5 beams, :5
        quantized_pitch_seqs = codebook_lookup([np.array([ip[0] for ip in p[0]]).astype("int32") for p in b[:5]], pitch_codebook, 4)
        quantized_dur_seqs = codebook_lookup([np.array([ip[1] for ip in p[0]]).astype("int32") for p in b[:5]], dur_codebook, 4)

        """
        # rerank top 5 by length
        quantized_pitch_seqs = list(sorted(quantized_pitch_seqs, key=lambda x: len(x)))[::-1]
        quantized_dur_seqs = list(sorted(quantized_dur_seqs, key=lambda x: len(x)))[::-1]
        """

        final_pitches.extend(quantized_pitch_seqs)
        final_durs.extend(quantized_dur_seqs)

        # make into a minibatch
        pad_size = max([len(fp) for fp in final_pitches])
        new_qps = np.zeros((pad_size, len(final_pitches), final_pitches[0].shape[1])).astype("float32")
        for n, fp in enumerate(final_pitches):
            new_qps[:len(fp), n] = fp

        pad_size = max([len(fd) for fd in final_durs])
        new_qds = np.zeros((pad_size, len(final_durs), final_durs[0].shape[1])).astype("float32")
        for n, fd in enumerate(final_durs):
            new_qds[:len(fd), n] = fd

        q_pitch_mb = new_qps
        q_dur_mb = new_qds

        min_len = min([q_pitch_mb.shape[0], q_dur_mb.shape[0]])
        q_pitch_mb = q_pitch_mb[:min_len]
        q_dur_mb = q_dur_mb[:min_len]

        # is this just unfixup????
        pitch_where = []
        duration_where = []
        pl = mu['pitch_list']
        dl = mu['duration_list']

        for n, pli in enumerate(pl):
            pitch_where.append(np.where(q_pitch_mb == n))

        for n, dli in enumerate(dl):
            duration_where.append(np.where(q_dur_mb == n))

        for n, pw in enumerate(pitch_where):
            q_pitch_mb[pw] = pl[n]

        for n, dw in enumerate(duration_where):
            q_dur_mb[dw] = dl[n]

        q_pitch_mb = np.cumsum(q_pitch_mb, axis=0)
        # ext in order to avoid influence of "priming"
        q_pitch_mb = q_pitch_mb[ext:]
        q_dur_mb = q_dur_mb[ext:]

        if q_pitch_mb.shape[0] == 0:
            continue
        if q_dur_mb.shape[0] == 0:
            continue

        f_q_pitch_mb = o_q_pitch_mb[ext:]
        f_q_dur_mb = o_q_dur_mb[ext:]
        if f_q_pitch_mb.shape[0] == 0:
            continue
        if f_q_dur_mb.shape[0] == 0:
            continue

        final_q_pitch_mb = 0. * q_pitch_mb
        final_q_dur_mb = 0. * q_dur_mb
        leadin = 2 * joint_order

        copycats = []
        for mi in range(q_pitch_mb.shape[1]):
            # if any beam fails zero it out
            sz = np.prod(q_pitch_mb[:leadin, mi].shape)
            # ni here because multiple beams are comparing to one element
            lp = f_q_pitch_mb[:leadin, ni]
            rp = q_pitch_mb[:leadin, mi]
            if lp.shape[0] > 0 and rp.shape[0] > 0 and rp.shape[0] == lp.shape[0]:
                matches = (lp == rp).sum()
            else:
                matches = 0

            if matches < (sz // 2):
                final_q_pitch_mb[:, mi] = q_pitch_mb[:, mi]
                final_q_dur_mb[:, mi] = q_dur_mb[:, mi]
            else:
                final_q_pitch_mb[:, mi] = 0. * q_pitch_mb[:, mi]
                final_q_dur_mb[:, mi] = 0. * q_dur_mb[:, mi]
                copycats.append(mi)
                print("Sequence {}:{} failed copycat check".format(ni, mi))

        q_pitch_mb = final_q_pitch_mb
        q_dur_mb = final_q_dur_mb


        if ni < 10:
            name_tag = "test_markov_sample_0{}".format(ni) + "_{}.mid"
        else:
            name_tag = "test_markov_sample_{}".format(ni) + "_{}.mid"
        path = "samples/samples/"
        pitches_and_durations_to_pretty_midi(q_pitch_mb, q_dur_mb,
                                             save_dir=path,
                                             name_tag=name_tag,
        #                                    list_of_quarter_length=qpms,
                                             voice_params=voice_type,
                                             default_quarter_length=default_quarter_length)

        for cc in copycats:
            cc_file = path + name_tag.format(cc)
            os.remove(cc_file)
            print("Removed copycat file {}".format(cc_file))
