import numbers
import numpy as np
import itertools
import re

from ..core import get_logger
from collections import Counter

logger = get_logger()


class base_iterator(object):
    def __init__(self, list_of_containers, minibatch_size,
                 axis,
                 start_index=0,
                 stop_index=np.inf,
                 make_mask=False,
                 one_hot_class_size=None):
        self.list_of_containers = list_of_containers
        self.minibatch_size = minibatch_size
        self.make_mask = make_mask
        self.start_index = start_index
        self.stop_index = stop_index
        self.slice_start_ = start_index
        self.axis = axis
        if axis not in [0, 1]:
            raise ValueError("Unknown sample_axis setting %i" % axis)

    def reset(self):
        self.slice_start_ = self.start_index

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self.slice_end_ = self.slice_start_ + self.minibatch_size
        if self.slice_end_ > self.stop_index:
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        ind = np.arange(self.slice_start_, self.slice_end_)
        self.slice_start_ = self.slice_end_
        if self.make_mask is False:
            return self._slice_without_masks(ind)
        else:
            return self._slice_with_masks(ind)

    def _slice_without_masks(self, ind):
        try:
            if self.axis == 0:
                return [c[ind] for c in self.list_of_containers]
            elif self.axis == 1:
                return [c[:, ind] for c in self.list_of_containers]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")

    def _slice_with_masks(self, ind):
        try:
            cs = self._slice_without_masks(ind)
            if self.axis == 0:
                ms = [np.ones_like(c[:, 0]) for c in cs]
            elif self.axis == 1:
                ms = [np.ones_like(c[:, :, 0]) for c in cs]
            assert len(cs) == len(ms)
            return [i for sublist in list(zip(cs, ms)) for i in sublist]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")


class minibatch_iterator(base_iterator):
    def _slice_without_masks(self, ind):
        try:
            if self.axis == 0:
                if len(self.list_of_containers) > 1:
                    return [c[ind] for c in self.list_of_containers]
                else:
                    return self.list_of_containers[0][ind]

            elif self.axis == 1:
                if len(self.list_of_containers) > 1:
                    return [c[:, ind] for c in self.list_of_containers]
                else:
                    return self.list_of_containers[0][:, ind]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")

    def _slice_with_masks(self, ind):
        try:
            raise ValueError("Not yet implemented")
            cs = self._slice_without_masks(ind)
            if self.axis == 0:
                ms = [np.ones_like(c[:, 0]) for c in cs]
            elif self.axis == 1:
                ms = [np.ones_like(c[:, :, 0]) for c in cs]
            assert len(cs) == len(ms)
            return [i for sublist in list(zip(cs, ms)) for i in sublist]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")


class list_iterator(base_iterator):
    def _slice_without_masks(self, ind):
        try:
            sliced_c = [np.asarray(c[ind]) for c in self.list_of_containers]
            for n in range(len(sliced_c)):
                sc = sliced_c[n]
                if not isinstance(sc, np.ndarray) or sc.dtype == np.object:
                    maxlen = max([len(i) for i in sc])
                    # Assume they at least have the same internal dtype
                    if len(sc[0].shape) > 1:
                        total_shape = (maxlen, sc[0].shape[1])
                    elif len(sc[0].shape) == 1:
                        total_shape = (maxlen, 1)
                    else:
                        raise ValueError("Unhandled array size in list")
                    if self.axis == 0:
                        raise ValueError("Unsupported axis of iteration")
                        new_sc = np.zeros((len(sc), total_shape[0],
                                           total_shape[1]))
                        new_sc = new_sc.squeeze().astype(sc[0].dtype)
                    else:
                        new_sc = np.zeros((total_shape[0], len(sc),
                                           total_shape[1]))
                        new_sc = new_sc.astype(sc[0].dtype)
                        for m, sc_i in enumerate(sc):
                            new_sc[:len(sc_i), m, :] = sc_i
                    sliced_c[n] = new_sc
            return sliced_c
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")

    def _slice_with_masks(self, ind):
        try:
            cs = self._slice_without_masks(ind)
            if self.axis == 0:
                ms = [np.ones_like(c[:, 0]) for c in cs]
            elif self.axis == 1:
                ms = [np.ones_like(c[:, :, 0]) for c in cs]
            assert len(cs) == len(ms)
            return [i for sublist in list(zip(cs, ms)) for i in sublist]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")


class list_of_array_iterator(base_iterator):
    """ List of 2D arrays, with masks. """
    def __init__(self, list_of_containers, minibatch_size,
                 start_index=0,
                 stop_index=np.inf,
                 list_of_extra_info=None,
                 randomize=False,
                 reshuffle_on_reset=False,
                 random_state=None):
        self.list_of_containers = list_of_containers
        self.list_of_extra_info = list_of_extra_info
        # all containers must be the same length
        for li in list_of_containers:
            assert len(li) == len(list_of_containers[0])
        if list_of_extra_info is not None:
            for lei in list_of_extra_info:
                assert len(lei) == len(self.list_of_containers[0])
        self.minibatch_size = minibatch_size

        if start_index < 0:
            start_index = len(list_of_containers[0]) + start_index

        if start_index < 1:
            start_index = int(start_index * len(list_of_containers[0]))
        self.start_index = start_index

        if stop_index < 0:
            stop_index = len(list_of_containers[0]) + stop_index

        if stop_index < 1:
            stop_index = int(stop_index * len(list_of_containers[0]))

        if stop_index > len(list_of_containers[0]):
            stop_index = len(list_of_containers[0])
        self.stop_index = stop_index

        if self.stop_index - self.start_index < minibatch_size:
            raise ValueError("Start, stop indices too close together (%i, %i)" % (start_index, stop_index))

        self.randomize = randomize
        self.reshuffle_on_reset = reshuffle_on_reset
        if reshuffle_on_reset is not False:
            raise ValueError("NYI")

        self._idx = np.arange(len(list_of_containers[0]), dtype="int32")
        self._current_offset = 0
        self.random_state = random_state
        # deterministic splitting to avoid training/testing issues...
        self._idx = self._idx[self.start_index:self.stop_index]

        if self.randomize:
            assert self.random_state is not None
            self.random_state.shuffle(self._idx)


    def reset(self):
        self._current_offset = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self._next_offset = self._current_offset + self.minibatch_size
        if self._next_offset >= len(self._idx):
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        ind = self._idx[self._current_offset:self._next_offset]
        self._current_offset = self._next_offset
        if self.list_of_extra_info is None:
            return self._slice_with_masks(ind)
        else:
            r = self._slice_with_masks(ind)
            return r + [[lei[i] for i in ind] for lei in self.list_of_extra_info]

    def _slice_with_masks(self, ind):
        try:
            cs = [[lc[i] if len(lc[i].shape) > 1 else lc[i][:, None] for i in ind]
                  for lc in self.list_of_containers]
            max_len = np.max([len(lci) for ci in cs for lci in ci])
            empties = [np.zeros((max_len, self.minibatch_size, csi[0].shape[-1]), dtype=np.float32)
                       for csi in cs]
            ms = [np.ones_like(e[:, :, 0]) for e in empties]
            for n in range(len(empties)):
                for c, i in enumerate(ind):
                    empties[n][:len(cs[n][c]), c] = cs[n][c]
                    ms[n][len(cs[n][c]):, c] = 0.
            return [i for sublist in list(zip(empties, ms)) for i in sublist]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")


class contiguous_list_of_array_iterator(base_iterator):
    """ List of 2D arrays, with masks. """
    def __init__(self, list_of_containers, minibatch_size,
                 truncation_length,
                 start_index=0,
                 stop_index=np.inf):
        self.list_of_containers = list_of_containers
        # all containers must be the same length
        for li in list_of_containers:
            assert len(li) == len(list_of_containers[0])
        self.minibatch_size = minibatch_size

        if start_index < 0:
            start_index = len(list_of_containers[0]) + start_index

        if start_index < 1:
            start_index = int(start_index * len(list_of_containers[0]))
        self.start_index = start_index

        if stop_index < 0:
            stop_index = len(list_of_containers[0]) + stop_index

        if stop_index < 1:
            stop_index = int(stop_index * len(list_of_containers[0]))

        if stop_index > len(list_of_containers[0]):
            stop_index = len(list_of_containers[0])
        self.stop_index = stop_index

        if self.stop_index - self.start_index < minibatch_size:
            raise ValueError("Start, stop indices too close together (%i, %i)" % (start_index, stop_index))

        self._joined_list_of_arrays = []
        for c in range(len(list_of_containers)):
            self._joined_list_of_arrays.append(np.concatenate(list_of_containers[c][start_index:stop_index], axis=0))

        len_match = len(self._joined_list_of_arrays[0])
        self.truncation_length = int(truncation_length)
        tl = self.truncation_length
        for c in range(len(list_of_containers)):
            assert len(self._joined_list_of_arrays[c]) == len_match
            la = self._joined_list_of_arrays[c]
            if len_match % tl != 0:
                diff = len_match // tl
                assert diff > 0
                new_len = diff * tl
                la = la[:new_len]
            if len(la.shape) == 2:
                shp = la.shape
                la = la.transpose(1, 0).reshape(shp[1], -1, truncation_length)
                la = la.transpose(2, 1, 0)
            else:
                raise ValueError("shape of array unhandled")
            self._joined_list_of_arrays[c] = la

        self._total_minibatch_size = self._joined_list_of_arrays[0].shape[1]
        self._chunk_size = self._total_minibatch_size // self.minibatch_size
        c = np.arange(0, self.minibatch_size, dtype=np.int32)
        self._offset_array = c * self._chunk_size
        self._current_offset = 0

    def reset(self):
        self._current_offset = 0
        c = np.arange(0, self.minibatch_size, dtype=np.int32)
        self._offset_array = c * self._chunk_size

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self._next_offset = self._current_offset + 1
        if self._next_offset > self._chunk_size:
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        parts = []
        for c in range(len(self._joined_list_of_arrays)):
            parts.append(self._joined_list_of_arrays[c][:, self._offset_array].astype("float32"))
        self._current_offset += 1
        self._offset_array += 1
        return parts


cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lower = "abcdefghijklmnopqrstuvwxyz"
alpha = "0123456789"
punc = " \\`,.?!=-+!@#$%^&*()[]{}:;'|/"
punc += '"'
all_chars = cap + lower + alpha + punc

class character_sequence_iterator(object):
    def __init__(self, sentence_iterator, minibatch_size,
                 truncation_length,
                 iterator_length=None,
                 start_index=0,
                 stop_index=np.inf,
                 valid_items=None,
                 stop_items=None,
                 extra_preproc_options=None):
        self.sentence_iterator = sentence_iterator
        self.minibatch_size = minibatch_size
        self.truncation_length = truncation_length
        self.start_index = start_index
        self.stop_index = stop_index
        if start_index != 0 or stop_index != np.inf:
            raise AttributeError("start_index and stop_index not yet supported")
        self.slice_start_ = start_index
        self.extra_preproc_options = extra_preproc_options
        if self.extra_preproc_options is not None:
            raise AttributeError("Extra preproc options not yet supported")
        self.n_classes = len(all_chars)
        rlu = {k: v for k, v in enumerate(all_chars)}
        lu = {v: k for k, v in rlu.items()}
        self.char_to_class = lu
        self.class_to_char = rlu
        def process(s):
            return [si for si in s]

        self._process = process
        if iterator_length is None:
            logger.info("No iterator_length provided for truncation mode.")
            logger.info("Calculating...")
            l = 0
            for s in sentence_iterator:
                l += len(s)
            self.iterator_length = l

        self.iterator_length -= self.iterator_length % (minibatch_size * truncation_length)
        self.sequence_length = self.iterator_length // minibatch_size

        def rg():
            minibatch_gens = []
            sl = self.sequence_length
            def r(): return (i for s in sentence_iterator for i in self._process(s))
            for i in range(minibatch_size):
                self.flat_gen = r()
                minibatch_gens.append(itertools.islice(self.flat_gen, i * sl, (i + 1) * sl))
            self.flat_gen = r()
            self.minibatch_gens = minibatch_gens
        rg()
        self.reset_gens = rg

    def reset(self):
        #self.slice_start_ = self.start_index
        self.reset_gens()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def _t(self, el):
        return self.char_to_class[el]

    def __next__(self):
        try:
            arr  = [[self._t(next(self.minibatch_gens[i]))
                    for n in range(self.truncation_length)]
                    for i in range(self.minibatch_size)]
            return np.asarray(arr).T.astype("float32")
        except StopIteration:
            self.reset()
            raise StopIteration("Stop index reached")

    def transform(self, list_of_strings):
        """
        list_of strings should be, well, a list of strings
        """
        arr = [self._t(si) for s in list_of_strings for si in self._process(s)]
        arr = np.asarray(arr)
        if len(arr.shape) == 1:
            arr = arr[None, :]
        return arr.T.astype("float32")

    def inverse_transform(self, index_array):
        """
        index_array should be 2D, shape (n_steps, minibatch_index)
        """
        return [[self.class_to_char[int(ai)]
                 for ai in index_array[:, i]]
                 for i in range(index_array.shape[1])]


"""
# based on http://alexbowe.com/au-naturale/
# Word Tokenization Regex adapted from NLTK book
# (?x) sets flag to allow comments in regexps
sentence_re = r'''(?x)
  # abbreviations, e.g. U.S.A. (with optional last period)
  ([A-Z])(\.[A-Z])+\.?
  # words with optional internal hyphens
  | \w+(-\w+)*
  # currency and percentages, e.g. $12.40, 82%
  | \$?\d+(\.\d+)?%?
  # ellipsis
  | \.\.\.
  # these are separate tokens
  | [][.,;"'?():-_`]
'''

compiled_sentence_re = re.compile(sentence_re)
"""
compiled_process_re = re.compile('[^A-Za-z ]+')

class word_sequence_iterator(object):
    def __init__(self, sentence_iterator, minibatch_size,
                 truncation_length,
                 iterator_length=None,
                 start_index=0,
                 stop_index=np.inf,
                 vocabulary=None,
                 max_vocabulary_size=5000,
                 tokenizer="default"):
        """
        'default' is lowercase
        """
        self.sentence_iterator = sentence_iterator
        self.minibatch_size = minibatch_size
        self.truncation_length = truncation_length
        self.start_index = start_index
        self.stop_index = stop_index
        if start_index != 0 or stop_index != np.inf:
            raise AttributeError("start_index and stop_index not yet supported")
        self.slice_start_ = start_index
        self.tokenizer = tokenizer
        if tokenizer != "default":
            raise ValueError("Unsupported tokenizer option")
        self.vocabulary = None

        self._unk = "<UNK>"
        self._sos = "<START>"
        self._eos = "<EOS>"
        def process(s):
            s = s.lower()
            toks = re.sub(compiled_process_re, "", s).split(" ")
            # why is this infinity times slower
            #toks = nltk.regexp_tokenize(s, compiled_sentence_re)
            toks += [self._eos]
            return toks
        self._process = process

        calculated_iterator_length = 0
        words = Counter()
        if vocabulary is None:
            logger.info("No vocabulary provided for truncation mode.")
            logger.info("Calculating...")
            sg = (s for s in sentence_iterator)
            for n, s in enumerate(sg):
                toks = self._process(s)
                words.update(toks)
                calculated_iterator_length += len(toks)
                if n % 1000 == 0:
                    logger.info("Processed %s sentences so far" % n)
        elif vocabulary is not None:
            raise ValueError("TODO: Support fixed vocabulary!")

        self.iterator_length = calculated_iterator_length
        self.words_counter = words

        self.max_vocabulary_size = max_vocabulary_size

        v = sorted([w[0] for w in words.most_common(max_vocabulary_size - 1)]) + [self._unk]
        rlu = {k: v for k, v in enumerate(v)}
        lu = {v: k for k, v in rlu.items()}
        self.word_to_class = lu
        self.class_to_word = rlu
        self.vocabulary = v
        self.n_classes = len(v)

        self.iterator_length -= self.iterator_length % (minibatch_size * truncation_length)
        self.sequence_length = self.iterator_length // minibatch_size

        def rg():
            minibatch_gens = []
            sl = self.sequence_length
            def r(): return (i for s in sentence_iterator for i in self._process(s))
            for i in range(minibatch_size):
                self.flat_gen = r()
                minibatch_gens.append(itertools.islice(self.flat_gen, i * sl, (i + 1) * sl))
            self.flat_gen = r()
            self.minibatch_gens = minibatch_gens
        rg()
        self.reset_gens = rg

    def reset(self):
        #self.slice_start_ = self.start_index
        self.reset_gens()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def _t(self, el):
        return self.word_to_class.get(el, self.word_to_class[self._unk])

    def __next__(self):
        try:
            out = []
            for i in range(self.minibatch_size):
                a = [self._t(next(self.minibatch_gens[i])) for n in range(self.truncation_length)]
                out.append(a)
            return np.asarray(out).T.astype("float32")
        except StopIteration:
            self.reset()
            raise StopIteration("Stop index reached")

    def transform(self, list_of_strings):
        """
        list_of strings should be, well, a list of strings
        """
        arr = [self._t(si) for s in list_of_strings for si in self._process(s)]
        arr = np.asarray(arr)
        if len(arr.shape) == 1:
            arr = arr[None, :]
        return arr.T.astype("float32")

    def inverse_transform(self, index_array):
        """
        index_array should be 2D, shape (n_steps, minibatch_index)
        """
        return [[self.class_to_word[int(ai)]
                 for ai in index_array[:, i]]
                 for i in range(index_array.shape[1])]
