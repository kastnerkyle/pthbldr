# Author: Kyle Kastner
# License: BSD 3-clause
# Thanks to Jose (@sotelo) for tons of guidance and debug help
# Credit also to Junyoung (@jych) and Shawn (@shawntan) for help/utility funcs
# Strangeness in init could be from onehots, via @igul222. Ty init for one hot layer as N(0, 1) just as in embedding
# since oh.dot(w) is basically an embedding
import os
import re
import tarfile
from collections import Counter
import sys
import pickle
import numpy as np
from scipy import linalg


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
        self.one_hot_class_size = one_hot_class_size
        if one_hot_class_size is not None:
            assert len(self.one_hot_class_size) == len(list_of_containers)

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
        ind = slice(self.slice_start_, self.slice_end_)
        self.slice_start_ = self.slice_end_
        if self.make_mask is False:
            res = self._slice_without_masks(ind)
            if not all([self.minibatch_size in r.shape for r in res]):
                # TODO: Check that things are even
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res
        else:
            res = self._slice_with_masks(ind)
            # TODO: Check that things are even
            if not all([self.minibatch_size in r.shape for r in res]):
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res

    def _slice_without_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")

    def _slice_with_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")


class list_iterator(base_iterator):
    def _slice_without_masks(self, ind):
        sliced_c = [np.asarray(c[ind]) for c in self.list_of_containers]
        if min([len(i) for i in sliced_c]) < self.minibatch_size:
            self.reset()
            raise StopIteration("Invalid length slice")
        for n in range(len(sliced_c)):
            sc = sliced_c[n]
            if self.one_hot_class_size is not None:
                convert_it = self.one_hot_class_size[n]
                if convert_it is not None:
                    raise ValueError("One hot conversion not implemented")
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

    def _slice_with_masks(self, ind):
        cs = self._slice_without_masks(ind)
        if self.axis == 0:
            ms = [np.ones_like(c[:, 0]) for c in cs]
        elif self.axis == 1:
            ms = [np.ones_like(c[:, :, 0]) for c in cs]
        assert len(cs) == len(ms)
        return [i for sublist in list(zip(cs, ms)) for i in sublist]



def check_fetch_iamondb():
    """ Check for IAMONDB data

        This dataset cannot be downloaded automatically!
    """
    partial_path = os.sep + "Tmp" + os.sep + "kastner" + os.sep + "iamondb"
    ascii_path = os.path.join(partial_path, "lineStrokes-all.tar.gz")
    lines_path = os.path.join(partial_path, "ascii-all.tar.gz")
    files_path = os.path.join(partial_path, "task1.tar.gz")
    for p in [ascii_path, lines_path, files_path]:
        if not os.path.exists(p):
            files = "lineStrokes-all.tar.gz, ascii-all.tar.gz, and task1.tar.gz"
            url = "http://www.iam.unibe.ch/fki/databases/"
            url += "iam-on-line-handwriting-database/"
            url += "download-the-iam-on-line-handwriting-database"
            err = "Path %s does not exist!" % p
            err += " Download the %s files from %s" % (files, url)
            err += " and place them in the directory %s" % partial_path
            raise ValueError(err)
    return partial_path


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    labels_shape = labels_dense.shape
    labels_dense = labels_dense.reshape([-1])
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape(labels_shape+(num_classes,))
    return labels_one_hot


def tokenize_ind(phrase, vocabulary):
    phrase = phrase + " "
    vocabulary_size = len(vocabulary.keys())
    phrase = [vocabulary[char_] for char_ in phrase]
    phrase = np.array(phrase, dtype='int32').ravel()
    phrase = dense_to_one_hot(phrase, vocabulary_size)
    return phrase


def fetch_iamondb():
    from lxml import etree
    partial_path = check_fetch_iamondb()
    pickle_path = os.path.join(partial_path, "iamondb_saved.pkl")
    if not os.path.exists(pickle_path):
        input_file = os.path.join(partial_path, 'lineStrokes-all.tar.gz')
        raw_data = tarfile.open(input_file)
        transcript_files = []
        strokes = []
        idx = 0
        for member in raw_data.getmembers():
            if member.isreg():
                transcript_files.append(member.name)
                content = raw_data.extractfile(member)
                tree = etree.parse(content)
                root = tree.getroot()
                content.close()
                points = []
                for StrokeSet in root:
                    for i, Stroke in enumerate(StrokeSet):
                        for Point in Stroke:
                            points.append([i,
                                int(Point.attrib['x']),
                                int(Point.attrib['y'])])
                points = np.array(points)
                points[:, 2] = -points[:, 2]
                change_stroke = points[:-1, 0] != points[1:, 0]
                pen_up = points[:, 0] * 0
                pen_up[:-1][change_stroke] = 1
                pen_up[-1] = 1
                points[:, 0] = pen_up
                strokes.append(points)
                idx += 1

        strokes_bp = strokes

        strokes = [x[1:] - x[:-1] for x in strokes]
        strokes = [np.vstack([[0, 0, 0], x]) for x in strokes]

        for i, stroke in enumerate(strokes):
            strokes[i][:, 0] = strokes_bp[i][:, 0]

        # Computing mean and variance seems to not be necessary.
        # Training is going slower than just scaling.

        data_mean = np.array([0., 0., 0.])
        data_std = np.array([1., 20., 20.])

        strokes = [(x - data_mean) / data_std for x in strokes]

        transcript_files = [x.split(os.sep)[-1]
                            for x in transcript_files]
        transcript_files = [re.sub('-[0-9][0-9].xml', '.txt', x)
                            for x in transcript_files]

        counter = Counter(transcript_files)

        input_file = os.path.join(partial_path, 'ascii-all.tar.gz')

        raw_data = tarfile.open(input_file)
        member = raw_data.getmembers()[10]

        all_transcripts = []
        for member in raw_data.getmembers():
            if member.isreg() and member.name.split("/")[-1] in transcript_files:
                fp = raw_data.extractfile(member)

                cleaned = [t.strip() for t in fp.readlines()
                        if t != '\r\n'
                        and t != '\n'
                        and t != '\r\n'
                        and t.strip() != '']

                # Try using CSR
                idx = [n for n, li in enumerate(cleaned) if li == "CSR:"][0]
                cleaned_sub = cleaned[idx + 1:]
                corrected_sub = []
                for li in cleaned_sub:
                    # Handle edge case with %%%%% meaning new line?
                    if "%" in li:
                        li2 = re.sub('\%\%+', '%', li).split("%")
                        li2 = [l.strip() for l in li2]
                        corrected_sub.extend(li2)
                    else:
                        corrected_sub.append(li)

                if counter[member.name.split("/")[-1]] != len(corrected_sub):
                    pass

                all_transcripts.extend(corrected_sub)

        # Last file transcripts are almost garbage
        all_transcripts[-1] = 'A move to stop'
        all_transcripts.append('garbage')
        all_transcripts.append('A move to stop')
        all_transcripts.append('garbage')
        all_transcripts.append('A move to stop')
        all_transcripts.append('A move to stop')
        all_transcripts.append('Marcus Luvki')
        all_transcripts.append('Hallo Well')
        # Remove outliers and big / small sequences
        # Makes a BIG difference.
        filter_ = [len(x) <= 1200 and len(x) >= 300 and
                   x.max() <= 100 and x.min() >= -50 for x in strokes]

        strokes = [x for x, cond in zip(strokes, filter_) if cond]
        all_transcripts = [x for x, cond in
                           zip(all_transcripts, filter_) if cond]

        num_examples = len(strokes)

        # Shuffle for train/validation/test division
        rng = np.random.RandomState(1999)
        shuffle_idx = rng.permutation(num_examples)

        strokes = [strokes[x] for x in shuffle_idx]
        all_transcripts = [all_transcripts[x] for x in shuffle_idx]

        all_chars = ([chr(ord('a') + i) for i in range(26)] +
                     [chr(ord('A') + i) for i in range(26)] +
                     [chr(ord('0') + i) for i in range(10)] +
                     [',', '.', '!', '?', ';', ' ', ':'] +
                     ["#", '&', '+', '[', ']', '{', '}'] +
                     ["/", "*"] +
                     ['(', ')', '"', "'", '-', '<UNK>'])

        code2char = dict(enumerate(all_chars))
        char2code = {v: k for k, v in code2char.items()}
        vocabulary_size = len(char2code.keys())
        unk_char = '<UNK>'

        y = []
        for n, li in enumerate(all_transcripts):
            y.append(tokenize_ind(li, char2code))

        pickle_dict = {}
        pickle_dict["target_phrases"] = all_transcripts
        pickle_dict["vocabulary_size"] = vocabulary_size
        pickle_dict["vocabulary_tokenizer"] = tokenize_ind
        pickle_dict["vocabulary"] = char2code
        pickle_dict["data"] = strokes
        pickle_dict["target"] = y
        f = open(pickle_path, "wb")
        pickle.dump(pickle_dict, f, -1)
        f.close()
    with open(pickle_path, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict


def plot_lines_iamondb_example(X, title="", save_name=None):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    x = np.cumsum(X[:, 1])
    y = np.cumsum(X[:, 2])

    size_x = x.max() - x.min()
    size_y = y.max() - y.min()

    f.set_size_inches(5 * size_x / size_y, 5)
    cuts = np.where(X[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=1.5)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title(title)

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)


def implot(arr, title="", cmap="gray", save_name=None):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    ax.matshow(arr, cmap=cmap)
    plt.axis("off")

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)

    x1 = arr.shape[0]
    y1 = arr.shape[1]
    asp = autoaspect(x1, y1)
    ax.set_aspect(asp)
    plt.title(title)
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)
