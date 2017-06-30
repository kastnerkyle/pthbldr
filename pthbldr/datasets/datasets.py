# Author: Kyle Kastner
# License: BSD 3-clause
# Ideas from Junyoung Chung and Kyunghyun Cho
# See https://github.com/jych/cle for a library in this style
import numpy as np
from collections import Counter
from scipy.io import loadmat, wavfile
from scipy.linalg import svd
from functools import reduce
from ..core import safe_zip, get_logger
from .audio_utils import stft
import shutil
import string
import tarfile
import fnmatch
import zipfile
import gzip
import os
import json
import re
import csv
import time
import signal
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle

floatX = "float32"
logger = get_logger()

regex = re.compile('[%s]' % re.escape(string.punctuation))

bitmap_characters = np.array([
    0x0,
    0x808080800080000,
    0x2828000000000000,
    0x287C287C280000,
    0x81E281C0A3C0800,
    0x6094681629060000,
    0x1C20201926190000,
    0x808000000000000,
    0x810202010080000,
    0x1008040408100000,
    0x2A1C3E1C2A000000,
    0x8083E08080000,
    0x81000,
    0x3C00000000,
    0x80000,
    0x204081020400000,
    0x1824424224180000,
    0x8180808081C0000,
    0x3C420418207E0000,
    0x3C420418423C0000,
    0x81828487C080000,
    0x7E407C02423C0000,
    0x3C407C42423C0000,
    0x7E04081020400000,
    0x3C423C42423C0000,
    0x3C42423E023C0000,
    0x80000080000,
    0x80000081000,
    0x6186018060000,
    0x7E007E000000,
    0x60180618600000,
    0x3844041800100000,
    0x3C449C945C201C,
    0x1818243C42420000,
    0x7844784444780000,
    0x3844808044380000,
    0x7844444444780000,
    0x7C407840407C0000,
    0x7C40784040400000,
    0x3844809C44380000,
    0x42427E4242420000,
    0x3E080808083E0000,
    0x1C04040444380000,
    0x4448507048440000,
    0x40404040407E0000,
    0x4163554941410000,
    0x4262524A46420000,
    0x1C222222221C0000,
    0x7844784040400000,
    0x1C222222221C0200,
    0x7844785048440000,
    0x1C22100C221C0000,
    0x7F08080808080000,
    0x42424242423C0000,
    0x8142422424180000,
    0x4141495563410000,
    0x4224181824420000,
    0x4122140808080000,
    0x7E040810207E0000,
    0x3820202020380000,
    0x4020100804020000,
    0x3808080808380000,
    0x1028000000000000,
    0x7E0000,
    0x1008000000000000,
    0x3C023E463A0000,
    0x40407C42625C0000,
    0x1C20201C0000,
    0x2023E42463A0000,
    0x3C427E403C0000,
    0x18103810100000,
    0x344C44340438,
    0x2020382424240000,
    0x800080808080000,
    0x800180808080870,
    0x20202428302C0000,
    0x1010101010180000,
    0x665A42420000,
    0x2E3222220000,
    0x3C42423C0000,
    0x5C62427C4040,
    0x3A46423E0202,
    0x2C3220200000,
    0x1C201804380000,
    0x103C1010180000,
    0x2222261A0000,
    0x424224180000,
    0x81815A660000,
    0x422418660000,
    0x422214081060,
    0x3C08103C0000,
    0x1C103030101C0000,
    0x808080808080800,
    0x38080C0C08380000,
    0x324C000000,
], dtype=np.uint64)

bitmap = np.unpackbits(bitmap_characters.view(np.uint8)).reshape(
    bitmap_characters.shape[0], 8, 8)
bitmap = bitmap[:, ::-1, :]
all_vocabulary_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTU"
all_vocabulary_chars += "VWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
char_mapping = {c: i for i, c in enumerate(all_vocabulary_chars)}


def string_to_character_index(string):
    return np.asarray([char_mapping[c] for c in string])


def get_dataset_dir(dataset_name, data_dir=None, folder=None, create_dir=True):
    """ Get dataset directory path """
    if not data_dir:
        data_dir = os.getenv("PTHBLDR_DATA", os.path.join(
            os.path.expanduser("~"), "pthbldr_data"))
    if folder is None:
        data_dir = os.path.join(data_dir, dataset_name)
    else:
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir) and create_dir:
        os.makedirs(data_dir)
    return data_dir


def download(url, server_fname, local_fname=None, progress_update_percentage=5):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def check_fetch_uci_words():
    """ Check for UCI vocabulary """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    url += 'bag-of-words/'
    partial_path = get_dataset_dir("uci_words")
    full_path = os.path.join(partial_path, "uci_words.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        # Download all 5 vocabularies and zip them into a file
        all_vocabs = ['vocab.enron.txt', 'vocab.kos.txt', 'vocab.nips.txt',
                      'vocab.nytimes.txt', 'vocab.pubmed.txt']
        for vocab in all_vocabs:
            dl_url = url + vocab
            download(dl_url, os.path.join(partial_path, vocab),
                     progress_update_percentage=1)

            def zipdir(path, zipf):
                # zipf is zipfile handle
                for root, dirs, files in os.walk(path):
                    for f in files:
                        if "vocab" in f:
                            zipf.write(os.path.join(root, f))

            zipf = zipfile.ZipFile(full_path, 'w')
            zipdir(partial_path, zipf)
            zipf.close()
    return full_path


def fetch_uci_words():
    """ Returns UCI vocabulary text. """
    data_path = check_fetch_uci_words()
    all_data = []
    with zipfile.ZipFile(data_path, "r") as f:
        for name in f.namelist():
            if ".txt" not in name:
                # Skip README
                continue
            data = f.read(name)
            data = data.split("\n")
            data = [l.strip() for l in data if l != ""]
            all_data.extend(data)
    return list(set(all_data))


def whitespace_tokenizer(line):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', line) if x.strip()]


def _parse_stories(lines, only_supporting=False):
    """ Preprocessing code modified from Keras and Stephen Merity
    http://smerity.com/articles/2015/keras_qa.html
    https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

    Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are
    kept.
    """
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = whitespace_tokenizer(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = whitespace_tokenizer(line)
            story.append(sent)
    return data


def _get_stories(f, only_supporting=False, max_length=None):
    """ Preprocessing code modified from Keras and Stephen Merity
    http://smerity.com/articles/2015/keras_qa.html
    https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

    Given a file name, read the file, retrieve the stories, and then convert
    the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be
    discarded.
    """
    data = _parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data


def _vectorize_stories(data, vocab_size, word_idx):
    """ Preprocessing code modified from Keras and Stephen Merity
    http://smerity.com/articles/2015/keras_qa.html
    https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py
    """
    X = []
    Xq = []
    y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        yi = np.zeros(vocab_size)
        yi[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        y.append(yi)
    return X, Xq, np.array(y)


def check_fetch_babi():
    """ Check for babi task data

    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    J. Weston, A. Bordes, S. Chopra, T. Mikolov, A. Rush
    http://arxiv.org/abs/1502.05698
    """
    url = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"
    partial_path = get_dataset_dir("babi")
    full_path = os.path.join(partial_path, "tasks_1-20_v1-2.tar.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_babi(task_number=2):
    """ Fetch data for babi tasks described in
    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    J. Weston, A. Bordes, S. Chopra, T. Mikolov, A. Rush
    http://arxiv.org/abs/1502.05698

    Preprocessing code modified from Keras and Stephen Merity
    http://smerity.com/articles/2015/keras_qa.html
    https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

    n_samples : 1000 - 10000 (task dependent)

    Returns
    -------
    summary : dict
        A dictionary cantaining data

        summary["stories"] : list
            List of list of ints

        summary["queries"] : list
            List of list of ints

        summary["target"] : list
            List of list of int

        summary["train_indices"] : array
            Indices for training samples

        summary["valid_indices"] : array
            Indices for validation samples

        summary["vocabulary_size"] : int
            Total vocabulary size
    """

    data_path = check_fetch_babi()
    tar = tarfile.open(data_path)
    if task_number == 2:
        challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
    else:
        raise ValueError("No other supported tasks at this time")
    # QA2 with 1000 samples
    train = _get_stories(tar.extractfile(challenge.format('train')))
    test = _get_stories(tar.extractfile(challenge.format('test')))

    vocab = sorted(reduce(lambda x, y: x | y, (
        set(story + q + [answer]) for story, q, answer in train + test)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    # story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    # query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    X_story, X_query, y_answer = _vectorize_stories(train, vocab_size, word_idx)
    valid_X_story, valid_X_query, valid_y_answer = _vectorize_stories(
        test, vocab_size, word_idx)
    train_indices = np.arange(len(y_answer))
    valid_indices = np.arange(len(valid_y_answer)) + len(y_answer)

    X_story, X_query, y_answer = _vectorize_stories(train + test, vocab_size,
                                                    word_idx)
    return {"stories": X_story,
            "queries": X_query,
            "target": y_answer,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "vocabulary_size": vocab_size}


def check_fetch_fruitspeech():
    """ Check for fruitspeech data

    Recorded by Hakon Sandsmark
    """
    url = "https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz"
    partial_path = get_dataset_dir("fruitspeech")
    full_path = os.path.join(partial_path, "audio.tar.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    audio_path = os.path.join(partial_path, "audio")
    if not os.path.exists(audio_path):
        tar = tarfile.open(full_path)
        os.chdir(partial_path)
        tar.extractall()
        tar.close()
    return audio_path


def fetch_fruitspeech():
    """ Check for fruitspeech data

    Recorded by Hakon Sandsmark

    Returns
    -------
    summary : dict
        A dictionary cantaining data

        summary["data"] : list
            List of list of ints

        summary["specgrams"] : list
            List of arrays in (n_frames, n_features) format

        summary["target_names"] : list
            List of strings

        summary["target"] : list
            List of list of int

        summary["train_indices"] : array
            Indices for training samples

        summary["valid_indices"] : array
            Indices for validation samples

        summary["vocabulary_size"] : int
            Total vocabulary size

        summary["vocabulary"] : string
            The whole vocabulary as a string
    """

    data_path = check_fetch_fruitspeech()
    audio_matches = []
    for root, dirnames, filenames in os.walk(data_path):
        for filename in fnmatch.filter(filenames, '*.wav'):
            audio_matches.append(os.path.join(root, filename))
    all_chars = []
    all_words = []
    all_data = []
    all_specgram_data = []
    for wav_path in audio_matches:
        # Convert chars to int classes
        word = wav_path.split(os.sep)[-1][:-6]
        chars = string_to_character_index(word)
        fs, d = wavfile.read(wav_path)
        d = d.astype("int32")
        # Preprocessing from A. Graves "Towards End-to-End Speech
        # Recognition"
        Pxx = 10. * np.log10(np.abs(stft(d, fftsize=128))).astype(
            floatX)
        all_data.append(d)
        all_specgram_data.append(Pxx)
        all_chars.append(chars)
        all_words.append(word)
    vocabulary_size = len(all_vocabulary_chars)
    # Shuffle data
    all_lists = list(safe_zip(all_data, all_specgram_data, all_chars,
                              all_words))
    random_state = np.random.RandomState(1999)
    random_state.shuffle(all_lists)
    all_data, all_specgram_data, all_chars, all_words = zip(*all_lists)
    wordset = list(set(all_words))
    train_matches = []
    valid_matches = []
    for w in wordset:
        matches = [n for n, i in enumerate(all_words) if i == w]
        # Hold out ~25% of the data, keeping some of every class
        train_matches.append(matches[:-4])
        valid_matches.append(matches[-4:])
    train_indices = np.array(sorted(
        [r for i in train_matches for r in i])).astype("int32")
    valid_indices = np.array(sorted(
        [r for i in valid_matches for r in i])).astype("int32")

    # reorganize into contiguous blocks
    def reorg(list_):
        ret = [list_[i] for i in train_indices] + [
            list_[i] for i in valid_indices]
        return np.asarray(ret)
    all_data = reorg(all_data)
    all_specgram_data = reorg(all_specgram_data)
    all_chars = reorg(all_chars)
    all_words = reorg(all_words)
    # after reorganizing finalize indices
    train_indices = np.arange(len(train_indices))
    valid_indices = np.arange(len(valid_indices)) + len(train_indices)
    return {"data": all_data,
            "specgrams": all_specgram_data,
            "target": all_chars,
            "target_names": all_words,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "vocabulary_size": vocabulary_size,
            "vocabulary": all_vocabulary_chars}


def check_fetch_lovecraft():
    """ Check for lovecraft data """
    url = 'https://dl.dropboxusercontent.com/u/15378192/lovecraft_fiction.zip'
    partial_path = get_dataset_dir("lovecraft")
    full_path = os.path.join(partial_path, "lovecraft_fiction.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_lovecraft():
    """ All the fiction text written by H. P. Lovecraft

    n_samples : 40363
    n_chars : 84 (Counting UNK, EOS)
    n_words : 26644 (Counting UNK)

    Returns
    -------
    summary : dict
        A dictionary cantaining data

        summary["data"] : list, shape (40363,)
            List of strings

        summary["words"] : list,
            List of strings

    """
    data_path = check_fetch_lovecraft()
    all_data = []
    all_words = Counter()
    with zipfile.ZipFile(data_path, "r") as f:
        for name in f.namelist():
            if ".txt" not in name:
                # Skip README
                continue
            data = f.read(name)
            data = data.split("\n")
            data = [l.strip() for l in data if l != ""]
            words = [w for l in data for w in regex.sub('', l.lower()).split(
                " ") if w != ""]
            all_data.extend(data)
            all_words.update(words)
    return {"data": all_data,
            "words": all_words.keys()}


def load_mountains():
    """
    H. P. Lovecraft's At The Mountains Of Madness

    Used for tests which need text data

    n_samples : 3575
    n_chars : 75 (Counting UNK, EOS)
    n_words : 6346 (Counting UNK)

    Returns
    -------
    summary : dict
        A dictionary cantaining data

        summary["data"] : list, shape (3575, )
            List of strings

        summary["words"] : list,

    """
    module_path = os.path.dirname(__file__)
    all_words = Counter()
    with open(os.path.join(module_path, 'data', 'mountains.txt')) as f:
        data = f.read()
        data = data.split("\n")
        data = [l.strip() for l in data if l != ""]
        words = [w for l in data for w in regex.sub('', l.lower()).split(
            " ") if l != ""]
        all_words.update(words)
    return {"data": data,
            "words": all_words.keys()}


def check_fetch_fer():
    """ Check that fer faces are downloaded """
    url = 'https://dl.dropboxusercontent.com/u/15378192/fer2013.tar.gz'
    partial_path = get_dataset_dir("fer")
    full_path = os.path.join(partial_path, "fer2013.tar.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_fer():
    """
    Flattened 48x48 fer faces with pixel values in [0 - 1]

    n_samples : 35888
    n_features : 2304

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : array, shape (35888, 2304)
            The flattened data for FER

    """
    data_path = check_fetch_fer()
    t = tarfile.open(data_path, 'r')
    f = t.extractfile(t.getnames()[0])
    reader = csv.reader(f)
    valid_indices = 2 * 3859
    data = np.zeros((35888, 48 * 48), dtype=floatX)
    target = np.zeros((35888,), dtype="int32")
    header = None
    for n, row in enumerate(reader):
        if n % 1000 == 0:
            print("Reading sample %i" % n)
        if n == 0:
            header = row
        else:
            target[n] = int(row[0])
            data[n] = np.array(map(float, row[1].split(" "))) / 255.
    train_indices = np.arange(23709)
    valid_indices = np.arange(23709, len(data))
    train_mean0 = data[train_indices].mean(axis=0)
    saved_pca_path = os.path.join(get_dataset_dir("fer"), "FER_PCA.npy")
    if not os.path.exists(saved_pca_path):
        print("Saved PCA not found for FER, computing...")
        U, S, V = svd(data[train_indices] - train_mean0, full_matrices=False)
        train_pca = V
        np.save(saved_pca_path, train_pca)
    else:
        train_pca = np.load(saved_pca_path)
    return {"data": data,
            "target": target,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "mean0": train_mean0,
            "pca_matrix": train_pca}


def check_fetch_tfd():
    """ Check that tfd faces are downloaded """
    partial_path = get_dataset_dir("tfd")
    full_path = os.path.join(partial_path, "TFD_48x48.mat")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        raise ValueError("Put TFD_48x48 in %s" % str(partial_path))
    return full_path


def fetch_tfd():
    """
    Flattened 48x48 TFD faces with pixel values in [0 - 1]

    n_samples : 102236
    n_features : 2304

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : array, shape (102236, 2304)
            The flattened data for TFD

    """
    data_path = check_fetch_tfd()
    matfile = loadmat(data_path)
    all_data = matfile['images'].reshape(len(matfile['images']), -1) / 255.
    all_data = all_data.astype(floatX)
    train_indices = np.arange(0, 90000)
    valid_indices = np.arange(0, 10000) + len(train_indices) + 1
    test_indices = np.arange(valid_indices[-1] + 1, len(all_data))
    train_data = all_data[train_indices]
    train_mean0 = train_data.mean(axis=0)
    random_state = np.random.RandomState(1999)
    subset_indices = random_state.choice(train_indices, 25000, replace=False)
    saved_pca_path = os.path.join(get_dataset_dir("tfd"), "TFD_PCA.npy")
    if not os.path.exists(saved_pca_path):
        print("Saved PCA not found for TFD, computing...")
        U, S, V = svd(train_data[subset_indices] - train_mean0,
                      full_matrices=False)
        train_pca = V
        np.save(saved_pca_path, train_pca)
    else:
        train_pca = np.load(saved_pca_path)
    return {"data": all_data,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "test_indices": test_indices,
            "mean0": train_mean0,
            "pca_matrix": train_pca}


def check_fetch_frey():
    """ Check that frey faces are downloaded """
    url = 'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat'
    partial_path = get_dataset_dir("frey")
    full_path = os.path.join(partial_path, "frey_rawface.mat")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_frey():
    """
    Flattened 20x28 frey faces with pixel values in [0 - 1]

    n_samples : 1965
    n_features : 560

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : array, shape (1965, 560)

    """
    data_path = check_fetch_frey()
    matfile = loadmat(data_path)
    all_data = (matfile['ff'] / 255.).T
    all_data = all_data.astype(floatX)
    return {"data": all_data,
            "mean0": all_data.mean(axis=0),
            "var0": all_data.var(axis=0)}


def check_fetch_mnist():
    """ Check that mnist is downloaded. May need fixing for py3 compat """
    # py3k version is available at mnist_py3k.pkl.gz ... might need to fix
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    partial_path = get_dataset_dir("mnist")
    full_path = os.path.join(partial_path, "mnist.pkl.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_mnist():
    """
    Flattened 28x28 mnist digits with pixel values in [0 - 1]

    n_samples : 70000
    n_feature : 784

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : array, shape (70000, 784)
        summary["target"] : array, shape (70000,)
        summary["images"] : array, shape (70000, 1, 28, 28)
        summary["train_indices"] : array, shape (50000,)
        summary["valid_indices"] : array, shape (10000,)
        summary["test_indices"] : array, shape (10000,)

    """
    data_path = check_fetch_mnist()
    f = gzip.open(data_path, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()
    train_indices = np.arange(0, len(train_set[0]))
    valid_indices = np.arange(0, len(valid_set[0])) + train_indices[-1] + 1
    test_indices = np.arange(0, len(test_set[0])) + valid_indices[-1] + 1
    data = np.concatenate((train_set[0], valid_set[0], test_set[0]),
                          axis=0).astype()
    target = np.concatenate((train_set[1], valid_set[1], test_set[1]),
                            axis=0).astype(np.int32)
    return {"data": data,
            "target": target,
            "images": data.reshape((len(data), 1, 28, 28)),
            "train_indices": train_indices.astype(np.int32),
            "valid_indices": valid_indices.astype(np.int32),
            "test_indices": test_indices.astype(np.int32)}


def check_fetch_binarized_mnist():
    raise ValueError("Binarized MNIST has no labels! Do not use")
    """
    # public version
    url = 'https://github.com/mgermain/MADE/releases/download/ICML2015/'
    url += 'binarized_mnist.npz'
    partial_path = get_dataset_dir("binarized_mnist")
    fname = "binarized_mnist.npz"
    full_path = os.path.join(partial_path, fname)
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    # personal version
    url = "https://dl.dropboxusercontent.com/u/15378192/binarized_mnist_%s.npy"
    fname = "binarized_mnist_%s.npy"
    for s in ["train", "valid", "test"]:
        full_path = os.path.join(partial_path, fname % s)
        if not os.path.exists(partial_path):
            os.makedirs(partial_path)
        if not os.path.exists(full_path):
            download(url % s, full_path, progress_update_percentage=1)
    return partial_path
    """


def fetch_binarized_mnist():
    """
    Flattened 28x28 mnist digits with pixel of either 0 or 1, sampled from
    binomial distribution defined by the original MNIST values

    n_samples : 70000
    n_features : 784

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : array, shape (70000, 784)
        summary["target"] : array, shape (70000,)
        summary["train_indices"] : array, shape (50000,)
        summary["valid_indices"] : array, shape (10000,)
        summary["test_indices"] : array, shape (10000,)

    """
    mnist = fetch_mnist()
    random_state = np.random.RandomState(1999)

    def get_sampled(arr):
        # make sure that a pixel can always be turned off
        return random_state.binomial(1, arr * 255 / 256., size=arr.shape)

    data = get_sampled(mnist["data"]).astype(floatX)
    return {"data": data,
            "target": mnist["target"],
            "train_indices": mnist["train_indices"],
            "valid_indices": mnist["valid_indices"],
            "test_indices": mnist["test_indices"]}


def make_sincos(n_timesteps, n_pairs):
    """
    Generate a 2D array of sine and cosine pairs at random frequencies and
    linear phase offsets depending on position in minibatch.

    Used for simple testing of RNN algorithms.

    Parameters
    ----------
    n_timesteps : int
        number of timesteps

    n_pairs : int
        number of sine, cosine pairs to generate

    Returns
    -------
    pairs : array, shape (n_timesteps, n_pairs, 2)
        A minibatch of sine, cosine pairs with the RNN minibatch converntion
        (timestep, sample, feature).
    """
    n_timesteps = int(n_timesteps)
    n_pairs = int(n_pairs)
    random_state = np.random.RandomState(1999)
    frequencies = 5 * random_state.rand(n_pairs) + 1
    frequency_base = np.arange(n_timesteps) / (2 * np.pi)
    steps = frequency_base[:, None] * frequencies[None]
    phase_offset = np.arange(n_pairs) / (2 * np.pi)
    sines = np.sin(steps + phase_offset)
    cosines = np.sin(steps + phase_offset + np.pi / 2)
    sines = sines[:, :, None]
    cosines = cosines[:, :, None]
    pairs = np.concatenate((sines, cosines), axis=-1).astype(
        floatX)
    return pairs


def load_iris():
    """
    Load and return the iris dataset (classification).

    This is basically the sklearn dataset loader, except returning a dictionary.

    n_samples : 150
    n_features : 4

    Returns
    -------
    summary : dict
        A dictionary cantaining data and target labels

        summary["data"] : array, shape (150, 4)
            The data for iris

        summary["target"] : array, shape (150,)
            The classification targets

    """
    module_path = os.path.dirname(__file__)
    with open(os.path.join(module_path, 'data', 'iris.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features), dtype=floatX)
        target = np.empty((n_samples,), dtype=np.int32)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=floatX)
            target[i] = np.asarray(ir[-1], dtype=np.int32)

    return {"data": data, "target": target}


def load_digits():
    """
    Load and return the digits dataset (classification).

    This is basically the sklearn dataset loader, except returning a dictionary.

    n_samples : 1797
    n_features : 64

    Returns
    -------
    summary : dict
        A dictionary cantaining data and target labels

        summary["data"] : array, shape (1797, 64)
            The data for digits

        summary["target"] : array, shape (1797,)
            The classification targets

    """

    module_path = os.path.dirname(__file__)
    data = np.loadtxt(os.path.join(module_path, 'data', 'digits.csv.gz'),
                      delimiter=',')
    target = data[:, -1].astype("int32")
    flat_data = data[:, :-1].astype(floatX)
    return {"data": flat_data, "target": target}


def make_ocr(list_of_strings):
    """
    Create an optical character recognition (OCR) dataset from a list of strings

    n_steps : variable
    n_samples : len(list_of_strings)
    n_features : 8

    Returns
    -------
    summary : dict
        A dictionary containing dataset information

        summary["data"] : array, shape (n_steps, n_samples, 8)
           Array containing list_of_strings, converted to bitmap images

        summary["target"] : array, shape (n_samples, )
            Array of variable length arrays, containing character indices for
            strings in list_of_strings

        summary["train_indices"] : array, shape (n_samples, )
            Indices array of the same length as summary["data"]

        summary["vocabulary"] : string
           All possible character labels as one long string

        summary["vocabulary_size"] : int
           len(summary["vocabulary"])

        summary["target_names"] : list
           list_of_strings stored for ease-of-access

    Notes
    -----
    Much of these bitmap utils modified from Shawn Tan

    https://github.com/shawntan/theano-ctc/
    """
    def string_to_bitmap(string):
        return np.hstack(np.array(
            [bitmap[char_mapping[c]] for c in string])).T[:, ::-1]

    data = []
    target = []
    for n, s in enumerate(list_of_strings):
        X_n = string_to_bitmap(s)
        y_n = string_to_character_index(s)
        data.append(X_n)
        target.append(y_n)
    data = np.asarray(data).transpose(1, 0, 2)
    target = np.asarray(target)
    return {"data": data, "target": target,
            "train_indices": np.arange(len(list_of_strings)),
            "vocabulary": all_vocabulary_chars,
            "vocabulary_size": len(all_vocabulary_chars),
            "target_names": list_of_strings}


def check_fetch_iamondb():
    """ Check for IAMONDB data

        This dataset cannot be downloaded automatically!
    """
    partial_path = get_dataset_dir("iamondb")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
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


def fetch_iamondb():
    from lxml import etree
    partial_path = check_fetch_iamondb()
    pickle_path = os.path.join(partial_path, "iamondb_saved.pkl")
    if not os.path.exists(pickle_path):
        files_path = os.path.join(partial_path, "task1.tar.gz")

        with tarfile.open(files_path) as tf:
            train_file = [fname for fname in tf.getnames()
                          if "trainset" in fname][0]

            def _s(lines):
                return [l.strip().decode("utf-8") for l in lines]

            f = tf.extractfile(train_file)
            train_names = _s(f.readlines())

            valid_files = [fname for fname in tf.getnames()
                           if "testset" in fname]
            valid_names = []
            for v in valid_files:
                f = tf.extractfile(v)
                valid_names.extend(_s(f.readlines()))

        strokes_path = os.path.join(partial_path, "lineStrokes-all.tar.gz")
        ascii_path = os.path.join(partial_path, "ascii-all.tar.gz")
        lsf = tarfile.open(strokes_path)
        af = tarfile.open(ascii_path)
        sf = [fs for fs in lsf.getnames() if ".xml" in fs]

        def construct_ascii_path(f):
            primary_dir = f.split("-")[0]
            if f[-1].isalpha():
                sub_dir = f[:-1]
            else:
                sub_dir = f
            file_path = os.path.join("ascii", primary_dir, sub_dir, f + ".txt")
            return file_path

        def construct_stroke_paths(f):
            primary_dir = f.split("-")[0]
            if f[-1].isalpha():
                sub_dir = f[:-1]
            else:
                sub_dir = f
            files_path = os.path.join("lineStrokes", primary_dir, sub_dir)

            # Dash is crucial to obtain correct match!
            files = [sif for sif in sf if f in sif]
            files = sorted(files, key=lambda x: int(
                x.split(os.sep)[-1].split("-")[-1][:-4]))
            return files

        train_ascii_files = [construct_ascii_path(fta) for fta in train_names]
        valid_ascii_files = [construct_ascii_path(fva) for fva in valid_names]
        train_stroke_files = [construct_stroke_paths(fts)
                              for fts in train_names]
        valid_stroke_files = [construct_stroke_paths(fvs)
                              for fvs in valid_names]

        train_set_files = list(zip(train_stroke_files, train_ascii_files))
        valid_set_files = list(zip(valid_stroke_files, valid_ascii_files))

        dataset_storage = {}
        x_set = []
        y_set = []
        char_set = []
        for sn, se in enumerate([train_set_files, valid_set_files]):
            for n, (strokes_files, ascii_file) in enumerate(se):
                if n % 100 == 0:
                    print("Processing file %i of %i" % (n, len(se)))
                fp = af.extractfile(ascii_file)
                cleaned = [t.strip().decode("utf-8") for t in fp.readlines()
                           if t != '\r\n'
                           and t != ' \n'
                           and t != '\n'
                           and t != ' \r\n']

                # Try using CSR
                idx = [w for w, li in enumerate(cleaned) if li == "CSR:"][0]
                cleaned_sub = cleaned[idx + 1:]
                corrected_sub = []

                for li in cleaned_sub:
                    # Handle edge case with %%%%% meaning new line?
                    if "%" in li:
                        li2 = re.sub('\%\%+', '%', li).split("%")
                        li2 = ''.join([l.strip() for l in li2])
                        corrected_sub.append(li2)
                    else:
                        corrected_sub.append(li)
                corrected_sub = [c for c in corrected_sub if c != '']
                fp.close()

                n_one_hot = 57
                y = [np.zeros((len(li), n_one_hot), dtype='int16')
                     for li in corrected_sub]

                # A-Z, a-z, space, apostrophe, comma, period
                charset = list(range(65, 90 + 1)) + list(range(97, 122 + 1)) + [
                    32, 39, 44, 46]
                tmap = {k: w + 1 for w, k in enumerate(charset)}

                # 0 for UNK/other
                tmap[0] = 0

                def tokenize_ind(line):
                    t = [ord(c) if ord(c) in charset else 0 for c in line]
                    r = [tmap[i] for i in t]
                    return r

                for n, li in enumerate(corrected_sub):
                    y[n][np.arange(len(li)), tokenize_ind(li)] = 1

                x = []
                for stroke_file in strokes_files:
                    fp = lsf.extractfile(stroke_file)
                    tree = etree.parse(fp)
                    root = tree.getroot()
                    # Get all the values from the XML
                    # 0th index is stroke ID, will become up/down
                    s = np.array([[i, int(Point.attrib['x']),
                                   int(Point.attrib['y'])]
                                  for StrokeSet in root
                                  for i, Stroke in enumerate(StrokeSet)
                                  for Point in Stroke])

                    # flip y axis
                    s[:, 2] = -s[:, 2]

                    # Get end of stroke points
                    c = s[1:, 0] != s[:-1, 0]
                    ci = np.where(c == True)[0]
                    nci = np.where(c == False)[0]

                    # set pen down
                    s[0, 0] = 0
                    s[nci, 0] = 0

                    # set pen up
                    s[ci, 0] = 1
                    s[-1, 0] = 1
                    x.append(s)
                    fp.close()

                if len(x) != len(y):
                    x_t = np.vstack((x[-2], x[-1]))
                    x = x[:-2] + [x_t]

                if len(x) == len(y):
                    x_set.extend(x)
                    y_set.extend(y)
                    char_set.extend(corrected_sub)
                else:
                    print("Skipping %i, couldn't make x and y len match!" % n)
            if sn == 0:
                dataset_storage["train_indices"] = np.arange(len(x_set))
            elif sn == 1:
                offset = dataset_storage["train_indices"][-1] + 1
                dataset_storage["valid_indices"] = np.arange(offset, len(x_set))
                dataset_storage["data"] = np.array(x_set)
                dataset_storage["target"] = np.array(y_set)
                dataset_storage["target_phrases"] = char_set
                dataset_storage["vocabulary_size"] = n_one_hot
                c = "".join([chr(a) for a in [ord("-")] + charset])
                dataset_storage["vocabulary"] = c
            else:
                raise ValueError("Undefined number of files")
        f = open(pickle_path, "wb")
        pickle.dump(dataset_storage, f, -1)
        f.close()
    with open(pickle_path, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict


def music21_to_chord_duration(p):
    """
    Takes in a Music21 score, and outputs two lists
    List for chords (by string name)
    List for durations
    """
    p_chords = p.chordify()
    p_chords_o = p_chords.flat.getElementsByClass('Chord')
    chord_list = []
    duration_list = []
    for ch in p_chords_o:
        chord_list.append(ch.primeFormString)
        #chord_list.append(ch.pitchedCommonName)
        duration_list.append(ch.duration.quarterLength)
    return chord_list, duration_list


def music21_to_pitch_duration(p):
    """
    Takes in a Music21 score, and outputs two numpy arrays and a list
    One for pitch
    One for duration
    list for part times of each voice
    """
    parts = []
    parts_times = []
    for i, pi in enumerate(p.parts):
        part = []
        part_time = []
        for n in pi.stream().flat.notesAndRests:
            if n.isRest:
                part.append(0)
            else:
                try:
                    part.append(n.midi)
                except AttributeError:
                    continue
            part_time.append(n.duration.quarterLength)
        parts.append(part)
        parts_times.append(part_time)

    from IPython import embed; embed(); raise ValueError()
    cumulative_times = map(lambda x: np.cumsum(x), parts_times)
    # shift so all starts at 0
    cumulative_times = [cu - cu[0] for cu in cumulative_times]
    # find event times and align them
    parts = [np.array(part) for part in parts]
    parts_times = [np.array(part_time) for part_time in parts_times]
    total_set = set()
    for pi in range(len(parts)):
        total_set = total_set | set(list(cumulative_times[pi]))
    total_set = np.array(sorted(list(set(total_set))))
    # -1 for no-op?
    to_fill_pitch = np.zeros((len(p.parts), len(total_set))) - 1
    to_fill_dur = np.zeros((len(p.parts), len(total_set))) - 1
    for pi in range(len(p.parts)):
        for nts, ts in enumerate(total_set):
            match_ctime = np.where(cumulative_times[pi] == ts)[0]
            ppi = parts[pi][match_ctime]
            pti = parts_times[pi][match_ctime]
            if len(ppi) == 1:
                to_fill_pitch[pi, nts] = float(ppi)
                to_fill_dur[pi, nts] = float(pti)
            elif len(ppi) == 0:
                pass
            else:
                raise ValueError("Unexpected multi-match error")
    return to_fill_pitch, to_fill_dur, parts_times


# http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
# only works on Unix platforms though
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise ValueError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

# optional music21
try:
    from music21 import converter, interval, pitch, harmony, analysis, spanner, midi
except ImportError:
    pass

def _single_extract_music21(files, data_path, skip_chords, verbose, n):
    if verbose:
        logger.info("Starting file {} of {}".format(n, len(files)))
    f = files[n]
    file_path = os.path.join(data_path, f)

    start_time = time.time()

    try:
        p = converter.parse(file_path)
        k = p.analyze("key")
        parse_time = time.time()
        if verbose:
            r = parse_time - start_time
            logger.info("Parse time {}:{}".format(f, r))
    except (AttributeError, IndexError, UnicodeDecodeError,
            UnicodeEncodeError, harmony.ChordStepModificationException,
            ZeroDivisionError,
            ValueError,
            midi.MidiException,
            analysis.discrete.DiscreteAnalysisException,
            pitch.PitchException,
            spanner.SpannerException) as err:
        logger.info("Parse failed for {}".format(f))
        return ("null",)

    p.keySignature = k

    # none if there is no data aug
    an = "B" if "major" in k.name else "D"

    try:
        pc = pitch.Pitch(an)
        i = interval.Interval(k.tonic, pc)
        p = p.transpose(i)
        k = p.analyze("key")
        transpose_time = time.time()
        if verbose:
            r = transpose_time - start_time
            logger.info("Transpose time {}:{}".format(f, r))

        if skip_chords:
            chords = ["null"]
            chord_durations = ["null"]
        else:
            chords, chord_durations = music21_to_chord_duration(p)
        pitches, durations, part_times = music21_to_pitch_duration(p)
        pitch_duration_time = time.time()
        if verbose:
            r = pitch_duration_time - start_time
            logger.info("music21 to pitch_duration time {}:{}".format(f, r))
    except TypeError:
        #raise ValueError("Non-transpose not yet supported")
        return ("null",)
        """
        pc = pitch.Pitch(an)
        i = interval.Interval(k.tonic, pc)
        # FIXME: In this case chords are unnormed?
        if skip_chords:
            chords = ["null"]
            chord_durations = ["null"]
        else:
            chords, chord_durations = music21_to_chord_duration(p)
        pitches, durations = music21_to_pitch_duration(p)

        kt = k.tonic.pitchClass
        pct = pc.pitchClass
        assert kt >= 0
        if kt <= 6:
            pitches -= kt
        else:
            pitches -= 12
            pitches += (12 - kt)
        # now centered at C

        if "minor" in k.name:
            # C -> B -> B flat -> A
            pitches -= 3

        if pct <= 6:
            pitches += pct
        else:
            pitches -= 12
            pitches += pct
        """

    str_key = "{} minor".format(an) if "minor" in k.name else "{} major".format(an)

    ttime = time.time()
    if verbose:
        r = ttime - start_time
        logger.info("Overall file time {}:{}".format(f, r))
    return (pitches, durations, part_times, str_key, f, p.quarterLength, chords, chord_durations)


# http://stackoverflow.com/questions/29494001/how-can-i-abort-a-task-in-a-multiprocessing-pool-after-a-timeout
def abortable_worker(func, *args, **kwargs):
    # returns ("null",) if timeout
    timeout = kwargs.get('timeout', None)
    p = multiprocessing.dummy.Pool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        return ("null",)


def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


def _music_extract(data_path, pickle_path, ext=".xml",
                   pitch_augmentation=False,
                   skip_chords=True,
                   skip_drums=True,
                   lower_voice_limit=None,
                   upper_voice_limit=None,
                   equal_voice_count=4,
                   parse_timeout=100,
                   multiprocess_count=None,
                   verbose=False):

    if not os.path.exists(pickle_path):
        logger.info("Pickled file %s not found, creating. This may take a few minutes..." % pickle_path)
        itime = time.time()

        all_transposed_pitch = []
        all_transposed_duration = []
        all_transposed_keys = []
        all_part_times = []
        all_file_names = []
        all_transposed_chord = []
        all_transposed_chord_duration = []
        all_quarter_length = []

        if 'basestring' not in globals():
            basestring = str

        if isinstance(data_path, basestring):
            files = sorted([fi for fi in os.listdir(data_path) if fi.endswith(ext)])
        else:
            files = sorted([ap for ap in data_path if ap.endswith(ext)])

        #import pretty_midi
        logger.info("Processing {} files".format(len(files)))
        if multiprocess_count is not None:
            from multiprocessing import Pool
            import functools
            pool = Pool(4)
            ex = functools.partial(_single_extract_music21,
                                   files, data_path,
                                   skip_chords, verbose)
            abortable_ex = functools.partial(abortable_worker, ex, timeout=parse_timeout)
            result = pool.map(abortable_ex, range(len(files)))
            pool.close()
            pool.join()
        else:
            result = []
            for n in range(len(files)):
                r = _single_extract_music21(files, data_path, skip_chords,
                                            verbose, n)
                result.append(r)

        for n, r in enumerate(result):
            if r[0] != "null":
                (pitches, durations, part_times,
                key, fname, quarter_length,
                chords, chord_durations) = r

                all_transposed_chord.append(chords)
                all_transposed_chord_duration.append(chord_durations)
                all_transposed_pitch.append(pitches)
                all_transposed_duration.append(durations)
                all_part_times.append(part_times)
                all_transposed_keys.append(key)
                all_file_names.append(fname)
                all_quarter_length.append(quarter_length)
            else:
                logger.info("Result {} timed out".format(n))
        gtime = time.time()
        if verbose:
            r = gtime - itime
            logger.info("Overall time {}".format(r))
        d = {"data_pitch": all_transposed_pitch,
             "data_duration": all_transposed_duration,
             "data_part_times": all_part_times,
             "data_key": all_transposed_keys,
             "data_chord": all_transposed_chord,
             "data_chord_duration": all_transposed_chord_duration,
             "data_quarter_length": all_quarter_length,
             "file_names": all_file_names}
        with open(pickle_path, "wb") as f:
            logger.info("Saving pickle file %s" % pickle_path)
            pickle.dump(d, f)
        logger.info("Pickle file %s saved" % pickle_path)
    else:
        logger.info("Loading cached data from {}".format(pickle_path))
        with open(pickle_path, "rb") as f:
            d = pickle.load(f)

    from IPython import embed; embed(); raise ValueError()
    major_pitch = []
    minor_pitch = []

    major_duration = []
    minor_duration = []

    major_chord = []
    minor_chord = []

    major_chord_duration = []
    minor_chord_duration = []

    major_filename = []
    minor_filename = []

    major_quarter_length = []
    minor_quarter_length = []

    major_part_times = []
    minor_part_times = []

    keys = []
    for i in range(len(d["data_key"])):
        k = d["data_key"][i]
        ddp = d["data_pitch"][i]
        ddd = d["data_duration"][i]
        ddt = d["data_part_times"][i]
        nm = d["file_names"][i]
        ql = d["data_quarter_length"][i]
        try:
            ch = d["data_chord"][i]
            chd = d["data_chord_duration"][i]
        except IndexError:
            ch = "null"
            chd = -1

        if "major" in k:
            major_pitch.append(ddp)
            major_duration.append(ddd)
            major_part_times.append(ddt)
            major_filename.append(nm)
            major_chord.append(ch)
            major_chord_duration.append(chd)
            major_quarter_length.append(ql)
            keys.append(k)
        elif "minor" in k:
            minor_pitch.append(ddp)
            minor_duration.append(ddd)
            minor_part_times.append(ddt)
            minor_filename.append(nm)
            minor_chord.append(ch)
            minor_chord_duration.append(chd)
            minor_quarter_length.append(ql)
            keys.append(k)
        else:
            raise ValueError("Unknown key %s" % k)


    def replace_with_indices(arr, lu):
        "Inplace but return reference"
        all_idx = [np.where(arr.ravel() == u)[0] for u in sorted(lu.keys())]

        for u, idx in zip(sorted(lu.keys()), all_idx):
            if len(idx) > 0:
                arr.flat[idx] = lu[u]
        return arr, lu

    all_pitches = major_pitch + minor_pitch
    all_durations = major_duration + minor_duration
    all_part_times = major_part_times + minor_part_times
    all_filenames = major_filename + minor_filename
    all_chord = major_chord + minor_chord
    all_chord_duration = major_chord_duration + minor_chord_duration
    all_quarter_length = major_quarter_length + minor_quarter_length

    shps = [ap.shape for ap in all_pitches]
    shps0 = np.array([shpsi[0] for shpsi in shps])
    n_notes = np.unique(shps0)

    # find max
    max_count = -1
    max_note = -1
    for ni in n_notes:
        r = np.sum(0. * np.where(shps0 == ni)[0] + 1)
        if r > max_count:
            max_count = r
            max_note = ni

    final_chord_set = []
    final_chord_duration_set = []
    for n in range(len(all_chord)):
        final_chord_set.extend(all_chord[n])
        final_chord_duration_set.extend(all_chord_duration[n])

    final_chord_set = sorted(set(final_chord_set))
    final_chord_lookup = {k: v for k, v in zip(final_chord_set, range(len(final_chord_set)))}
    final_chord_duration_set = sorted(set(final_chord_duration_set))
    final_chord_duration_lookup = {k: v for k, v in zip(final_chord_duration_set, range(len(final_chord_duration_set)))}

    final_chord = []
    final_chord_duration = []
    for n in range(len(all_chord)):
        final_chord.append(np.array([final_chord_lookup[ch] for ch in all_chord[n]]).astype(floatX))
        final_chord_duration.append(np.array([final_chord_duration_lookup[chd] for chd in all_chord_duration[n]]).astype(floatX))

    final_pitches = []
    final_durations = []
    final_part_times = []
    final_filenames = []
    final_keys = []
    final_quarter_length = []

    invalid_idx = []
    for i in range(len(all_pitches)):
        n = all_pitches[i].shape[0]
        if lower_voice_limit is None and upper_voice_limit is None:
            cond = True
        else:
            raise ValueError("Voice limiting not yet implemented...")

        if cond:
            #if n == max_note:
            final_pitches.append(all_pitches[i])
            final_durations.append(all_durations[i])
            final_part_times.append(all_part_times[i])
            final_filenames.append(all_filenames[i])
            final_keys.append(keys[i])
            final_quarter_length.append(all_quarter_length[i])
        else:
            invalid_idx.append(i)
            if verbose:
                logger.info("Skipping file {}: {} had invalid note count != {}".format(
                    i, all_filenames[i], max_note))

    # drop and align
    final_chord = [fc for n, fc in enumerate(final_chord)
                   if n not in invalid_idx]
    final_chord_duration = [fcd for n, fcd in enumerate(final_chord_duration)
                            if n not in invalid_idx]

    final_chord = [fc[:final_pitches[n].shape[1]]
                   for n, fc in enumerate(final_chord)]
    final_chord_duration = [fcd[:final_pitches[n].shape[1]]
                            for n, fcd in enumerate(final_chord_duration)]

    all_chord = final_chord
    all_chord_duration = final_chord_duration

    all_pitches = final_pitches
    all_durations = final_durations
    all_part_times = final_part_times
    all_filenames = final_filenames
    all_keys = final_keys
    all_quarter_length = final_quarter_length

    pitch_list = list(np.unique(np.concatenate([np.unique(api) for api in all_pitches])))
    duration_list = list(np.unique(np.concatenate([np.unique(adi) for adi in all_durations])))

    basic_durs = [.25, .33, .5, .66, 1., 1.5, 2., 2.5, 3, 3.5, 4., 5., 6., 8.]
    if len(duration_list) > len(basic_durs):
        from scipy.cluster.vq import kmeans2, vq

        #cent, lbl = kmeans2(np.array(duration_list), 200)

        # relative to quarter length

        ul = np.percentile(duration_list, 90)
        duration_list = [dl if dl < ul else ul for dl in duration_list]
        counts, tt = np.histogram(duration_list, 30)
        cent = tt[:-1] + (tt[1:] - tt[:-1]) * .5
        cent = cent[cent > basic_durs[-1]]
        cent = sorted(basic_durs + list(cent))

        all_durations_new = []
        for adi in all_durations:
            shp = adi.shape
            fixed = vq(adi.flatten(), cent)[0]
            fixed = fixed.astype(floatX)

            code_where = []
            for n, ci in enumerate(cent):
                code_where.append(np.where(fixed == n))

            for n, cw in enumerate(code_where):
                fixed[cw] = cent[n]

            fixed = fixed.reshape(shp)
            all_durations_new.append(fixed)
        all_durations = all_durations_new
        duration_list = list(np.unique(np.concatenate([np.unique(adi) for adi in all_durations])))

    pitch_lu = {k: v for v, k  in enumerate(pitch_list)}
    duration_lu = {k: v for v, k in enumerate(duration_list)}

    ldp = [replace_with_indices(dpi.T, pitch_lu)[0][:, ::-1]
           for dpi in all_pitches]
    ldd = [replace_with_indices(ddi.T, duration_lu)[0][:, ::-1]
           for ddi in all_durations]

    ldt = []

    # align part times into a matrix
    for vi in range(len(ldd)):
        # get voice count
        shp = ldd[vi].shape
        ldti = np.zeros_like(ldd[vi])

        # part times are backwards ordered from ldd
        part_time = all_part_times[vi][::-1]
        for i in range(shp[1]):
            nonzero_idx = ldd[vi][:, i] != 0
            assert nonzero_idx.sum() == len(part_time[i])
            ldti[nonzero_idx, i] = part_time[i]
        cs = np.cumsum(ldti, axis=0)
        ldt.append(cs)

    force_voices = equal_voice_count
    def _trunc(a, b, c):
        assert a.shape == b.shape == c.shape
        # take most active voices?
        actives = np.where(a > 0)[1]
        xx, zz = count_unique(actives)
        idx = np.sort(np.argsort(zz)[:force_voices])
        return a[:, idx], b[:, idx], c[:, idx]
        #return a[:, -force_voices:], b[:, -force_voices:]
        #return a[:, :force_voices], b[:, :force_voices]

    def _ext(a, b, c):
        assert a.shape == b.shape == c.shape
        # if extending, fill in SBAT order
        new_a = np.concatenate((a, np.zeros((a.shape[0], force_voices - a.shape[1])).astype(a.dtype)), axis=1)
        new_b = np.concatenate((b, np.zeros((b.shape[0], force_voices - b.shape[1])).astype(b.dtype)), axis=1)
        new_c = np.concatenate((c, np.zeros((c.shape[0], force_voices - c.shape[1])).astype(c.dtype)), axis=1)
        return new_a, new_b, new_c

    new_ldp = []
    new_ldd = []
    new_ldt = []
    for ii in range(len(ldp)):
        if ldp[ii].shape[1] == force_voices:
            pp = ldp[ii]
            dd = ldd[ii]
            tt = ldt[ii]
        elif ldp[ii].shape[1] < force_voices:
            pp, dd, tt = _ext(ldp[ii], ldd[ii], ldt[ii])
        elif ldp[ii].shape[1] > force_voices:
            pp, dd, tt = _trunc(ldp[ii], ldd[ii], ldt[ii])
        new_ldp.append(pp)
        new_ldd.append(dd)
        new_ldt.append(tt)
    ldp = new_ldp
    ldd = new_ldd
    ldt = new_ldt

    quarter_length_list = sorted([float(ql) for ql in list(set(all_quarter_length))])
    all_quarter_length = [float(ql) for ql in all_quarter_length]

    time_list = sorted(list(np.unique(np.concatenate([np.unique(ldti.ravel()) for ldti in ldt]))))

    r = {"list_of_data_pitch": ldp,
         "list_of_data_duration": ldd,
         "list_of_data_time": ldt,
         "list_of_data_key": all_keys,
         "list_of_data_chord": all_chord,
         "list_of_data_chord_duration": all_chord_duration,
         "list_of_data_quarter_length": all_quarter_length,
         "chord_list": final_chord_set,
         "chord_duration_list": final_chord_duration_set,
         "pitch_list": pitch_list,
         "duration_list": duration_list,
         "time_list": time_list,
         "quarter_length_list": quarter_length_list,
         "filename_list": all_filenames}
    return r


def check_fetch_symbtr_music21():
    """ Check for symbtr data """
    # https://github.com/kastnerkyle/SymbTr
    from music21 import corpus
    partial_path = get_dataset_dir("symbtr")
    full_path = os.path.join(partial_path, "symbtr.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        raise ValueError("DOWNLOAD NOT YET SETUP")
        download(url, full_path, progress_update_percentage=1)
        zipf = zipfile.ZipFile(full_path, 'r')
        zipf.extractall(partial_path)
        zipf.close()
    return partial_path


def fetch_symbtr_music21(keys=["C major", "A minor"], verbose=False):
    """
    SymbTr music, transposed to C major or A minor (depending on original key).
    Only contains chorales with 4 voices populated.
    Requires music21.

    '''
    n_timesteps : 34270
    n_features : 1
    n_classes : 12 (duration), 54 (pitch)
    '''

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data_pitch"] : array, shape (34270, 1)
            All pieces' pitches concatenated as an array
        summary["data_duration"] : array, shape (34270, 1)
            All pieces' durations concatenated as an array
        summary["list_of_data_pitch"] : list of array
            Pitches for each piece
        summary["list_of_data_duration"] : list of array
            Durations for each piece
        summary["list_of_data_key"] : list of str
            String key for each piece
        summary["pitch_list"] : list, len 54
        summary["duration_list"] : list, len 12
        summary["major_minor_split"] : int, 16963
            Index into data_pitch or data_duration to split for major and minor

    Can split the data to only have major or minor key songs.
    For major, summary["data_pitch"][:summary["major_minor_split"]]
    For minor, summary["data_pitch"][summary["major_minor_split"]:]
    The same operation works for duration.

    pitch_list and duration_list give the mapping back from array value to
    actual data value.
    """

    data_path = check_fetch_symbtr_music21()
    pickle_path = os.path.join(data_path, "__processed_symbtr.pkl")
    return _musicxml_extract(data_path, pickle_path, verbose=False)


def check_fetch_bach_chorales_music21():
    """ Move files into pthbldr dir, in case python is on nfs. """
    from music21 import corpus
    all_bach_paths = corpus.getComposer("bach")
    partial_path = get_dataset_dir("bach_chorales_music21")
    for path in all_bach_paths:
        if "riemenschneider" in path:
            continue
        filename = os.path.split(path)[-1]
        local_path = os.path.join(partial_path, filename)
        if not os.path.exists(local_path):
            shutil.copy2(path, local_path)
    return partial_path


def fetch_bach_chorales_music21(keys=["B major", "D minor"],
                                truncate_length=100,
                                compress_pitch=False,
                                compress_duration=False,
                                verbose=True):
    """
    Bach chorales, transposed to C major or A minor (depending on original key).
    Only contains chorales with 4 voices populated.
    Requires music21.

    n_timesteps : 34270
    n_features : 4
    n_classes : 12 (duration), 54 (pitch)

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["list_of_data_pitch"] : list of array
            Pitches for each piece
        summary["list_of_data_duration"] : list of array
            Durations for each piece
        summary["list_of_data_key"] : list of str
            String key for each piece
        summary["list_of_data_chord"] : list of str
            String chords for each piece
        summary["list_of_data_chord_duration"] : list of str
            String chords for each piece
        summary["pitch_list"] : list
        summary["duration_list"] : list

    pitch_list and duration_list give the mapping back from array value to
    actual data value.
    """

    data_path = check_fetch_bach_chorales_music21()
    pickle_path = os.path.join(data_path, "__processed_bach.pkl")
    mu = _music_extract(data_path, pickle_path, ext=".mxl",
                        skip_chords=False, equal_voice_count=4,
                        verbose=verbose)

    lp = mu["list_of_data_pitch"]
    ld = mu["list_of_data_duration"]
    lt = mu["list_of_data_time"]
    lch = mu["list_of_data_chord"]
    lchd = mu["list_of_data_chord_duration"]
    lql = mu["list_of_data_quarter_length"]

    def _len_prune(l):
        return [li[:truncate_length] for li in l]

    lp2 = _len_prune(lp)
    ld2 = _len_prune(ld)
    lt2 = _len_prune(lt)
    lch2 = _len_prune(lch)
    lchd2 = _len_prune(lchd)

    def _key_prune(l):
        k = mu["list_of_data_key"]
        assert len(l) == len(k)
        return [li for li, ki in zip(l, k) if ki in keys]

    lp2 = _key_prune(lp2)
    ld2 = _key_prune(ld2)
    lt2 = _key_prune(lt2)
    lch2 = _key_prune(lch2)
    lchd2 = _key_prune(lchd2)
    lql2 = _key_prune(lql)

    lp = lp2
    ld = ld2
    lt = lt2
    lch = lch2
    lchd = lchd2
    lql = lql2

    def _fixup(mb, lookup_list):
        mb = mb.copy()
        where = []
        for n, li in enumerate(lookup_list):
            where.append(np.where(mb == n))

        for n, w in enumerate(where):
            mb[w] = lookup_list[n]
        return mb

    if not compress_pitch:
        pl = mu["pitch_list"]
        lp = [_fixup(lpi, pl) for lpi in lp]

    if not compress_duration:
        dl = mu["duration_list"]
        ld = [_fixup(ldi, dl) for ldi in ld]

    mu["list_of_data_pitch"] = lp
    mu["list_of_data_duration"] = ld
    mu["list_of_data_time"] = lt
    mu["list_of_data_chord"] = lch
    mu["list_of_data_chord_duration"] = lchd
    mu["list_of_data_quarter_length"] = lql
    return mu


def check_fetch_wikifonia_music21():
    # http://www.synthzone.com/forum/ubbthreads.php/topics/384909/Download_for_Wikifonia_all_6_6
    partial_path = get_dataset_dir("wikifonia")
    full_path = os.path.join(partial_path, "wikifonia.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        raise ValueError("DOWNLOAD NOT YET SETUP")
        download(url, full_path, progress_update_percentage=1)
        zipf = zipfile.ZipFile(full_path, 'r')
        zipf.extractall(partial_path)
        zipf.close()
    final_path = os.path.join(partial_path, "Wikifonia")
    return final_path


def fetch_wikifonia_music21():
    """
    Bach chorales, transposed to C major or C minor (depending on original key).
    Only contains chorales with 4 voices populated.
    Requires music21.

    n_timesteps : 34270
    n_features : 4
    n_classes : 12 (duration), 54 (pitch)

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data_pitch"] : array, shape (34270, 4)
            All pieces' pitches concatenated as an array
        summary["data_duration"] : array, shape (34270, 4)
            All pieces' durations concatenated as an array
        summary["list_of_data_pitch"] : list of array
            Pitches for each piece
        summary["list_of_data_duration"] : list of array
            Durations for each piece
        summary["list_of_data_key"] : list of str
            String key for each piece
        summary["pitch_list"] : list, len 54
        summary["duration_list"] : list, len 12
        summary["major_minor_split"] : int, 16963
            Index into data_pitch or data_duration to split for major and minor

    Can split the data to only have major or minor key songs.
    For major, summary["data_pitch"][:summary["major_minor_split"]]
    For minor, summary["data_pitch"][summary["major_minor_split"]:]
    The same operation works for duration.

    pitch_list and duration_list give the mapping back from array value to
    actual data value.
    """

    data_path = check_fetch_wikifonia_music21()
    pickle_path = os.path.join(data_path, "__processed_wikifonia.pkl")
    return _musicxml_extract(data_path, pickle_path, mxl_ext=".mxl")


def check_fetch_lakh_midi_music21():
    """ Check for lakh midi data """
    partial_path = get_dataset_dir("lakh_midi")
    full_path = os.path.join(partial_path, "lmd_full.tar.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        # https://archive.org/details/lakh-midi-dataset-v0_1
        raise ValueError("DOWNLOAD NOT YET SETUP")
        download(url, full_path, progress_update_percentage=1)

    if not os.path.exists(partial_path + os.sep + "lmd_full"):
        raise ValueError("Need to untar with `tar xzf lmd_full.tar.gz`")

    cache_name = "_cached_paths.pkl"
    if not os.path.exists(partial_path + os.sep + cache_name):
        all_paths = []
        for dirpath, dirnames, filenames in os.walk(partial_path):
            for filename in [f for f in filenames if f.endswith(".mid")]:
                all_paths.append(os.path.join(dirpath, filename))

        all_paths = sorted(all_paths)
        f = open(partial_path + os.sep + cache_name, "wb")
        pickle.dump(all_paths, f)
    else:
        with open(partial_path + os.sep + cache_name, "rb") as f:
            all_paths = pickle.load(f)

    return partial_path, all_paths


def fetch_lakh_midi_music21(keys=["C major", "A minor"],
                            subset=None):
    """
    supported subsets
    """
    data_path, all_paths = check_fetch_lakh_midi_music21()
    with open(data_path + os.sep + "md5_to_paths.json") as data_file:
           paths_lookup = json.load(data_file)

    all_info = []
    for idx in range(len(all_paths)):
        md5_id = all_paths[idx].split(os.sep)[-1].split(".")[0]
        tru = paths_lookup[md5_id]
        full_info = ["idx {}".format(idx) + ": " + trui for trui in tru]
        all_info.extend(full_info)

    search_term = None
    if subset == "chopin":
        search_term = "chopin"
        subset_idx = [int(ai.split(" ")[1].split(":")[0])
                     for ai in all_info
                     if search_term in ai.lower()]
    elif subset == "pop":
        all_idx = []
        for st in ["jackson", "beatles", "backstreet", "madonna", "beach_boys",
                   "beach boys", "britney"]:
            subset_idx = [int(ai.split(" ")[1].split(":")[0])
                          for ai in all_info if st in ai.lower()]
            all_idx.extend(subset_idx)
    else:
        raise ValueError("Invalid subset {}".format(subset))

    subset_idx = sorted(list(set(subset_idx)))

    all_paths = [ap for n, ap in enumerate(all_paths) if n in subset_idx]
    pickle_path = os.path.join(data_path, "__processed_lakh_{}.pkl".format(subset))
    return _music_extract(all_paths, pickle_path, ext=".mid",
                          skip_chords=True, verbose=True, equal_voice_count=4)


def check_fetch_haralick_midi_music21():
    """ Check for haralick data """
    from music21 import corpus
    partial_path = get_dataset_dir("haralick")
    full_path = os.path.join(partial_path, "k_collection.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        # http://www.haralick.org/ML/k_collection.zip
        raise ValueError("DOWNLOAD NOT YET SETUP")
        download(url, full_path, progress_update_percentage=1)

    if not os.path.exists(partial_path + os.sep + "vivaldi"):
        zipf = zipfile.ZipFile(full_path, 'r')
        zipf.extractall(partial_path)
        zipf.close()

        # fix poor zipping
        poor_zip = [partial_path + os.sep + base for base in os.listdir(partial_path)
                    if base.endswith(".mid")]
        movedir = partial_path + os.sep + "assorted"
        os.mkdir(movedir)
        for pz in poor_zip:
            shutil.move(pz, movedir + os.sep + pz.split(os.sep)[-1])

    all_paths = []
    for dirpath, dirnames, filenames in os.walk(partial_path):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            all_paths.append(os.path.join(dirpath, filename))

    all_paths = sorted(all_paths)
    return partial_path, all_paths


def fetch_haralick_midi_music21(keys=["C major", "A minor"],
                                subset=None):
    """
    supported subsets

    mozart_piano
    """
    data_path, all_paths = check_fetch_haralick_midi_music21()
    pickle_path = os.path.join(data_path, "__processed_haralick_{}.pkl".format(subset))
    if subset is not None:
        if subset == "mozart_piano":
            all_paths = [ap for ap in all_paths
                         if "mozart" in ap
                         and "piano" in ap
                         and "!piano-rolls" not in ap
                         and "mozart_piano" in ap]
        else:
            raise ValueError("Subset {} currently not supported".format(subset))
    return _music_extract(all_paths, pickle_path, ext=".mid",
                          skip_chords=True, verbose=True)


def quantized_to_pretty_midi(quantized,
                             quantized_bin_size,
                             save_dir="samples",
                             name_tag="sample_{}.mid",
                             add_to_name=0,
                             lower_pitch_limit=12,
                             list_of_quarter_length=None,
                             default_quarter_length=47,
                             voice_params="woodwinds"):
    is_seq_of_seq = False
    try:
        quantized[0][0]
        if not hasattr(quantized, "flatten"):
            is_seq_of_seq = True
    except:
        try:
            quantized.shape
        except AttributeError:
            raise ValueError("quantized must be a sequence of sequence (such as list of array, or list of list) or numpy array")
    # list of list or mb?
    pitches = []
    durations = []
    if is_seq_of_seq:
        sz = len(quantized)
        for i in range(sz):
            q = quantized[i]
            pitch_i = []
            dur_i = []
            cur = None
            count = 0
            for qi in q:
                if qi != cur:
                    pitch_i.append(qi)
                    quarter_count = quantized_bin_size * (count + 1)
                    dur_i.append(quarter_count)
                    cur = qi
                    count = 0
                else:
                    count += 1
            pitches.append(pitch_i)
            durations.append(dur_i)
    else:
        sz = quantized.shape[1]
        raise ValueError("NYI")
    pitches_and_durations_to_pretty_midi(pitches, durations,
                                             save_dir=save_dir,
                                             name_tag=name_tag,
                                             add_to_name=add_to_name,
                                             lower_pitch_limit=lower_pitch_limit,
                                             list_of_quarter_length=list_of_quarter_length,
                                             default_quarter_length=default_quarter_length,
                                             voice_params=voice_params)


def pitches_and_durations_to_pretty_midi(pitches, durations,
                                         save_dir="samples",
                                         name_tag="sample_{}.mid",
                                         add_to_name=0,
                                         lower_pitch_limit=12,
                                         list_of_quarter_length=None,
                                         default_quarter_length=47,
                                         voice_params="woodwinds"):
    # allow list of list
    """
    durations assumed to be scaled to quarter lengths e.g. 1 is 1 quarter note
    2 is a half note, etc
    """
    is_seq_of_seq = False
    try:
        pitches[0][0]
        durations[0][0]
        if not hasattr(pitches, "flatten") and not hasattr(durations, "flatten"):
            is_seq_of_seq = True
    except:
        try:
            pitches.shape
            durations.shape
        except AttributeError:
            raise ValueError("pitches and durations must be a sequence of sequence (such as list of array, or list of list) or numpy array")

    import pretty_midi
    # BTAS mapping
    def weird():
        voice_mappings = ["Sitar", "Orchestral Harp", "Acoustic Guitar (nylon)",
                          "Pan Flute"]
        voice_velocity = [20, 80, 80, 40]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., .95]
        return voice_mappings, voice_velocity, voice_offset, voice_decay

    if voice_params == "weird":
        voice_mappings, voice_velocity, voice_offset, voice_decay = weird()
    elif voice_params == "weird_r":
        voice_mappings, voice_velocity, voice_offset, voice_decay = weird()
        voice_mappings = voice_mappings[::-1]
        voice_velocity = voice_velocity[::-1]
        voice_offset = voice_offset[::-1]
        voice_decay = voice_decay[::-1]
    elif voice_params == "legend":
        # LoZ
        voice_mappings = ["Acoustic Guitar (nylon)"] * 3 + ["Pan Flute"]
        voice_velocity = [20, 16, 25, 20]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., .95]
    elif voice_params == "organ":
        voice_mappings = ["Church Organ"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [.98, .98, .98, .98]
    elif voice_params == "piano":
        voice_mappings = ["Acoustic Grand Piano"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    elif voice_params == "electric_piano":
        voice_mappings = ["Electric Piano 1"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    elif voice_params == "harpsichord":
        voice_mappings = ["Harpsichord"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    elif voice_params == "woodwinds":
        voice_mappings = ["Bassoon", "Clarinet", "English Horn", "Oboe"]
        voice_velocity = [50, 30, 30, 40]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    else:
        # eventually add and define dictionary support here
        raise ValueError("Unknown voice mapping specified")

    # normalize
    mm = float(max(voice_velocity))
    mi = float(min(voice_velocity))
    dynamic_range = min(80, (mm - mi))
    # keep same scale just make it louder?
    voice_velocity = [int((80 - dynamic_range) + int(v - mi)) for v in voice_velocity]

    if not is_seq_of_seq:
        order = durations.shape[-1]
    else:
        try:
            order = durations[0].shape[-1]
        except:
            order = 1
            pitches = [np.array(p)[:, None] for p in pitches]
            durations = [np.array(d)[:, None] for d in durations]
    voice_mappings = voice_mappings[-order:]
    voice_velocity = voice_velocity[-order:]
    voice_offset = voice_offset[-order:]
    voice_decay = voice_decay[-order:]
    if not is_seq_of_seq:
        pitches = [pitches[:, i, :] for i in range(pitches.shape[1])]
        durations = [durations[:, i, :] for i in range(durations.shape[1])]

    n_samples = len(durations)
    for ss in range(n_samples):
        durations_ss = durations[ss]
        pitches_ss = pitches[ss]
        assert len(durations_ss) == len(pitches_ss)
        pm_obj = pretty_midi.PrettyMIDI()
        # Create an Instrument instance for a cello instrument
        def mkpm(name):
            return pretty_midi.instrument_name_to_program(name)

        def mki(p):
            return pretty_midi.Instrument(program=p)

        pm_programs = [mkpm(n) for n in voice_mappings]
        pm_instruments = [mki(p) for p in pm_programs]

        if list_of_quarter_length is None:
            # qpm to s per quarter = 60 s per min / quarters per min
            time_scale = 60. / default_quarter_length
        else:
            time_scale = 60. / list_of_quarter_length[ss]

        time_offset = np.zeros((order,))
        for ii in range(len(durations_ss)):
            for jj in range(order):
                pitches_isj = pitches_ss[ii, jj]
                durations_isj = durations_ss[ii, jj]
                p = int(pitches_isj)
                d = durations_isj
                if d < 0:
                    continue
                if p < 0:
                    continue
                # hack out the whole last octave?
                s = time_scale * time_offset[jj]
                e = time_scale * (time_offset[jj] + voice_decay[jj] * d)
                time_offset[jj] += d
                if p < lower_pitch_limit:
                    continue
                note = pretty_midi.Note(velocity=voice_velocity[jj],
                                        pitch=p + voice_offset[jj],
                                        start=s, end=e)
                # Add it to our instrument
                pm_instruments[jj].notes.append(note)
        # Add the instrument to the PrettyMIDI object
        for pm_instrument in pm_instruments:
            pm_obj.instruments.append(pm_instrument)
        # Write out the MIDI data

        sv = save_dir + os.sep + name_tag.format(ss + add_to_name)
        try:
            pm_obj.write(sv)
        except ValueError:
            logger.info("Unable to write file {} due to mido error".format(sv))


def check_fetch_midi_template():
    partial_path = get_dataset_dir("midi_template")
    full_path = os.path.join(partial_path, "midi_template.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        url = "https://www.dropbox.com/s/w23hlsw778kyeno/midi_template.zip?dl=1"
        download(url, full_path, progress_update_percentage=1)
    return full_path


def dump_midi_player_template(save_dir="samples"):
    full_path = check_fetch_midi_template()
    zipf = zipfile.ZipFile(full_path, 'r')
    zipf.extractall(save_dir)
    zipf.close()
