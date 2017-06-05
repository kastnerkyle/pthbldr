# Authors: Kyle Kastner
from __future__ import print_function
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
try:
    import Queue
except ImportError:
    import queue as Queue
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib
try:
    import cPickle as pickle
except ImportError:
    import pickle

import json
import hashlib
import readline
import uuid
readline.parse_and_bind('tab: complete')
readline.parse_and_bind('set editing-mode vi')

def _dumps(arg):
   # paranoia
   # http://bugs.python.org/issue770997
   return pickle.dumps(arg, 1)


def get_name():
    base = str(uuid.uuid4())
    return base


# decided at import, should be consistent over training
checkpoint_uuid = get_name()[:6]
def get_checkpoint_uuid():
    return checkpoint_uuid


import copy
import threading
import logging
import uuid
from collections import OrderedDict
import socket
import random
import os
import glob
import subprocess
import numpy as np
from itertools import cycle
import __main__ as main
import re
import shutil
import numbers
import sys
import warnings
import inspect
import zipfile
import time
import pprint
from collections import defaultdict
from functools import reduce

sys.setrecursionlimit(40000)

logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger(__name__)

string_f = StringIO()
ch = logging.StreamHandler(string_f)
# Automatically put the HTML break characters on there
formatter = logging.Formatter('%(message)s<br>')
ch.setFormatter(formatter)
logger.addHandler(ch)

USER = os.getenv('USER')

def get_logger():
    """
    Fetch the global logger.
    """
    return logger


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass  # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=False, verbose=True):
    """
    Print and execute command on system
    """
    all_lines = []
    for line in execute(cmd, shell=shell):
        if verbose:
            print(line, end="")
        all_lines.append(line.strip())
    return all_lines


FINALIZE_TRAINING = False


def _get_finalize_train():
    return FINALIZE_TRAINING


def _set_finalize_train():
    global FINALIZE_TRAINING
    FINALIZE_TRAINING = True


# universal time
tt = str(time.time()).split(".")[0]
def get_time_string():
    return tt


checkpoint_import_time = time.strftime("%H-%M-%S_%Y-%d-%m", time.gmtime())
def get_checkpoint_import_time():
    return checkpoint_import_time


def set_checkpoint_import_time(time_str):
    logger.info("Setting global dagbldr import time to %s" % time_str)
    global checkpoint_import_time
    checkpoint_import_time = time_str


def get_script():
    py_file = None
    for argv in sys.argv[::-1]:
        if argv[-3:] == ".py":
            py_file = argv
        # slurm_script
        elif "slurm_" in argv:
            py_file = argv
    if "slurm" in py_file:
        script_name = os.environ['SLURM_JOB_NAME']
        script_name = script_name.split(".")[0]
    else:
        assert py_file is not None
        script_path = os.path.abspath(py_file)
        script_name = script_path.split(os.path.sep)[-1].split(".")[0]
        # gotta play games for slurm runner
    return script_name

_type = "float32"


def get_type():
    return _type


# TODO: Fetch from env
NUM_SAVED_TO_KEEP = 2


# copied from utils to avoid circular deps
def safe_zip(*args):
    """Like zip, but ensures arguments are of same length.

       Borrowed from pylearn2 - copied from utils to avoid circular import
    """
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)


def _special_check(verbose=True):
    ip_addr = socket.gethostbyname(socket.gethostname())
    subnet = ".".join(ip_addr.split(".")[:-1])
    whitelist = ["132.204.24", "132.204.25", "132.204.26", "132.204.27"]
    subnet_match = [subnet == w for w in whitelist]
    hostname = socket.gethostname()
    if any(subnet_match):
        if verbose:
            logger.info("Found special runtime environment!")
            logger.info("IP address: %s" % ip_addr)
            logger.info("Hostname: %s" % hostname)
        return True
    else:
        return False


def in_nosetest():
    return sys.argv[0].endswith('nosetests')


def write_results_as_html(results_dict, save_path, default_show="all"):
    as_html = filled_js_template_from_results_dict(
        results_dict, default_show=default_show)
    with open(save_path, "w") as f:
        f.writelines(as_html)


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def monitor_status_func(results_dict, append_name=None,
                        status_type="checkpoint",
                        nan_check=True, print_output=True):
    """ Dump the last results from a results dictionary """
    n_seen = max([len(l) for l in results_dict.values()])
    last_results = {k: v[-1] for k, v in results_dict.items()}
    # This really, really assumes a 1D numpy array (1,) or (1, 1)
    last_results = {k: float("%.15f" % v.ravel()[-1])
                    if isinstance(v, (np.generic, np.ndarray))
                    else float("%.15f" % v)
                    for k, v in last_results.items()}
    pp = pprint.PrettyPrinter()
    filename = main.__file__
    fileline = "Script %s" % str(filename)
    if status_type == "checkpoint":
        statusline = "Checkpoint %i" % n_seen
    else:
        raise ValueError("Unknown status_type %s" % status_type)
    breakline = "".join(["-"] * (len(statusline) + 1))
    if print_output:
        logger.info(breakline)
        logger.info(fileline)
        logger.info(statusline)
        logger.info(breakline)
        logger.info(pp.pformat(last_results))
    if status_type == "checkpoint":
        save_path = os.path.join(get_checkpoint_dir(),
                                 "model_checkpoint_%i.html" % n_seen)

    if append_name is not None:
        split = save_path.split("_")
        save_path = "_".join(
            split[:-1] + [append_name] + split[-1:])
    if not in_nosetest():
        # Don't dump if testing!
        # Only enable user defined keys
        nan_test = [(k, True) for k, r_v in results_dict.items()
                    for v in r_v if np.isnan(v)]
        if nan_check and len(nan_test) > 0:
            nan_keys = set([tup[0] for tup in nan_test])
            raise ValueError("Found NaN values in the following keys ",
                             "%s, exiting training" % nan_keys)
        show_keys = [k for k in results_dict.keys()
                     if "_auto" not in k]
        write_results_as_html(results_dict, save_path,
                              default_show=show_keys)
        if status_type == "checkpoint":
            cleanup_monitors("checkpoint", append_name)


def default_status_func(status_number, epoch_number, epoch_results):
    """ Default status function for iterate_function. Prints epoch info.

    This is exactly equivalent to defining your own status_function as such:
        def status_func(status_number, epoch_number, epoch_results):
            print_status_func(epoch_results)

    Parameters
    ----------
    status_number

    epoch_number

    epoch_results

    """
    monitor_status_func(epoch_results)


def init_results_dict():
    results = defaultdict(list)
    results["total_number_of_epochs_auto"] = [0]
    return results


def plot_training_epochs(epochs_dict, plot_name, plot_limit=None,
                         turn_on_agg=True):
    # plot_limit can be a positive integer, negative integer, or float in 0 - 1
    # float between 0 and 1 assumed to be percentage of total to keep
    if turn_on_agg:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # colors from seaborn flatui
    color_list = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e",
                  "#2ecc71"]
    colors = cycle(color_list)
    for key in epochs_dict.keys():
        if plot_limit < 1 and plot_limit > 0:
            plot_limit = int(plot_limit * len(epochs_dict[key]))
        plt.plot(epochs_dict[key][:plot_limit], color=colors.next())
        plt.title(str(key))
        plt.savefig(plot_name + "_" + str(key) + ".png")
        plt.close()


def plot_images_as_subplots(list_of_plot_args, plot_name, width, height,
                            invert_y=False, invert_x=False,
                            figsize=None, turn_on_agg=True):
    if turn_on_agg:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lengths = [len(a) for a in list_of_plot_args]
    if len(list(filter(lambda x: x != lengths[0], lengths))) > 0:
        raise ValueError("list_of_plot_args has elements of different lengths!")

    if figsize is None:
        f, axarr = plt.subplots(lengths[0], len(lengths))
    else:
        f, axarr = plt.subplots(lengths[0], len(lengths), figsize=figsize)
    for n, v in enumerate(list_of_plot_args):
        for i, X_i in enumerate(v):
            axarr[i, n].matshow(X_i.reshape(width, height), cmap="gray",
                                interpolation="none")
            axarr[i, n].axis('off')
            if invert_y:
                axarr[i, n].set_ylim(axarr[i, n].get_ylim()[::-1])
            if invert_x:
                axarr[i, n].set_xlim(axarr[i, n].get_xlim()[::-1])
    plt.tight_layout()
    plt.savefig(plot_name + ".png")


def make_gif(arr, gif_name, plot_width, plot_height, resize_scale_width=5,
             resize_scale_height=5, list_text_per_frame=None, invert_y=False,
             invert_x=False, list_text_per_frame_color=None, delay=1,
             grayscale=False, loop=False, turn_on_agg=True):
    """ Make a gif from a series of pngs using matplotlib matshow """
    if turn_on_agg:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot temporaries for making gif
    # use random code to try and avoid deleting surprise files...
    random_code = random.randrange(2 ** 32)
    pre = str(random_code)
    for n, arr_i in enumerate(arr):
        plt.matshow(arr_i.reshape(plot_width, plot_height), cmap="gray",
                    interpolation="none")
        if invert_y:
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
        if invert_x:
            ax = plt.gca()
            ax.set_xlim(ax.get_xlim()[::-1])

        plt.axis('off')
        if list_text_per_frame is not None:
            text = list_text_per_frame[n]
            if list_text_per_frame_color is not None:
                color = list_text_per_frame_color[n]
            else:
                color = "white"
            plt.text(0, plot_height, text, color=color,
                     fontsize=2 * plot_height)
        # This looks rediculous but should count the number of digit places
        # also protects against multiple runs
        # plus 1 is to maintain proper ordering
        plotpath = '__%s_giftmp_%s.png' % (str(n).zfill(len(
            str(len(arr))) + 1), pre)
        plt.savefig(plotpath)
        plt.close()

    # make gif
    assert delay >= 1
    gif_delay = int(delay)
    basestr = "convert __*giftmp_%s.png -delay %s " % (pre, str(gif_delay))
    if loop:
        basestr += "-loop 1 "
    else:
        basestr += "-loop 0 "
    if grayscale:
        basestr += "-depth 8 -type Grayscale -depth 8 "
    basestr += "-resize %sx%s " % (str(int(resize_scale_width * plot_width)),
                                   str(int(resize_scale_height * plot_height)))
    basestr += gif_name
    print("Attempting gif")
    print(basestr)
    subprocess.call(basestr, shell=True)
    filelist = glob.glob("__*giftmp_%s.png" % pre)
    for f in filelist:
        os.remove(f)


def get_pthbldr_models_dir(verbose=True):
    checkpoint_dir = os.getenv("PTHBLDR_MODELS", os.path.join(
        os.path.expanduser("~"), "pthbldr_models"))

    # Figure out if this is necessary to run on localdisk @ U de M
    if _special_check(verbose=verbose):
        new_checkpoint_dir = "/Tmp/" + USER + "/pthbldr_models"
        if os.path.exists("/Tmp/" + USER):
            # on a MILA machine!
            checkpoint_dir = new_checkpoint_dir

    return checkpoint_dir


def get_resource_dir(name, resource_dir=None, folder=None, create_dir=True):
    """ Get dataset directory path """
    # Only used for JS downloader
    if resource_dir == None:
        resource_dir = get_pthbldr_models_dir(verbose=False)
    if folder is None:
        resource_dir = os.path.join(resource_dir, name)
    else:
        resource_dir = os.path.join(resource_dir, folder)
    if create_dir:
        if not os.path.exists(resource_dir):
            os.makedirs(resource_dir)
    return resource_dir


def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.next()
        return cr
    return start


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            logger.info("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        logger.info("Downloading: %s Bytes: %s" % (server_fname, file_size))
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
                logger.info(status)
                p += progress_update_percentage


def get_checkpoint_dir(checkpoint_dir=None, folder=None, create_dir=True):
    """ Get checkpoint directory path """
    if checkpoint_dir is None:
        checkpoint_dir = get_pthbldr_models_dir()

    if folder is None:
        checkpoint_name = get_script()
        checkpoint_import_time = get_checkpoint_import_time()
        checkpoint_uuid = get_checkpoint_uuid()
        tmp = checkpoint_dir + os.path.sep + checkpoint_name + "_" + checkpoint_import_time  + "_" + checkpoint_uuid
        checkpoint_dir = tmp
    else:
        checkpoint_dir = os.path.join(checkpoint_dir, folder)

    if not os.path.exists(checkpoint_dir) and create_dir:
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def filled_js_template_from_results_dict(results_dict, default_show="all"):
    # Uses arbiter strings in the template to split the template and stick
    # values in
    partial_path = get_resource_dir("js_plot_dependencies")
    full_path = os.path.join(partial_path, "master.zip")
    url = "http://github.com/kastnerkyle/simple_template_plotter/archive/master.zip"
    if not os.path.exists(full_path):
        logger.info("Downloading plotter template code from %s" % url)
        download(url, full_path)
        zip_ref = zipfile.ZipFile(full_path, 'r')
        zip_ref.extractall(partial_path)
        zip_ref.close()

    js_path = os.path.join(partial_path, "simple_template_plotter-master")
    template_path =  os.path.join(js_path, "template.html")
    f = open(template_path, mode='r')
    all_template_lines = f.readlines()
    f.close()
    imports_split_index = [n for n, l in enumerate(all_template_lines)
                           if "IMPORTS_SPLIT" in l][0]
    data_split_index = [n for n, l in enumerate(all_template_lines)
                        if "DATA_SPLIT" in l][0]
    log_split_index = [n for n, l in enumerate(all_template_lines)
                       if "LOGGING_SPLIT" in l][0]
    first_part = all_template_lines[:imports_split_index]
    imports_part = []
    js_files_path = os.path.join(js_path, "js")
    js_file_names = ["jquery-1.9.1.js", "knockout-3.0.0.js",
                     "highcharts.js", "exporting.js"]
    js_files = [os.path.join(js_files_path, jsf) for jsf in js_file_names]
    for js_file in js_files:
        with open(js_file, "r") as f:
            imports_part.extend(
                ["<script>\n"] + f.readlines() + ["</script>\n"])
    post_imports_part = all_template_lines[
        imports_split_index + 1:data_split_index]
    log_part = all_template_lines[data_split_index + 1:log_split_index]
    last_part = all_template_lines[log_split_index + 1:]

    def gen_js_field_for_key_value(key, values, show=True):
        assert type(values) is list
        if isinstance(values[0], (np.generic, np.ndarray)):
            values = [float(v.ravel()) for v in values]
        maxlen = 1500
        if len(values) > maxlen:
            values = list(np.interp(np.linspace(0, len(values), maxlen),
                          np.arange(len(values)), values))
        show_key = "true" if show else "false"
        return "{\n    name: '%s',\n    data: %s,\n    visible: %s\n},\n" % (
            str(key), str(values), show_key)

    data_part = [gen_js_field_for_key_value(k, results_dict[k], True)
                 if k in default_show or default_show == "all"
                 else gen_js_field_for_key_value(k, results_dict[k], False)
                 for k in sorted(results_dict.keys())]
    all_filled_lines = first_part + imports_part + post_imports_part
    all_filled_lines = all_filled_lines + data_part + log_part
    # add logging output
    tmp = copy.copy(string_f)
    tmp.seek(0)
    log_output = tmp.readlines()
    del tmp
    all_filled_lines = all_filled_lines + log_output + last_part
    return all_filled_lines


def save_results_as_html(save_path, results_dict, use_resource_dir=True,
                         default_no_show="_auto", latest_tag=None):
    show_keys = [k for k in results_dict.keys()
                 if default_no_show not in k]
    as_html = filled_js_template_from_results_dict(
        results_dict, default_show=show_keys)
    if use_resource_dir:
        save_path = os.path.join(get_checkpoint_dir(), save_path)
    logger.info("Saving HTML results %s" % save_path)
    with open(save_path, "w") as f:
        f.writelines(as_html)
    if latest_tag is not None:
        latest_path = os.path.join(get_checkpoint_dir(), latest_tag + "_latest.html")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(save_path, latest_path)
    logger.info("Completed HTML results saving %s" % save_path)


@coroutine
def threaded_timed_writer(sleep_time=15 * 60, tag=None):
    """
    Expects to be sent a tuple of
    (objective,
    ((results_save_path, results_dict),
     (weights_save_path, checkpoint_dict),
     (checkpoint_save_path, checkpoint_dict)))

    Alternatively, pass None to bypass saving for that entry.

    (objective,
    ((results_save_path, results_dict),
     None,
     None))
    """
    messages = []

    def run_thread(msg_queue):
        # always save the very first one
        last_time = time.time() - (sleep_time + 1)
        while True:
            # avoid busy loop
            time.sleep(0.25)
            while len(messages) > 5:
                wq = [mm[0] for mm in msg_queue]
                max_i = wq.index(max(wq))
                messages.pop(max_i)

            time_flag = (time.time() - last_time) > sleep_time
            # check if train loop has set FINALIZE_TRAINING
            # if so, write out the best one and exit
            train_flag = _get_finalize_train()
            if time_flag or train_flag:
                if len(msg_queue) > 0:
                    wq = [mm[0] for mm in msg_queue]
                    min_i = wq.index(min(wq))
                    r = msg_queue.pop(min_i)
                    # unused priority
                    p = r[0]
                    item = r[1:]
                    # remove extra bracketing
                    item = item[0]
                    last_time = time.time()
                    if item is GeneratorExit:
                        return
                    else:
                        results_tup, weights_tup, checkpoint_tup = item
                        if results_tup is not None:
                            results_save_path, results_dict = results_tup
                            save_results_as_html(results_save_path,
                                                 results_dict,
                                                 latest_tag=tag)
                        if weights_tup is not None:
                            logging.info("Unable to save weights, NYI")
                            weights_save_path, items_dict = weights_tup
                            save_weights(weights_save_path, items_dict,
                                         latest_tag=tag)
                        if checkpoint_tup is not None:
                            logging.info("Unable to save checkpoint, NYI")
                            checkpoint_save_path, pickle_item = checkpoint_tup
                            save_checkpoint(checkpoint_save_path, pickle_item,
                                            latest_tag=tag)
                        # write the last one if training is done
                        # but do not stop on a "results only" save
                        artifact_flag = checkpoint_tup is not None or weights_tup is not None
                        if train_flag and artifact_flag:
                            logger.info("Last checkpoint written, exiting save thread")
                            return

    t = threading.Thread(target=run_thread, args=(messages,)).start()
    try:
        # Some of this logic (conversion to int) leftover from priority queue
        # May go back to that in time
        last_best = np.inf
        n = -1
        while True:
            ii = (yield)
            if ii[0] < last_best:
                n = n - 1
                last_best = ii[0]
                messages.append((n, ii[1:]))
            else:
                messages.append((n + 1, ii[1:]))
    except GeneratorExit:
        messages.append((1, GeneratorExit))


class TrainingLoop(object):
    """
    # TODO: add model saving function call? reference to the model for torch?
    Runs the loop - thin wrapper for serializing

    loop functions should return a list of costs, and accept 2 arguments:
    train_step(itr, extra_info)

    where extra_info may contain information useful for a variety of tasks

    The iterators *must* reset themselves, then raise StopIteration

    checkpoint_every_n_epochs - useful for reducing disk writes when there are many epochs
    checkpoint_every_n_updates - useful for models where 1 epoch would have many updates
    checkpoint_every_n_seconds - useful for models where 1 epoch takes a long time
    write_every_n_seconds - the frequency at which the best checkpoint according to the train and valid objectives gets written

    monitor frequency
    skip_minimums - skip checkpoints based on minimum training/valid
    skip_intermediates - skip within epoch checkpoints
    skip_most_recents - skip writing most recent results html
    """
    def __init__(self, train_loop_function, train_itr,
                 valid_loop_function, valid_itr,
                 n_epochs, checkpoint_dict,
                 checkpoint_delay=0,
                 checkpoint_every_n_epochs=1,
                 checkpoint_every_n_updates=np.inf,
                 checkpoint_every_n_seconds=np.inf,
                 write_every_n_seconds=15 * 60,
                 monitor_frequency=1000,
                 skip_minimums=False,
                 skip_intermediates=True,
                 skip_most_recents=False):
        self.train_loop_function = train_loop_function
        self.train_itr = train_itr

        self.valid_loop_function = valid_loop_function
        self.valid_itr = valid_itr

        self.n_epochs = n_epochs
        self.checkpoint_dict = checkpoint_dict

        # These parameters should be serialized
        self.checkpoint_delay = checkpoint_delay
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.checkpoint_every_n_updates = checkpoint_every_n_updates
        self.checkpoint_every_n_seconds = checkpoint_every_n_seconds
        self.write_every_n_seconds = write_every_n_seconds
        self.monitor_frequency = monitor_frequency
        self.skip_minimums = skip_minimums
        self.skip_intermediates = skip_intermediates
        self.skip_most_recents = skip_most_recents

        # tracker to ensure restarting at the correct minibatch
        self.num_train_minibatches_run = -1

    def __getstate__(self):
        skiplist = [self.train_loop_function,
                    self.train_itr,
                    self.valid_loop_function,
                    self.valid_itr,
                    self.n_epochs,
                    self.checkpoint_dict]
        return {k:v for k, v in self.__dict__.items() if v not in skiplist}

    def refresh(self, train_loop_function, train_itr,
                valid_loop_function, valid_itr,
                n_epochs,
                checkpoint_dict):
        # Must refresh after reloading from pkl
        self.train_loop_function = train_loop_function
        self.train_itr = train_itr

        self.valid_loop_function = valid_loop_function
        self.valid_itr = valid_itr
        self.n_epochs = n_epochs
        self.checkpoint_dict = checkpoint_dict

    def run(self):
        run_loop(self.train_loop_function, self.train_itr,
                 self.valid_loop_function, self.valid_itr,
                 self.n_epochs,
                 self.checkpoint_dict,
                 self.checkpoint_delay,
                 self.checkpoint_every_n_epochs,
                 self.checkpoint_every_n_updates,
                 self.checkpoint_every_n_seconds,
                 self.write_every_n_seconds,
                 self.monitor_frequency,
                 self.skip_minimums,
                 self.skip_intermediates,
                 self.skip_most_recents,
                 self.num_train_minibatches_run,
                 self)


def run_loop(train_loop_function, train_itr,
             valid_loop_function, valid_itr,
             n_epochs, checkpoint_dict,
             checkpoint_delay=10, checkpoint_every_n_epochs=1,
             checkpoint_every_n_updates=np.inf,
             checkpoint_every_n_seconds=10 * 60,
             write_every_n_seconds=15 * 60,
             monitor_frequency=1000, skip_minimums=False,
             skip_intermediates=True, skip_most_recents=False,
             skip_n_train_minibatches=-1,
             stateful_object=None):
    """
    TODO: add upload fields to add data to an html and save a copy?
    """
    ignore_keys = []

    train_loop = train_loop_function
    valid_loop = valid_loop_function
    ident = checkpoint_uuid
    random_state = np.random.RandomState(2177)
    monitor_prob = 1. / monitor_frequency

    non_ignored_keys = [k for k in checkpoint_dict.keys()
                        if k not in ignore_keys]
    if len(non_ignored_keys) > 0:
        overall_train_costs = checkpoint_dict["train_costs"]
        overall_valid_costs = checkpoint_dict["valid_costs"]
        # Auto tracking times
        overall_epoch_deltas = checkpoint_dict["epoch_deltas_auto"]
        overall_epoch_times = checkpoint_dict["epoch_times_auto"]
        overall_train_deltas = checkpoint_dict["train_deltas_auto"]
        overall_train_times = checkpoint_dict["train_times_auto"]
        overall_valid_deltas = checkpoint_dict["valid_deltas_auto"]
        overall_valid_times = checkpoint_dict["valid_times_auto"]
        overall_checkpoint_deltas = checkpoint_dict["checkpoint_deltas_auto"]
        overall_checkpoint_times = checkpoint_dict["checkpoint_times_auto"]
        overall_joint_deltas = checkpoint_dict["joint_deltas_auto"]
        overall_joint_times = checkpoint_dict["joint_times_auto"]
        overall_train_checkpoint = checkpoint_dict["train_checkpoint_auto"]
        overall_valid_checkpoint = checkpoint_dict["valid_checkpoint_auto"]
        keys_checked = ["train_costs",
                        "valid_costs",
                        "epoch_deltas_auto",
                        "epoch_times_auto",
                        "train_deltas_auto",
                        "train_times_auto",
                        "valid_deltas_auto",
                        "valid_times_auto",
                        "checkpoint_deltas_auto",
                        "checkpoint_times_auto",
                        "joint_deltas_auto",
                        "joint_times_auto",
                        "train_checkpoint_auto",
                        "valid_checkpoint_auto"]
        extra_keys = [k for k in checkpoint_dict.keys() if "zoom_" in k]
        extra_keys += ["epoch_count_auto"]
        not_handled = [k for k in checkpoint_dict.keys()
                       if k not in keys_checked
                       and k not in extra_keys
                       and k not in ignore_keys]
        if len(not_handled) > 0:
            raise ValueError("Unhandled keys %s in checkpoint_dict, exiting..." % not_handled)

        epoch_time_total = overall_epoch_times[-1]
        train_time_total = overall_train_times[-1]
        valid_time_total = overall_valid_times[-1]
        checkpoint_time_total = overall_checkpoint_times[-1]
        joint_time_total = overall_joint_times[-1]

        start_epoch = len(overall_train_costs)
    else:
        overall_train_costs = []
        overall_valid_costs = []
        overall_train_checkpoint = []
        overall_valid_checkpoint = []

        epoch_time_total = 0
        train_time_total = 0
        valid_time_total = 0
        checkpoint_time_total = 0
        joint_time_total = 0
        overall_epoch_times = []
        overall_epoch_deltas = []
        overall_train_times = []
        overall_train_deltas = []
        overall_valid_times = []
        overall_valid_deltas = []
        # Add zeros to avoid errors
        overall_checkpoint_times = [0]
        overall_checkpoint_deltas = [0]
        overall_joint_times = [0]
        overall_joint_deltas = [0]
        start_epoch = 0

    script = get_script()
    hostname = socket.gethostname()
    logger.info("Host %s, script %s" % (hostname, script))
    """
    logger.info("Model parameter summary")
    logger.info("-----------------------")
    total = 0
    for k, v in get_params().items():
        shp = v.get_value().shape
        logger.info("%s: %s" % (k, shp))
        total += np.prod(shp)
    logger.info("Total parameter count %f M" % (total / 1E6))
    """

    # Timed versus forced here
    tcw = threaded_timed_writer(write_every_n_seconds, tag="time")
    vcw = threaded_timed_writer(write_every_n_seconds, tag="valid")

    if _special_check(verbose=False):
        fcw = threaded_timed_writer(sleep_time=write_every_n_seconds,
                                    tag="force")
    else:
        fcw = threaded_timed_writer(sleep_time=0, tag="force")

    best_train_checkpoint_pickle = None
    best_train_checkpoint_epoch = 0
    best_valid_checkpoint_pickle = None
    best_train_checkpoint_epoch = 0
    # If there are more than 1M minibatches per epoch this will break!
    # Not reallocating buffer greatly helps fast training models though
    # Also we have bigger problems if there are 1M minibatches per epoch...
    # This will get sliced down to the correct number of minibatches down below
    train_costs = [0.] * 1000000
    valid_costs = [0.] * 1000000
    overall_start = time.time()
    try:
        for e in range(start_epoch, start_epoch + n_epochs):
            logger.info(" ")
            e_i = e + 1
            joint_start = time.time()
            epoch_start = time.time()
            logger.info("Starting training, epoch %i" % e_i)
            train_mb_count = 0
            valid_mb_count = 0
            results_dict = {k: v for k, v in checkpoint_dict.items()
                            if k not in ignore_keys}
            this_results_dict = results_dict
            loop_info = {"epoch": e_i}
            try:
                # train loop
                train_start = time.time()
                last_time_checkpoint = train_start
                status_line_count = 0
                while True:
                    if train_mb_count < skip_n_train_minibatches:
                        train_mb_count += 1
                        continue
                    partial_train_costs = train_loop(train_itr, loop_info)
                    train_costs[train_mb_count] = np.mean(partial_train_costs)
                    tc = train_costs[train_mb_count]
                    if np.isnan(tc):
                        logger.info("NaN detected in train cost, update %i" % train_mb_count)
                        raise StopIteration("NaN detected in train")

                    train_mb_count += 1

                    # hardcoded 15 s status update
                    if (time.time() - last_time_checkpoint) >= 15:
                        if status_line_count < 40:
                            print(".", end="")
                        else:
                            status_line_count = 0
                            print("")
                            print(".", end="")
                        status_line_count += 1

                    if (train_mb_count % checkpoint_every_n_updates) == 0:
                        print("")
                        checkpoint_save_path = "%s_model_update_checkpoint_%i.pkl" % (ident, train_mb_count)
                        weights_save_path = "%s_model_update_weights_%i.npz" % (ident, train_mb_count)
                        results_save_path = "%s_model_update_results_%i.html" % (ident, train_mb_count)
                        # Use pickle to preserve relationships between keys
                        # while still copying buffers
                        #copy_pickle = _dumps(checkpoint_dict)
                        #copy_dict = pickle.loads(copy_pickle)
                        #del copy_pickle

                        logger.info("Update checkpoint after train mb %i" % train_mb_count)
                        logger.info("Current mean cost %f" % np.mean(partial_train_costs))
                        # TODO: Add epoch traces to output plots
                        this_results_dict["this_epoch_train_auto"] = train_costs[:train_mb_count]
                        tmb = train_costs[:train_mb_count]
                        running_train_mean = np.cumsum(tmb) / (np.arange(train_mb_count) + 1)
                        # needs to be a list
                        running_train_mean = list(running_train_mean)
                        this_results_dict["this_epoch_train_mean_auto"] = running_train_mean

                        # FIXME: last mean vs min vs last value
                        objective = running_train_mean[-1]
                        tcw.send((objective,
                                 (results_save_path, this_results_dict),
                                 None,
                                 None))
                                 #(weights_save_path, copy_dict),
                                 #(checkpoint_save_path, copy_dict)))

                    elif (time.time() - last_time_checkpoint) >= checkpoint_every_n_seconds:
                        print("")
                        # Time since training started
                        time_diff = time.time() - overall_start
                        last_time_checkpoint = time.time()
                        checkpoint_save_path = "%s_model_time_checkpoint_%i.pkl" % (ident, int(time_diff))
                        weights_save_path = "%s_model_time_weights_%i.npz" % (ident, int(time_diff))
                        results_save_path = "%s_model_time_results_%i.html" % (ident, int(time_diff))
                        # Use pickle to preserve relationships between keys
                        # while still copying buffers
                        #copy_pickle = _dumps(checkpoint_dict)
                        #copy_dict = pickle.loads(copy_pickle)
                        #del copy_pickle

                        logger.info("Time checkpoint after train mb %i" % train_mb_count)
                        logger.info("Current mean cost %f" % np.mean(partial_train_costs))
                        this_results_dict["this_epoch_train_auto"] = train_costs[:train_mb_count]
                        tmb = train_costs[:train_mb_count]
                        running_train_mean = np.cumsum(tmb) / (np.arange(train_mb_count) + 1)
                        # needs to be a list
                        running_train_mean = list(running_train_mean)
                        this_results_dict["this_epoch_train_mean_auto"] = running_train_mean

                        # FIXME: last mean vs min vs last value?
                        objective = running_train_mean[-1]
                        tcw.send((objective,
                                 (results_save_path, this_results_dict),
                                 None,
                                 None))
                                 #(weights_save_path, copy_dict),
                                 #(checkpoint_save_path, copy_dict)))
                        #del copy_dict
                    draw = random_state.rand()
                    if draw < monitor_prob and not skip_intermediates:
                        print("")
                        logger.info("Starting train mb %i" % train_mb_count)
                        logger.info("Current mean cost %f" % np.mean(partial_train_costs))
                        results_save_path = "%s_intermediate_results.html" % ident
                        this_results_dict["this_epoch_train_auto"] = train_costs[:train_mb_count]

                        objective = np.mean(partial_train_costs)
                        fcw.send((objective,
                                 (results_save_path, this_results_dict),
                                 None,
                                 None))
            except StopIteration:
                train_stop = time.time()

                # Save the epoch trace
                final_train_costs = train_costs[:train_mb_count]
                tmb = final_train_costs
                running_train_mean = np.cumsum(tmb) / (np.arange(train_mb_count) + 1)
                # needs to be a list
                running_train_mean = list(running_train_mean)
                this_results_dict["this_epoch_train_auto"] = final_train_costs
                this_results_dict["this_epoch_train_mean_auto"] = running_train_mean

                print("")
                logger.info("Starting validation, epoch %i" % e_i)
                valid_start = time.time()
                try:
                    # Valid loop
                    while True:
                        partial_valid_costs = valid_loop(valid_itr, loop_info)
                        valid_costs[valid_mb_count] = np.mean(partial_valid_costs)
                        vc = valid_costs[valid_mb_count]
                        if np.isnan(vc):
                            logger.info("NaN detected in valid cost, minibatch %i" % valid_mb_count)
                            raise StopIteration("NaN detected in valid")
                        valid_mb_count += 1
                        draw = random_state.rand()
                        if draw < monitor_prob and not skip_intermediates:
                            logger.info("Valid mb %i" % valid_mb_count)
                            logger.info("Current validation mean cost %f" % np.mean(
                                valid_costs))
                            results_save_path = "%s_intermediate_results.html" % ident
                            this_results_dict["this_epoch_valid_auto"] = valid_costs[:valid_mb_count]

                            objective = np.mean(valid_costs)
                            fcw.send((objective,
                                     (results_save_path, this_results_dict),
                                     None,
                                     None))
                except StopIteration:
                    pass
                print("")
                valid_stop = time.time()
                epoch_stop = time.time()
                final_valid_costs = valid_costs[:valid_mb_count]

                # Logging and tracking training statistics
                epoch_time_delta = epoch_stop - epoch_start
                epoch_time_total += epoch_time_delta
                overall_epoch_deltas.append(epoch_time_delta)
                overall_epoch_times.append(epoch_time_total)

                train_time_delta = train_stop - train_start
                train_time_total += train_time_delta
                overall_train_deltas.append(train_time_delta)
                overall_train_times.append(train_time_total)

                valid_time_delta = valid_stop - valid_start
                valid_time_total += valid_time_delta
                overall_valid_deltas.append(valid_time_delta)
                overall_valid_times.append(valid_time_total)

                mean_epoch_train_cost = np.mean(final_train_costs)
                # np.inf trick to avoid taking the min of length 0 list
                old_min_train_cost = min(overall_train_costs + [np.inf])
                if np.isnan(mean_epoch_train_cost):
                    logger.info("Previous train costs %s" % overall_train_costs[-5:])
                    print(train_mb_count)
                    logger.info("NaN detected in train cost, epoch %i" % e)
                    raise StopIteration("NaN detected in train")
                overall_train_costs.append(mean_epoch_train_cost)

                mean_epoch_valid_cost = np.mean(final_valid_costs)
                old_min_valid_cost = min(overall_valid_costs + [np.inf])
                if np.isnan(mean_epoch_valid_cost):
                    logger.info("Previous valid costs %s" % overall_valid_costs[-5:])
                    print(train_mb_count)
                    logger.info("NaN detected in valid cost, epoch %i" % e)
                    logger.info("Replacing with previous valid cost")
                    mean_epoch_valid_cost = overall_valid_costs[-1]
                    # try to keep going
                    #raise StopIteration("NaN detected in valid")
                overall_valid_costs.append(mean_epoch_valid_cost)

                if mean_epoch_train_cost < old_min_train_cost:
                    overall_train_checkpoint.append(mean_epoch_train_cost)
                else:
                    overall_train_checkpoint.append(old_min_train_cost)

                if mean_epoch_valid_cost < old_min_valid_cost:
                    overall_valid_checkpoint.append(mean_epoch_valid_cost)
                else:
                    overall_valid_checkpoint.append(old_min_valid_cost)

                checkpoint_dict["train_costs"] = overall_train_costs
                checkpoint_dict["valid_costs"] = overall_valid_costs
                # Auto tracking times
                checkpoint_dict["epoch_deltas_auto"] = overall_epoch_deltas
                checkpoint_dict["epoch_times_auto"] = overall_epoch_times
                checkpoint_dict["epoch_count_auto"] = list([i + 1 for i in range(len(overall_epoch_times))])

                checkpoint_dict["train_deltas_auto"] = overall_train_deltas
                checkpoint_dict["train_times_auto"] = overall_train_times

                checkpoint_dict["valid_deltas_auto"] = overall_valid_deltas
                checkpoint_dict["valid_times_auto"] = overall_valid_times

                checkpoint_dict["checkpoint_deltas_auto"] = overall_checkpoint_deltas
                checkpoint_dict["checkpoint_times_auto"] = overall_checkpoint_times

                checkpoint_dict["joint_deltas_auto"] = overall_joint_deltas
                checkpoint_dict["joint_times_auto"] = overall_joint_times

                # Tracking if checkpoints are made
                checkpoint_dict["train_checkpoint_auto"] = overall_train_checkpoint
                checkpoint_dict["valid_checkpoint_auto"] = overall_valid_checkpoint

                this_keys = this_results_dict.keys()
                if "this_epoch_train_auto" in this_keys:
                    checkpoint_dict["zoom_train_trace_epoch_%i_auto" % e_i] = this_results_dict["this_epoch_train_auto"]

                if "this_epoch_train_mean_auto" in this_keys:
                    checkpoint_dict["zoom_train_trace_epoch_mean_%i_auto" % e_i] = this_results_dict["this_epoch_train_mean_auto"]

                all_zoom_keys = [ck for ck in checkpoint_dict.keys()
                                 if "zoom_" in ck]
                zoom_keys = [ck for ck in checkpoint_dict.keys()
                             if (("zoom_" in ck)
                                 and ("epoch_mean" not in ck)
                                 and ("epoch_%s" % e_i not in ck))]

                # keep traces with:
                # overall min
                # overall max
                # average min
                # average max
                # lowest max vs median
                # highest max vs median
                matched_keys = {}
                matched_num = {}

                matched_num["local_min"] = np.inf
                matched_keys["local_min"] = None

                matched_num["local_max"] = -np.inf
                matched_keys["local_max"] = None

                matched_num["average_min"] = np.inf
                matched_keys["average_min"] = None

                matched_num["average_max"] = -np.inf
                matched_keys["average_max"] = -np.inf

                matched_num["lowest_max_vs_med"] = np.inf
                matched_keys["lowest_max_vs_med"] = None

                matched_keys["highest_max_vs_med"] = -np.inf
                matched_num["highest_max_vs_med"] = None

                matched_num["lowest_med_vs_min"] = np.inf
                matched_keys["lowest_med_vs_min"] = None

                matched_keys["highest_med_vs_min"] = -np.inf
                matched_num["highest_med_vs_min"] = None

                if len(zoom_keys) > len(list(matched_keys.keys())):
                    for zk in zoom_keys:
                        mn = min(checkpoint_dict[zk])
                        mx = max(checkpoint_dict[zk])
                        avg = np.mean(checkpoint_dict[zk])
                        md = np.median(checkpoint_dict[zk])

                        if mn < matched_num["local_min"]:
                            matched_num["local_min"] = mn
                            matched_keys["local_min"] = zk

                        if mx > matched_num["local_max"]:
                            matched_num["local_max"] = mx
                            matched_keys["local_max"] = zk

                        if avg < matched_num["average_min"]:
                            matched_num["average_min"] = avg
                            matched_keys["average_min"] = zk

                        if avg > matched_num["average_max"]:
                            matched_num["average_max"] = avg
                            matched_keys["average_max"] = zk

                        mx_vs_md = mx - md
                        if mx_vs_md > matched_num["highest_max_vs_med"]:
                            matched_num["highest_max_vs_med"] = mx_vs_md
                            matched_keys["highest_max_vs_med"] = zk

                        if mx_vs_md < matched_num["lowest_max_vs_med"]:
                            matched_num["lowest_max_vs_med"] = mx_vs_md
                            matched_keys["lowest_max_vs_med"] = zk

                        md_vs_mn = md - mn
                        if md_vs_mn > matched_num["highest_med_vs_min"]:
                            matched_num["highest_med_vs_min"] = md_vs_mn
                            matched_keys["highest_med_vs_min"] = zk

                        if md_vs_mn < matched_num["lowest_med_vs_min"]:
                            matched_num["lowest_med_vs_min"] = md_vs_mn
                            matched_keys["lowest_med_vs_min"] = zk

                    rev_matched = {v: k for k, v in matched_keys.items()}
                    keep_keys = rev_matched.keys()
                    keep_epochs = [int(kk.split("_")[-2]) for kk in keep_keys]
                    keep_epochs = keep_epochs + [e_i]

                    delete_keys = []
                    for azk in all_zoom_keys:
                        if any(["_%i_" % ke in azk for ke in keep_epochs]):
                            continue
                        else:
                            delete_keys.append(azk)

                    keep_keys = [azk for azk in all_zoom_keys if azk not in delete_keys]
                    for zk in all_zoom_keys:
                        if zk in keep_keys:
                            # could add logic to change name
                            pass
                        elif zk in delete_keys:
                            del checkpoint_dict[zk]
                        else:
                            raise ValueError("Unexpected error, key %s" % zk)

                logger.info("Host %s, script %s" % (hostname, script))
                logger.info("Epoch %i complete" % e_i)
                logger.info("Epoch mean train cost %f" % mean_epoch_train_cost)
                logger.info("Epoch mean valid cost %f" % mean_epoch_valid_cost)
                logger.info("Previous train costs %s" % overall_train_costs[-5:])
                logger.info("Previous valid costs %s" % overall_valid_costs[-5:])

                results_dict = {k: v for k, v in checkpoint_dict.items()
                                if k not in ignore_keys}

                # Checkpointing part
                checkpoint_start = time.time()
                if e < checkpoint_delay or skip_minimums:
                    pass
                elif mean_epoch_valid_cost < old_min_valid_cost:
                    logger.info("Checkpointing valid...")
                    # Using dumps so relationship between keys in the pickle
                    # is preserved
                    checkpoint_save_path = "%s_model_checkpoint_valid_%i.pkl" % (ident, e_i)
                    weights_save_path = "%s_model_weights_valid_%i.npz" % (ident, e_i)
                    results_save_path = "%s_model_results_valid_%i.html" % (ident, e_i)
                    #best_valid_checkpoint_pickle = _dumps(checkpoint_dict)
                    #best_valid_checkpoint_epoch = e
                    # preserve key relations
                    #copy_dict = pickle.loads(best_valid_checkpoint_pickle)

                    objective = mean_epoch_valid_cost
                    vcw.send((objective,
                             (results_save_path, this_results_dict),
                             None,
                             None))
                             #(weights_save_path, copy_dict),
                             #(checkpoint_save_path, copy_dict)))

                    if mean_epoch_train_cost < old_min_train_cost:
                        checkpoint_save_path = "%s_model_checkpoint_train_%i.pkl" % (ident, e_i)
                        weights_save_path = "%s_model_weights_train_%i.npz" % (ident, e_i)
                        results_save_path = "%s_model_results_train_%i.html" % (ident, e_i)
                        #best_train_checkpoint_pickle = _dumps(checkpoint_dict)
                        #best_train_checkpoint_epoch = e

                        objective = mean_epoch_train_cost
                        vcw.send((objective,
                                (results_save_path, this_results_dict),
                                None,
                                None))
                                #(weights_save_path, copy_dict),
                                #(checkpoint_save_path, copy_dict)))
                    logger.info("Valid checkpointing complete.")
                elif mean_epoch_train_cost < old_min_train_cost:
                    logger.info("Checkpointing train...")
                    checkpoint_save_path = "%s_model_checkpoint_train_%i.pkl" % (ident, e_i)
                    weights_save_path = "%s_model_weights_train_%i.npz" % (ident, e_i)
                    results_save_path = "%s_model_results_train_%i.html" % (ident, e_i)
                    #best_train_checkpoint_pickle = _dumps(checkpoint_dict)
                    #best_train_checkpoint_epoch = e
                    # preserve key relations
                    #copy_dict = pickle.loads(best_train_checkpoint_pickle)

                    objective = mean_epoch_train_cost
                    vcw.send((objective,
                            (results_save_path, this_results_dict),
                            None,
                            None))
                            #(weights_save_path, copy_dict),
                            #(checkpoint_save_path, copy_dict)))
                    logger.info("Train checkpointing complete.")

                if e < checkpoint_delay:
                    pass
                    # Don't skip force checkpoints after default delay
                    # Printing already happens above
                elif((e % checkpoint_every_n_epochs) == 0) or (e == (n_epochs - 1)):
                    logger.info("Checkpointing force...")
                    checkpoint_save_path = "%s_model_checkpoint_%i.pkl" % (ident, e_i)
                    weights_save_path = "%s_model_weights_%i.npz" % (ident, e_i)
                    results_save_path = "%s_model_results_%i.html" % (ident, e_i)
                    # Use pickle to preserve relationships between keys
                    # while still copying buffers
                    #copy_pickle = _dumps(checkpoint_dict)
                    #copy_dict = pickle.loads(copy_pickle)

                    objective = mean_epoch_train_cost
                    fcw.send((objective,
                            (results_save_path, this_results_dict),
                            None,
                            None))
                            #(weights_save_path, copy_dict),
                            #(checkpoint_save_path, copy_dict)))
                    # TODO: Make this better at handling time vs forced
                    # using somehting besides hardcoded force_latest.pkl
                    #write_dagbldr_lookup_file()
                    logger.info("Force checkpointing complete.")

                checkpoint_stop = time.time()
                joint_stop = time.time()

                if skip_most_recents:
                    pass
                else:
                    # Save latest
                    results_save_path = "%s_most_recent_results.html" % ident
                    objective = mean_epoch_train_cost
                    fcw.send((objective,
                             (results_save_path, results_dict),
                              None,
                              None))

                # Will show up next go around
                checkpoint_time_delta = checkpoint_stop - checkpoint_start
                checkpoint_time_total += checkpoint_time_delta
                overall_checkpoint_deltas.append(checkpoint_time_delta)
                overall_checkpoint_times.append(checkpoint_time_total)

                joint_time_delta = joint_stop - joint_start
                joint_time_total += joint_time_delta
                overall_joint_deltas.append(joint_time_delta)
                overall_joint_times.append(joint_time_total)
    except KeyboardInterrupt:
        logger.info("Training loop interrupted by user! Saving current best results.")

    if not skip_minimums:
        # Finalize saving best train and valid
        best_valid_checkpoint_dict = pickle.loads(best_valid_checkpoint_pickle)
        best_valid_results_dict = {k: v for k, v in best_valid_checkpoint_dict.items()
                                   if k not in ignore_keys}
        ee = best_valid_checkpoint_epoch
        checkpoint_save_path = "%s_model_checkpoint_valid_%i.pkl" % (ident, ee + 1)
        weights_save_path = "%s_model_weights_valid_%i.npz" % (ident, ee + 1)
        results_save_path = "%s_model_results_valid_%i.html" % (ident, ee + 1)

        objective = -np.inf
        fcw.send((objective,
                 (results_save_path, best_valid_results_dict),
                 None,
                 None))
                 #(weights_save_path, best_valid_checkpoint_dict),
                 #(checkpoint_save_path, best_valid_checkpoint_dict)))

        best_train_checkpoint_dict = pickle.loads(best_train_checkpoint_pickle)
        best_train_results_dict = {k: v for k, v in best_train_checkpoint_dict.items()
                                   if k not in ignore_keys}
        ee = best_train_checkpoint_epoch
        checkpoint_save_path = "%s_model_checkpoint_train_%i.pkl" % (ident, ee + 1)
        weights_save_path = "%s_model_weights_train_%i.npz" % (ident, ee + 1)
        results_save_path = "%s_model_results_train_%i.html" % (ident, ee + 1)

        objective = -np.inf
        fcw.send((objective,
                 (results_save_path, best_train_results_dict),
                 None,
                 None))
                 #(weights_save_path, best_train_checkpoint_dict),
                 #(checkpoint_save_path, best_train_checkpoint_dict)))

    logger.info("Loop finished, closing write threads (this may take a while!)")
    # set FINALIZE_TRAINING so that write threads know it is time to close
    _set_finalize_train()
    tcw.close()
    vcw.close()
    fcw.close()
