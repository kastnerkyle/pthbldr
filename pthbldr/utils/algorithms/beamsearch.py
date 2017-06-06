# Author: Kyle Kastner
# License: BSD 3-Clause
# See core implementations here http://geekyisawesome.blogspot.ca/2016/10/using-beam-search-to-generate-most.html
import numpy as np
import heapq
import collections
import multiprocessing
from multiprocessing import Pool
import functools
import time
from ...core import get_logger

logger = get_logger()

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


class Beam(object):
    """
    From http://geekyisawesome.blogspot.ca/2016/10/using-beam-search-to-generate-most.html
    For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
    This is so that if two prefixes have equal probabilities then a complete sentence is preferred
    over an incomplete one since (0.5, False, whatever_prefix) < (0.5, True, some_other_prefix)
    """
    def __init__(self, beam_width, init_beam=None, use_log=True,
                 stochastic=False, temperature=1.0, random_state=None):
        if init_beam is None:
            self.heap = list()
        else:
            self.heap = init_beam
            heapq.heapify(self.heap)
        self.stochastic = stochastic
        self.random_state = random_state
        self.temperature = temperature
        # use_log currently unused...
        self.use_log = use_log
        self.beam_width = beam_width

    def add(self, complete, score, prob, prefix):
        heapq.heappush(self.heap, (complete, score, prob, prefix))
        while len(self.heap) > self.beam_width:
            if self.stochastic:
                # use score instead...
                # same whether logspace or no?
                probs = np.array([h[1] for h in self.heap])
                probs = probs / self.temperature
                e_x = np.exp(probs - np.max(probs))
                s_x = e_x / e_x.sum()
                is_x = 1. - s_x
                is_x = is_x / is_x.sum()
                to_remove = self.random_state.multinomial(1, is_x).argmax()
                completed = [n for n, h in enumerate(self.heap) if h[0] == True]
                # Don't remove completed sentences by randomness
                if to_remove not in completed:
                    # there must be a faster way...
                    self.heap.pop(to_remove)
                    heapq.heapify(self.heap)
                else:
                    heapq.heappop(self.heap)
            else:
                # remove lowest score from heap
                heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)


def inner_beamsearch(probabilities_function, beam_width=10, clip_len=-1,
                     start_token="<START>", end_token="<EOS>", use_log=True,
                     renormalize=True, length_score=True,
                     stochastic=False, temperature=1.0,
                     random_state=None, eps=1E-9, verbose=False):
    """
    From http://geekyisawesome.blogspot.ca/2017/04/getting-top-n-most-probable-sentences.html

    returns a generator, which will yield beamsearched sequences in order of their probability

    "probabilities_function" returns a list of (next_prob, next_word) pairs given a prefix.
    this function should be in the outer scope of your python file, in order to work with multiprocessing

    "beam_width" is the number of prefixes to keep (so that instead of keeping the top 10 prefixes you can keep the top 100 for example).
    By making the beam search bigger you can get closer to the actual most probable sentence but it would also take longer to process.

    "clip_len" is a maximum length to tolerate, beyond which the most probable prefix is returned as an incomplete sentence.
    Without a maximum length, a faulty probabilities function which does not return a highly probable end token
    will lead to an infinite loop or excessively long garbage sentences.

    "start_token" can be a single string (token), or a sequence of tokens

    "end_token" is a single string (token), or a sequence of tokens that signifies end of the sequence. If token is a list of tuple, can set elements of last tuple to None for partial matching

    "use_log, renormalize, length_score" are all related to calculation of beams to keep
    and should improve results when True

    "stochastic" uses a different sampling algorithm for reducing/aggregating beams
    it should result in more diverse and interesting outputs

    "temperature" is the softmax temperature for the underlying stochastic
    beamsearch - the default of 1.0 is usually fine

    "random_state" is a np.random.RandomState() object, passed when using the
    stochastic beamsearch in order to control randomness

    "eps" minimum probability for log-space calculations, to avoid numerical issues
    """
    if stochastic:
        if random_state is None:
            raise ValueError("Must pass np.random.RandomState() object if stochastic=True")

    completed_beams = 0
    prev_beam = Beam(beam_width - completed_beams, None, use_log, stochastic,
                     temperature, random_state)
    try:
        basestring
    except NameError:
        basestring = str

    if isinstance(start_token, collections.Sequence) and not isinstance(start_token, basestring):
        start_token = start_token
    elif isinstance(start_token, basestring) and len(start_token) > 1:
        start_token = list(start_token)
    else:
        # make it a list with 1 entry
        start_token = [start_token]

    if isinstance(end_token, collections.Sequence) and not isinstance(end_token, basestring):
        end_token = end_token
        end_token_is_seq = True
    elif isinstance(end_token, basestring) and len(end_token) > 1:
        end_token = list(end_token)
        end_token_is_seq = True
    else:
        # make it a list with 1 entry
        end_token = [end_token]
        end_token_is_seq = False

    # can't compare none with string...
    if end_token_is_seq:
        # trying to see if it is a tuple token
        any_none = [et for et in end_token if not isinstance(et, basestring) and (et == None or (hasattr(et, '__len__') and None in et))]
        none_compare = True if len(any_none) > 0 else False
    else:
        any_none = []
        none_compare = False

    if none_compare:
        try:
            # one of these should barf
            et = end_token[0][0]
            # should barf...
            len(et)
            if not isinstance(et, basestring):
                raise ValueError("Shouldn't be string...")
            #raise AttributeError("End token with tuples should be passed as a list")
        except:
            pass

        if len(any_none) > 1 or any_none[-1] != end_token[-1]:
            raise ValueError("Can only compare to None for the last element! Change value for end_token, currently {}".format(end_token))

    if use_log:
        prev_beam.add(False, .0, .0, start_token)
    else:
        prev_beam.add(False, 1.0, 1.0, start_token)


    full_outputs = []
    while True:
        curr_beam = Beam(beam_width - completed_beams, None, use_log, stochastic,
                         temperature, random_state)
        if renormalize:
            sorted_prev_beam = sorted(prev_beam)
            # renormalize by the previous minimum value in the beam
            min_prob = sorted_prev_beam[0][1]
        else:
            if use_log:
                min_prob = 0.
            else:
                min_prob = 1.

        # Add complete sentences that do not yet have the best probability to the current beam, the rest prepare to add more words to them.
        for (complete, prefix_score, prefix_prob, prefix) in prev_beam:
            if complete == True:
                curr_beam.add(True, prefix_score, prefix_prob, prefix)
            else:
                # Get probability of each possible next word for the incomplete prefix
                for (next_prob, next_word) in probabilities_function(prefix):
                    # use eps tolerance to avoid log(0.) issues
                    if next_prob > eps:
                        n = next_prob
                    else:
                        n = eps

                    # score is renormalized prob
                    if use_log:
                        if length_score:
                            score = prefix_prob + np.log(n) + np.log(len(prefix)) - min_prob
                        else:
                            score = prefix_prob + np.log(n) - min_prob
                        prob = prefix_prob + np.log(n)
                    else:
                        if length_score:
                            score = (prefix_prob * n * len(prefix)) / min_prob
                        else:
                            score = (prefix_prob * n) / min_prob
                        prob = prefix_prob * n

                    if end_token_is_seq:
                        if len(end_token) > 1:
                            left_cmp = prefix[-len(end_token) + 1:] + [next_word]
                        else:
                            left_cmp = [next_word]
                        right_cmp = end_token
                        if none_compare:
                            lpartial = (left_cmp[:-1] == right_cmp[:-1])
                            if hasattr(left_cmp[-1], "__len__"):
                                rpartial = all([lc == rc for lc, rc in zip(left_cmp[-1], right_cmp[-1]) if rc is not None])
                            else:
                                assert right_cmp[-1] == None
                                # if none_compare is True then we don't need to match
                                rpartial = True
                            cmp_result = all([lpartial, rpartial])
                        else:
                            cmp_result = (left_cmp == right_cmp)
                    else:
                        left_cmp = next_word
                        right_cmp = end_token[0]
                        cmp_result = (left_cmp == right_cmp)

                    if cmp_result:
                        # If next word is the end token then mark prefix as complete
                        curr_beam.add(True, score, prob, prefix + [next_word])
                    else:
                        curr_beam.add(False, score, prob, prefix + [next_word])

        # Get all prefixes in beam sorted by completeness, then probability
        sorted_beam = sorted(curr_beam)

        any_removals = False
        while True:
            # Get highest probability prefix - heapq is sorted in ascending order
            (best_complete, best_score, best_prob, best_prefix) = sorted_beam[-1]
            if best_complete == True or len(best_prefix) - 1 == clip_len:
                # If most probable prefix is a complete sentence or has a length that
                # exceeds the clip length (ignoring the start token) then return it
                # yield best without start token, along with probability
                full_outputs.append((best_complete, best_score, best_prob, best_prefix))
                sorted_beam.pop()
                completed_beams += 1
                if verbose:
                    logger.info("Completed beams {} of {}".format(completed_beams, beam_width))
                any_removals = True
                # if there are no more sentences in the beam then stop checking
                if len(sorted_beam) == 0:
                    break
            else:
                break

        if any_removals == True:
            if len(sorted_beam) == 0:
                return full_outputs
                #break
            else:
                prev_beam = Beam(beam_width - completed_beams, sorted_beam, use_log,
                                 stochastic, temperature, random_state)
        else:
            prev_beam = curr_beam


def run_beamsearch(probabilities_function, beam_width, clip_len,
                    start_token, end_token, use_log,
                    renormalize, length_score, stochastic, temperature,
                    random_state, eps, verbose, n):
    # n unused, just for pool usage
    b = inner_beamsearch(probabilities_function, beam_width=beam_width,
                         clip_len=clip_len, start_token=start_token,
                         end_token=end_token, use_log=use_log,
                         renormalize=renormalize, length_score=length_score,
                         stochastic=stochastic, temperature=temperature,
                         random_state=random_state, eps=eps, verbose=verbose)
    return b


def beamsearch(probabilities_function, beam_width=10, clip_len=-1,
               start_token="<START>", end_token="<EOS>", use_log=True,
               renormalize=True, length_score=True,
               stochastic=False, temperature=1.0,
               random_state=None, eps=1E-9, verbose=False,
               beam_timeout=None):
    """
    If timeout is reached, returns an empty list (return current beams instead?)

    From http://geekyisawesome.blogspot.ca/2017/04/getting-top-n-most-probable-sentences.html

    returns a list of tuples (seqeunce, score, prob) in order from best score to worst

    "probabilities_function" returns a list of (next_prob, next_word) pairs given a prefix.

    "beam_width" is the number of prefixes to keep (so that instead of keeping the top 10 prefixes you can keep the top 100 for example).
    By making the beam search bigger you can get closer to the actual most probable sentence but it would also take longer to process.

    "clip_len" is a maximum length to tolerate, beyond which the most probable prefix is returned as an incomplete sentence.
    Without a maximum length, a faulty probabilities function which does not return a highly probable end token
    will lead to an infinite loop or excessively long garbage sentences.

    "start_token" can be a single string (token), or a sequence of tokens

    "end_token" is a single string (token), or a sequence of tokens that signifies end of the sequence. If token is a tuple, can set elements to None for partial matching

    "use_log, renormalize, length_score" are all related to calculation of beams to keep
    and should improve results when True

    "stochastic" uses a different sampling algorithm for reducing/aggregating beams
    it should result in more diverse and interesting outputs

    "temperature" is the softmax temperature for the underlying stochastic
    beamsearch - the default of 1.0 is usually fine

    "random_state" is a np.random.RandomState() object, passed when using the
    stochastic beamsearch in order to control randomness

    "eps" minimum probability for log-space calculations, to avoid numerical issues

    "verbose" to see timing of beamsearch

    "beam_timeout" None or float time in seconds to wait for beam completion
    """
    start_time = time.time()

    # may need a way to run for debugging...
    b = inner_beamsearch(probabilities_function, beam_width=beam_width,
                         clip_len=clip_len, start_token=start_token,
                         end_token=end_token, use_log=use_log,
                         renormalize=renormalize, length_score=length_score,
                         stochastic=stochastic, temperature=temperature,
                         random_state=random_state, eps=eps, verbose=verbose)
    pool = Pool(1)
    # don't do verbose inner loop
    ex = functools.partial(run_beamsearch, probabilities_function, beam_width,
                           clip_len, start_token, end_token, use_log,
                           renormalize, length_score, stochastic, temperature,
                           random_state, eps, False)
    if beam_timeout == None:
        beam_timeout = 10000000000000000000
    abortable_ex = functools.partial(abortable_worker, ex, timeout=beam_timeout)
    # only 1 job
    all_results = pool.map(abortable_ex, [1])
    pool.close()
    pool.join()
    all_results = all_results[0]
    end_time = time.time()

    # beamsearch was killed due to time
    if all_results[0] == "null":
        if verbose:
            logger.info("Beamsearch timed out, total time {}".format(end_time - start_time))
        return []

    # sort by score, top down
    all_results = sorted(all_results, reverse=True)
    # drop complete/incomplete designation
    # now prefix, score, prob
    all_completed = [ar[0] for ar in all_results]
    all_results = [(ar[3], ar[1], ar[2]) for ar in all_results]
    if verbose:
        if all_completed[0] == True:
            if clip_len > 0:
                if len(all_results[0]) < clip_len:
                    logger.info("At least one beam found end token!")

    if verbose:
        logger.info("Beamsearch complete, total time {}".format(end_time - start_time))
    return all_results
