from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
import torch
from torch.distributions import Categorical, Uniform

from collections import deque


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    text = utils.one_hot(some_text)
    predict_all = model.predict_all(some_text)[:, 0:len(some_text)]
    likelihoods = torch.mm(predict_all.t(), text)
    return sum(likelihoods.diag())


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    s = ""
    next_s = ""
    while len(s) <= max_length and next_s != '.':
        log_probs = model.predict_next(s)
        sample = Categorical(logits=log_probs).sample().item()
        next_s = utils.vocab[sample]
        # dist = Categorical(probs=None, logits=log_probs)
        # sample randomly from the log_probs distribution
        s += next_s
    return s

class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)

    def remove(self):
        from heapq import heappop
        if len(self.elements) > 0:
            val = heappop(self.elements)
            return val
        else:
            return None


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):

    # heap will sort by first element in tuple
    heap = TopNHeap(beam_size)
    term_heap = TopNHeap(n_results)
    visited = set()

    # initialize heap
    likelihoods = model.predict_next("")
    for i, l in enumerate(likelihoods):
        visited.add(utils.vocab[i])
        if utils.vocab[i] == '.':
            term_heap.add(( l, utils.vocab[i]) )
        else:
            heap.add( (l, utils.vocab[i]) )

    iters = 0
    while iters < 50:

        for curr_l, curr_s in heap.elements:
            likelihoods = model.predict_next(curr_s)

            for i, l in enumerate(likelihoods):
                new_s = curr_s + utils.vocab[i]
                if average_log_likelihood:
                    new_l = log_likelihood(model, new_s) / len(new_s)
                else:
                    new_l = curr_l + l

                if new_s not in visited:
                    visited.add(new_s)
                    # add to terminated heap
                    if new_s[-1] == '.' or len(new_s) > max_length:
                        term_heap.add( (new_l, new_s) )
                    else:
                        heap.add( (new_l, new_s) )
        iters += 1

    # sort and return
    sort_list = []
    for i in range(len(term_heap.elements)):
        sort_list.append(term_heap.elements[i])
    #sort_list.sort(reverse=True)
    sort_list.sort()

    return_list = []
    for l, s in sort_list:
        return_list.append(s)
    print(return_list)
    return return_list


if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
