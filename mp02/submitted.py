"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
"""

import copy
from typing import Literal
import numpy as np
from collections import Counter
from itertools import chain

stopwords = set(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "'t",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "couldn",
        "did",
        "didn",
        "do",
        "does",
        "doesn",
        "doing",
        "don",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "has",
        "hasn",
        "have",
        "haven",
        "having",
        "he",
        "he",
        "'d",
        "he",
        "'ll",
        "he",
        "'s",
        "her",
        "here",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how",
        "i",
        "'m",
        "'ve",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "it",
        "its",
        "itself",
        "let",
        "'s",
        "me",
        "more",
        "most",
        "mustn",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan",
        "she",
        "she",
        "'d",
        "she",
        "ll",
        "she",
        "should",
        "shouldn",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there",
        "these",
        "they",
        "they",
        "they",
        "they",
        "'re",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn",
        "we",
        "we",
        "we",
        "we",
        "we",
        "'ve",
        "were",
        "weren",
        "what",
        "what",
        "when",
        "when",
        "where",
        "where",
        "which",
        "while",
        "who",
        "who",
        "whom",
        "why",
        "why",
        "with",
        "won",
        "would",
        "wouldn",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)


def create_frequency_table(
    train: dict[Literal["pos", "neg"], list[list[str]]]
) -> dict[Literal["pos", "neg"], Counter[str, int]]:
    """
    Parameters:
    train (dict of list of lists)
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters)
        - frequency[y][x] = number of tokens of word x in texts of class y
    """
    neg_counter = Counter(chain.from_iterable(train["neg"]))
    pos_counter = Counter(chain.from_iterable(train["pos"]))

    frequency: dict[Literal["pos", "neg"], Counter[str]] = {
        "neg": neg_counter,
        "pos": pos_counter,
    }
    return frequency


def remove_stopwords(
    frequency: dict[Literal["pos", "neg"], Counter[str]]
) -> dict[Literal["pos", "neg"], Counter[str]]:
    """
    Parameters:
    frequency (dict of Counters)
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters)
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    """
    nonstop: dict[Literal["pos", "neg"], Counter[str]] = copy.deepcopy(frequency)

    for word in stopwords:
        try:
            del nonstop["pos"][word]
        except KeyError:
            pass

        try:
            del nonstop["neg"][word]
        except KeyError:
            pass

    return nonstop


def laplace_smoothing(
    nonstop: dict[Literal["pos", "neg"], Counter[str]], smoothness: float
) -> dict[Literal["pos", "neg"], dict[str, float]]:
    """
    Parameters:
    nonstop (dict of Counters)
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts)
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    """

    num_tokens_pos: int = sum(nonstop["pos"].values())
    num_wordtypes_pos: int = len(nonstop["pos"])
    num_tokens_neg: int = sum(nonstop["neg"].values())
    num_wordtypes_neg: int = len(nonstop["neg"])

    likelihood_pos: dict[str, float] = {
        word: (count + smoothness)
        / (num_tokens_pos + smoothness * (num_wordtypes_pos + 1))
        for word, count in nonstop["pos"].items()
    }
    likelihood_pos["OOV"] = smoothness / (
        num_tokens_pos + smoothness * (num_wordtypes_pos + 1)
    )
    likelihood_neg: dict[str, float] = {
        word: (count + smoothness)
        / (num_tokens_neg + smoothness * (num_wordtypes_neg + 1))
        for word, count in nonstop["neg"].items()
    }
    likelihood_neg["OOV"] = smoothness / (
        num_tokens_neg + smoothness * (num_wordtypes_neg + 1)
    )

    likelihood: dict[Literal["pos", "neg"], dict[str, float]] = {
        "pos": likelihood_pos,
        "neg": likelihood_neg,
    }

    return likelihood


def naive_bayes(
    texts: list[list[str]],
    likelihood: dict[Literal["pos", "neg"], dict[str, float]],
    prior: float,
) -> list[Literal["pos", "neg", "undecided"]]:
    """
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts)
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    """

    hypotheses: list[Literal["pos", "neg", "undecided"]] = []

    for text in texts:
        likelihood_probs: list[float] = []
        for word in text:
            if word in stopwords:
                pass
            else:
                likelihood_pos: float
                likelihood_neg: float
                try:
                    likelihood_pos = likelihood["pos"][word]
                except KeyError:
                    likelihood_pos = likelihood["pos"]["OOV"]

                try:
                    likelihood_neg = likelihood["neg"][word]
                except KeyError:
                    likelihood_neg = likelihood["neg"]["OOV"]

                likelihood_probs.append(likelihood_pos / likelihood_neg)

        probability: float = np.log(prior / (1 - prior)) + np.sum(
            np.log(np.array(likelihood_probs))
        )

        if probability > 0:
            hypotheses.append("pos")
        elif probability < 0:
            hypotheses.append("neg")
        else:
            hypotheses.append("undecided")

    return hypotheses


def optimize_hyperparameters(
    texts: list[list[str]],
    labels: list[Literal["pos", "neg"]],
    nonstop: dict[Literal["pos", "neg"], Counter[str]],
    priors: list[float],
    smoothnesses: list[float],
) -> np.ndarray:
    """
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters)
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    """

    accuracies: np.ndarray = np.empty([len(priors), len(smoothnesses)])

    num_labels: int = len(labels)

    for i, prior in enumerate(priors):
        for j, smoothness in enumerate(smoothnesses):
            likelihood: dict[
                Literal["pos", "neg"], dict[str, float]
            ] = laplace_smoothing(nonstop, smoothness)
            hypotheses: list[Literal["pos", "neg", "undecided"]] = naive_bayes(
                texts, likelihood, prior
            )
            count_correct: int = 0
            for (y, yhat) in zip(labels, hypotheses):
                if y == yhat:
                    count_correct += 1

            accuracies[i, j] = count_correct / num_labels

    return accuracies
