"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
"""

from typing import Callable, Literal
import numpy as np
from numpy import ndarray
import collections
import itertools


def joint_distribution_of_word_counts(
    texts: list[list[str]], word0: str, word1: str
) -> np.ndarray:
    """
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    """

    num_word_pairs: list[tuple[int, int]] = []

    nums_word0: list[int] = []
    nums_word1: list[int] = []

    for text in texts:
        num_word0: int = text.count(word0)
        num_word1: int = text.count(word1)

        num_word_pairs.append((num_word0, num_word1))
        nums_word0.append(num_word0)
        nums_word1.append(num_word1)

    num_word_pair_counter = collections.Counter(num_word_pairs)

    most_common_pairs: list[
        tuple[tuple[int, int], int]
    ] = num_word_pair_counter.most_common()

    Pjoint_tmp: np.ndarray = np.zeros([max(nums_word0) + 1, max(nums_word1) + 1])

    for pair, count in most_common_pairs:
        Pjoint_tmp[*pair] = count

    Pjoint: np.ndarray = Pjoint_tmp / Pjoint_tmp.sum()

    return Pjoint


def marginal_distribution_of_word_counts(
    Pjoint: ndarray, index: Literal[0, 1]
) -> ndarray:
    """
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other)

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    """
    Pmarginal: np.ndarray = (
        np.sum(Pjoint, axis=1) if index == 0 else np.sum(Pjoint, axis=0)
    )
    return Pmarginal


def conditional_distribution_of_word_counts(
    Pjoint: ndarray, Pmarginal: ndarray
) -> ndarray:
    """
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs:
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    """
    Pcond: ndarray = (Pjoint.transpose() / Pmarginal).transpose()
    return Pcond


def mean_from_distribution(P: ndarray) -> float:
    """
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    mu (float) - the mean of X
    """
    mu: float = P @ np.array(range(P.size))
    return mu


def variance_from_distribution(P: ndarray) -> float:
    """
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    var (float) - the variance of X
    """
    mu: float = P @ np.array(range(P.size))

    var: ndarray = (np.array(range(P.size)) - mu) ** 2 @ P

    return var


def covariance_from_distribution(P: ndarray) -> float:
    """
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)

    Outputs:
    covar (float) - the covariance of X0 and X1
    """

    combination_products: ndarray = np.array(range(P.shape[0])).reshape(
        -1, 1
    ) @ np.array(range(P.shape[1])).reshape(1, -1)

    P0: ndarray = marginal_distribution_of_word_counts(P, 0)
    P1: ndarray = marginal_distribution_of_word_counts(P, 1)

    covar: float = np.sum(P * combination_products) - mean_from_distribution(
        P0
    ) * mean_from_distribution(P1)

    return covar


def expectation_of_a_function(P: ndarray, f: Callable) -> float:
    """
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    """

    num_X0: int
    num_X1: int

    num_X0, num_X1 = P.shape

    X0: list[int] = list(range(num_X0))
    X1: list[int] = list(range(num_X1))

    F: ndarray = np.zeros(P.shape)

    for x0, x1 in itertools.product(X0, X1):
        F[x0, x1] = f(x0, x1)

    expected: float = np.sum(F * P)

    return expected
