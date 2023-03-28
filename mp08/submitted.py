"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
"""

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here


def baseline(
    train: list[list[tuple[str, str]]], test: list[list[str]]
) -> list[list[tuple[str, str]]]:
    """
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """

    # Start training

    word_tag_count = defaultdict(Counter)

    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag_count:
                word_tag_count[word] = Counter()
            word_tag_count[word][tag] += 1

    tag_count: Counter[str] = Counter()

    for values in word_tag_count.values():
        for key, value in values.items():
            tag_count[key] += value

    most_common_tag: str = tag_count.most_common(1)[0][0]

    # Start testing

    output: list[list[tuple[str, str]]] = []

    for sentence in test:
        output_sentence = []
        for word in sentence:
            if word in word_tag_count:
                tag = word_tag_count[word].most_common(1)[0][0]
            else:
                tag = most_common_tag
            output_sentence.append((word, tag))
        output.append(output_sentence)

    return output


def viterbi(train, test):
    """
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    raise NotImplementedError("You need to write this part!")


def viterbi_ec(train, test):
    """
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    raise NotImplementedError("You need to write this part!")
