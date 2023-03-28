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

    word_tag_count: dict[str, Counter[str]] = defaultdict(Counter)

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
            tag: str
            if word in word_tag_count:
                tag = word_tag_count[word].most_common(1)[0][0]
            else:
                tag = most_common_tag
            output_sentence.append((word, tag))
        output.append(output_sentence)

    return output


def viterbi(
    train: list[list[tuple[str, str]]], test: list[list[str]]
) -> list[list[tuple[str, str]]]:
    """
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """

    # Define the smoothing parameter
    alpha: float = 1

    # Count occurrences of tags, tag pairs, tag/word pairs.
    tag_count: Counter[str] = Counter()
    tag_pair_count: dict[str, Counter[str]] = defaultdict(Counter)
    tag_word_count: dict[str, Counter[str]] = defaultdict(Counter)

    for sentence in train:
        for word, tag in sentence:
            tag_count[tag] += 1
            if tag not in tag_word_count:
                tag_word_count[tag] = Counter()
            tag_word_count[tag][word] += 1

        for i in range(len(sentence) - 1):
            tag_pair_count[sentence[i][1]][sentence[i + 1][1]] += 1

    # Compute smoothed probabilities
    # Take the log of each probability

    all_tags: list[str] = list(tag_count.keys())
    tag_pair_prob: dict[str, dict[str, float]] = defaultdict(dict)
    tag_word_prob: dict[str, dict[str, float]] = defaultdict(dict)

    for tag in all_tags:
        for next_tag in all_tags:
            tag_pair_prob[tag][next_tag] = math.log(
                (tag_pair_count[tag][next_tag] + alpha)
                / (sum(tag_pair_count[tag].values()) + alpha * len(all_tags))
            )
        for word in tag_word_count[tag]:
            tag_word_prob[tag][word] = math.log(
                (tag_word_count[tag][word] + alpha)
                / (sum(tag_word_count[tag].values()) + alpha * len(all_tags))
            )
    pass
    # Construct the trellis.   Notice that for each tag/time pair, you must store not only the probability of the best path but also a pointer to the previous tag/time pair in that path.
    # The trellis is a list of dictionaries, one for each time step.  Each dictionary maps a tag to a tuple (probability, previous_tag).  The previous_tag is None for the first time step.
    trellis: list[dict[str, tuple[float, str | None]]] = []
    # Initialize the trellis with the probabilities of the first word.
    init_dict: dict[str, tuple[float, str | None]] = {
        tag: (0, None) for tag in all_tags
    }
    init_dict["START"] = (log(0), None)
    trellis.append(init_dict)
    # For each time step, compute the probabilities of all possible paths through the trellis ending in that time step.
    for i in range(len(test)):
        trellis.append({tag: (0, None) for tag in all_tags})
        for tag in all_tags:
            for prev_tag in all_tags:
                # Compute the probability of the path ending in this tag/time pair.
                # Store the probability and the previous tag/time pair in the trellis.
                pass
    # For each time step, find the tag that gives the highest probability of the best path through the trellis.

    # Return the best path through the trellis.


def viterbi_ec(train, test):
    """
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    raise NotImplementedError("You need to write this part!")
