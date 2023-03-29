"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
"""

from collections import Counter, defaultdict
from math import log
from numpy import argmax

# define your epsilon for laplace smoothing here


def baseline(train, test):
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
            word_tag_count[word][tag] += 1

    tag_count = Counter()

    for values in word_tag_count.values():
        tag_count.update(values)

    most_common_tag = tag_count.most_common(1)[0][0]

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


def viterbi_tmp(
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
    alpha: float = 1e-4

    # Count occurrences of tags, tag pairs, tag/word pairs.
    init_tag_count: Counter[str] = Counter()
    tag_count: Counter[str] = Counter()
    tag_pair_count: dict[str, Counter[str]] = defaultdict(Counter)
    tag_word_count: dict[str, Counter[str]] = defaultdict(Counter)

    for sentence in train:
        for i, (word, tag) in enumerate(sentence):
            if i == 0:
                init_tag_count[tag] += 1
            else:
                pass
            tag_count[tag] += 1
            if tag not in tag_word_count:
                tag_word_count[tag] = Counter()
            tag_word_count[tag][word] += 1

        for i in range(len(sentence) - 1):
            tag_pair_count[sentence[i][1]][sentence[i + 1][1]] += 1

    init_tag_prob: dict[str, float] = {}
    tag_pair_prob: dict[str, dict[str, float]] = defaultdict(dict)
    tag_word_prob: dict[str, dict[str, float]] = defaultdict(dict)

    for tag in tag_count.keys():
        for next_tag in tag_count.keys():
            tag_pair_prob[tag][next_tag] = log(
                (tag_pair_count[tag][next_tag] + alpha)
                / (sum(tag_pair_count[tag].values()) + alpha * len(tag_count.keys()))
            )

        tag_word_count[tag]["OOV"] = 0
        for word in tag_word_count[tag]:
            tag_word_prob[tag][word] = log(
                (tag_word_count[tag][word] + alpha)
                / (sum(tag_word_count[tag].values()) + alpha * len(tag_count.keys()))
            )

        init_tag_prob[tag] = log(
            (init_tag_count[tag] + alpha)
            / (sum(init_tag_count.values()) + alpha * len(tag_count.keys()))
        )

    output: list[list[tuple[str, str]]] = []

    for sentence in test:

        trellis: list[dict[str, tuple[float, str | None]]] = []

        init_dict: dict[str, tuple[float, str | None]] = {}
        for tag in tag_count.keys():
            if sentence[0] in tag_word_prob[tag]:
                init_dict[tag] = (
                    init_tag_prob[tag] + tag_word_prob[tag][sentence[0]],
                    None,
                )
            else:
                init_dict[tag] = (init_tag_prob[tag] + tag_word_prob[tag]["OOV"], None)

        trellis.append(init_dict)

        for i in range(1, len(sentence)):
            trellis.append({})
            word: str = sentence[i]
            for tag in tag_count.keys():
                v_t: float = float("-inf")
                psi_t: str | None = None
                for prev_tag in tag_count.keys():
                    v_prev: float = trellis[i - 1][prev_tag][0]
                    b: float = (
                        tag_word_prob[tag][sentence[i]]
                        if sentence[i] in tag_word_prob[tag]
                        else tag_word_prob[tag]["OOV"]
                    )
                    a: float = tag_pair_prob[prev_tag][tag]
                    v_tmp: float = v_prev + a + b
                    if v_tmp > v_t:
                        v_t = v_tmp
                        psi_t = prev_tag
                    else:
                        pass

                trellis[i][tag] = (v_t, psi_t)

        output_sentence: list[tuple[str, str]] = []
        max_prob: float = float("-inf")
        max_tag: str | None = None
        for tag in tag_count.keys():
            if trellis[-1][tag][0] > max_prob:
                max_prob = trellis[-1][tag][0]
                max_tag = tag
            else:
                pass

        node: tuple[float, str | None] = (max_prob, max_tag)
        for i in range(len(sentence) - 1, -1, -1):
            output_sentence.append((sentence[i], node[1]))
            node = trellis[i][node[1]]
        output.append(output_sentence[::-1])

    return output


def viterbi(train, test):
    alpha = 1e-4

    # Count occurrences of tags, tag pairs, tag/word pairs.
    init_tag_count = Counter()
    tag_count = Counter()
    tag_pair_count = {}
    tag_word_count = {}

    for sentence in train:
        for i in range(len(sentence)):
            word, tag = sentence[i]
            if i == 0:
                init_tag_count[tag] += 1
            tag_count[tag] += 1
            if tag not in tag_word_count:
                tag_word_count[tag] = {"OOV": 0}
            if word in tag_word_count[tag]:
                tag_word_count[tag][word] += 1
            else:
                tag_word_count[tag][word] = 1

            if i < len(sentence) - 1:
                prev_tag = sentence[i][1]
                next_tag = sentence[i + 1][1]
                if prev_tag not in tag_pair_count:
                    tag_pair_count[prev_tag] = Counter()
                tag_pair_count[prev_tag][next_tag] += 1

    init_tag_prob = {}
    tag_pair_prob = {}
    tag_word_prob = {}

    for tag in tag_count:
        tag_pair_prob[tag] = {}
        for next_tag in tag_count:
            tag_pair_prob[tag][next_tag] = log(
                (tag_pair_count.get(tag, Counter())[next_tag] + alpha)
                / (
                    sum(tag_pair_count.get(tag, Counter()).values())
                    + alpha * len(tag_count)
                )
            )

        for word in tag_word_count[tag]:
            tag_word_prob.setdefault(tag, {})[word] = log(
                (tag_word_count[tag][word] + alpha)
                / (sum(tag_word_count[tag].values()) + alpha * len(tag_count))
            )

        init_tag_prob[tag] = log(
            (init_tag_count[tag] + alpha)
            / (sum(init_tag_count.values()) + alpha * len(tag_count))
        )

    output = []

    for sentence in test:

        init_dict = {}
        for tag in tag_count:
            if sentence[0] in tag_word_prob.get(tag, {}):
                init_dict[tag] = init_tag_prob[tag] + tag_word_prob[tag][sentence[0]]

            else:
                init_dict[tag] = init_tag_prob[tag] + tag_word_prob[tag]["OOV"]

        trellis = init_dict

        traces = []

        for i in range(1, len(sentence)):
            new_trellis = {}
            word = sentence[i]
            trace = {}
            for tag in tag_count:
                v_t: float = float("-inf")
                psi_t: str | None = None
                for prev_tag in tag_count.keys():
                    v_tmp: float = (
                        trellis[prev_tag]
                        + tag_pair_prob[prev_tag][tag]
                        + (
                            tag_word_prob[tag][sentence[i]]
                            if sentence[i] in tag_word_prob[tag]
                            else tag_word_prob[tag]["OOV"]
                        )
                    )
                    if v_tmp > v_t:
                        v_t = v_tmp
                        psi_t = prev_tag
                    else:
                        pass

                trace[tag] = psi_t

                new_trellis[tag] = v_t
            traces.append(trace)
            trellis = new_trellis

        output_sentence: list[tuple[str, str]] = [("END", "END")]
        tag = "END"
        for i in range(len(sentence) - 2, -1, -1):
            output_sentence.append((sentence[i], traces[i][tag]))
            tag = traces[i][tag]

        output.append(output_sentence[::-1])

    return output


def viterbi_ec(train, test):
    """
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    raise NotImplementedError("You need to write this part!")
