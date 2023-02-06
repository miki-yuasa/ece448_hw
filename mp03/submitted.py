'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np
from numpy import ndarray
from numpy import linalg as LA


def k_nearest_neighbors(image: ndarray, train_images: ndarray, train_labels: ndarray, k: int) -> tuple[ndarray, ndarray]:
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''

    distance: ndarray = train_images - image
    distance_norm: ndarray = LA.norm(distance, axis=1)

    idx: ndarray = np.argpartition(distance_norm, k)

    neighbors: ndarray = train_images[idx[:k]]
    lablels: ndarray = train_labels[idx[:k]]

    return neighbors, lablels


def classify_devset(dev_images: ndarray, train_images: ndarray, train_labels: ndarray, k: int) -> tuple[list[int], list[int]]:
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''

    hypotheses: list[int] = []
    scores: list[int] = []

    for image in dev_images:
        labels: ndarray
        _, labels = k_nearest_neighbors(
            image, train_images, train_labels, k)

        unique: ndarray
        counts: ndarray
        unique, counts = np.unique(labels, return_counts=True)
        hypothesis_indx: list[int] = [
            i for i, x in enumerate(counts) if x == max(counts)]

        hypothesis: bool = unique[np.argmax(counts)] if len(
            hypothesis_indx) == 1 else False

        hypotheses.append(int(hypothesis))
        scores.append(np.max(counts))

    return hypotheses, scores


def confusion_matrix(hypotheses: list[int], references: ndarray) -> tuple[ndarray, float, float]:
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

    for hypo, ref in zip(hypotheses, references.tolist()):
        if hypo == 0 and ref == 0:
            tn += 1
        elif hypo == 1 and ref == 0:
            fp += 1
        elif hypo == 0 and ref == 1:
            fn += 1
        else:
            tp += 1

    confusions: ndarray = np.array([[tn, fp], [fn, tp]])
    precision: float = tp/(tp+fp)
    recall: float = tp/(tp+fn)
    accuracy: float = (tp+tn)/(tp+tn+fp+fn)
    f1 = 2/(1/recall+1/precision)

    return confusions, accuracy, f1
