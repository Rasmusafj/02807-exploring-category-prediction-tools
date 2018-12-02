"""
Utility
"""
import itertools
import mmh3
import numpy as np

from datasketch import MinHash
from timeit import default_timer as timer
from collections import Counter

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def signature_permutation(S, k, single=False):
    m = MinHash(num_perm=k)
    for element in S:
        if not single:
            element = " ".join(element)

        m.update(element.encode("utf-8"))

    return m.hashvalues


def jaccard_distance(S, T):
    S = set(S)
    T = set(T)
    return len(S & T) / len(S | T)


def signature(S, k, single=False):
    sig = []

    # Append large ints to signatures
    # These will (hopefully) be overwritten
    for i in range(0, k):
        sig.append(2 ** 64)

    # For each element in our set S, we compute the hash
    # If this hash is smaller than the current min value
    # then we replace it
    for element in S:
        for i in range(0, k):
            # i is also the random seed of our hash function
            if not single:
                element = " ".join(element)

            curhash = mmh3.hash(element, i)
            if curhash < sig[i]:
                sig[i] = curhash
    return sig


def shingle_construction(document, n):
    shingles_in_lists = [document[i:i + n] for i in range(len(document) - n + 1)]
    return set(tuple(shingle) for shingle in shingles_in_lists)


def construct_set_similarities(data, k, n, method="hash"):
    # Signatures
    signatures = []
    # create shingles
    for document in data:
        if method == "hash":
            if n == 1:
                signatures.append(signature(set(document), k, single=True))
            else:
                shingles = shingle_construction(document, n)
                signatures.append(signature(shingles, k))

        if method == "permutation":
            if n == 1:
                signatures.append(signature_permutation(set(document), k, single=True))
            else:
                shingles = shingle_construction(document, n)
                signatures.append(signature_permutation(shingles, k))

    return signatures


def construct_token_frequencies(data, vectorizer, normalize=False):

    final_vectors = []

    for document in data:
        counter_dict = dict(Counter(document))
        vector = vectorizer.transform([counter_dict]).todense()
        vector = np.squeeze(np.asarray(vector))

        if normalize:
            vector /= len(document)

        final_vectors.append(vector)

    return final_vectors


class CustomTimer(object):

    def start(self):
        self.start_time = timer()

    def stop_timer_and_get_result(self):
        stop_time = timer()
        return stop_time - self.start_time


# See documentation on tracemalloc
def total_allocated_memory(snapshot, key_type='lineno'):
    top_stats = snapshot.statistics(key_type)
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          classes_x=None,
                          model="LSH",
                          title='Confusion matrix for LSH',
                          cmap=plt.cm.Blues):
    """
    Implementation taken from and modified:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(13,13))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))

    if classes_x:
        plt.xticks(np.arange(len(classes_x)), classes_x, rotation=90)
    else:
        plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("./figs/" + model + ".pdf")
    plt.show()
