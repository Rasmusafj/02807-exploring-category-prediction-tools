"""
Utility
"""
import mmh3

from datasketch import MinHash
from timeit import default_timer as timer

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
