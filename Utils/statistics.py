import numpy as np


def pooled_mean(means, counts):
    return np.dot(means, counts) / sum(counts)


def pooled_std(stds, counts):
    return np.sqrt(sum(np.array([(n - 1) * s ** 2 for s, n in zip(stds, counts)])) / (sum(counts) - len(counts)))
