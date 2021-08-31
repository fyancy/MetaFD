import numpy as np


def one_hot_encode(x, dim):
    res = np.zeros(np.shape(x) + (dim,), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


def one_hot_decode(x):
    return np.argmax(x, axis=-1)
