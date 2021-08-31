from sklearn.preprocessing import StandardScaler, normalize, maxabs_scale
import numpy as np
import random
import os
import tensorflow as tf

import torch
import torch.nn as nn
from torch.backends import cudnn


def my_normalization(x):  # x: (n, length)
    # method 1:
    # x = x - np.mean(x, axis=1, keepdims=True)
    # x = maxabs_scale(x, axis=1)  # 效果也很好, 2
    # method 2:
    x = normalize(x, norm='l2', axis=1)

    return np.asarray(x)


def one_hot_encode(x, dim):
    """

    :param x: shape, 2 or 3 dimensions.
    :param dim:
    :return:
    """
    res = np.zeros(np.shape(x) + (dim,), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


def sample_label_shuffle(data, label):
    """
    要求input是二维数组array
    :param data: [num, data_len, C]
    :param label: [num]
    :return:
    """
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label


def seed_tensorflow(seed=2021):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print("Set seed for tensorflow\n")


def seed_torch(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"set random seed: {seed}")
    # 下面两项比较重要，搭配使用。以精度换取速度，保证过程可复现。
    # https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = False  # False: 表示禁用
    cudnn.deterministic = True  # True: 每次返回的卷积算法将是确定的，即默认算法。
    # cudnn.enabled = True


def weights_init2(L):
    if isinstance(L, nn.Conv1d):
        n = L.kernel_size[0] * L.out_channels
        L.weight.data.normal_(mean=0, std=np.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm1d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)
        # print(L.bias.data)
    elif isinstance(L, nn.Linear):
        L.weight.data.normal_(0, 0.01)
        if L.bias is not None:
            L.bias.data.fill_(1)


if __name__ == "__main__":
    y = np.random.randint(0, 7, [5, 10])
    y_h = one_hot_encode(y, 7)
    print(y[0], y_h[0])

    y = np.random.randint(0, 7, 10)
    y_h = one_hot_encode(y, 7)
    print(y, y_h)