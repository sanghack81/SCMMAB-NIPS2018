import numpy as np

""" Drawing some plots for MAB or etc. """


def sparse_index(length, base_size=100):
    if length <= 2 * base_size:
        return np.arange(length)
    step = length // base_size  # >= 2
    if length % step == 0:
        temp = np.arange(1 + (length // step)) * step  # include length
        temp[-1] = length - 1
        return temp
    else:
        if (length // step) * step == length - 1:
            return np.arange(1 + (length // step)) * step
        else:
            temp = np.arange(2 + (length // step)) * step
            assert temp[-2] < length - 1
            temp[-1] = length - 1
            return temp
