#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import numpy as np
from jsklearn.util import get_data_path

def get_faithful():
    return np.genfromtxt(get_data_path("common/faithful.txt"))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = get_faithful()

    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)

    for i in range(data.shape[1]):
        data[:,i] = (data[:,i] - mu[i]) / sigma[i]

    plt.plot(data[:,0], data[:,1], 'go')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.grid()
    plt.show()
