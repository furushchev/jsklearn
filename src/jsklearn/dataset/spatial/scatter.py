#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import math
import numpy as np


def get_scatter_2d(num_instance=20, num_class=4, sigma=0.2):
    """
    Generate scatter points which is separatable to some classes.
    Number of points are num_instance * num_class
    """
    step = np.arange(0, math.pi * 2., math.pi * 2. / num_class)[:num_class]
    centers = [(math.sin(th), math.cos(th)) for th in step]
    x = np.array([np.random.normal(loc[0], sigma, num_instance) for loc in centers]).flatten()
    y = np.array([np.random.normal(loc[1], sigma, num_instance) for loc in centers]).flatten()
    l = np.repeat(range(num_class), num_instance)

    return np.dstack((x, y, l))[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(123)
    data = get_scatter_2d()

    x = data[:, 0]
    y = data[:, 1]
    labels = data[:, 2]

    plt.scatter(x, y, c=labels, marker='x', s=30)
    plt.title("Scatter Points 2D")
    plt.grid()
    plt.show()
