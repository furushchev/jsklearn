#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from __future__ import division
from __future__ import print_function
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans


class _Cluster:
    @classmethod
    def create(cls, x, kmeans, index=None):
        if index is None:
            index = np.array(range(0, x.shape[0]))
        labels = range(0, kmeans.get_params()["n_clusters"])
        return (cls(x, index, kmeans, label) for label in labels)

    def __init__(self, x, index, kmeans, label):
        """
        Args:
          x (array): input data at index row from entire data
          index(1-d int vector): row numbers of data
          kmeans (sklearn.cluster.KMeans object): clustering method object
          label(int): label
        """
        self.data = x[kmeans.labels_ == label]
        self.index = index[kmeans.labels_ == label]
        self.size = self.data.shape[0]
        self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
        self.center = kmeans.cluster_centers_[label]
        try:
            self.cov = np.cov(self.data.T)
        except:
            self.cov = None

    def log_likelihood(self):
        try:
            return sum(stats.multivariate_normal.logpdf(x, self.center, self.cov) for x in self.data)
        except:
            return None

    def bic(self):
        return -2 * self.log_likelihood() + self.df * np.log(self.size)

class XMeans(object):
    """
    A clustering method extended from X-means
    http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf
    """

    def __init__(self, init_cluster_num=2, **kmeans_args):
        """
        Args:
          init_cluster_num (int, default: 2): initial cluster num

          Other arguments are passed to sklearn.cluster.KMeans
        """
        assert init_cluster_num >= 2
        self.init_cluster_num = init_cluster_num
        self.kmeans_args = kmeans_args

        self.clusters_ = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.cluster_log_likelihoods_ = None
        self.cluster_sizes_ = None

    def fit(self, x):
        """
        Args:
          x: array or sparse matrix whose shape is (n_samples, n_features)
        """
        self._clusters = []

        all_clusters = _Cluster.create(x, KMeans(self.init_cluster_num, **self.kmeans_args).fit(x))
        clusters = []
        orphans = []
        for cluster in all_clusters:
            if cluster.cov is None:
                orphans.append(cluster)
            else:
                clusters.append(cluster)
        self._cluster_recursive(clusters)

        self.labels_ = np.empty(x.shape[0], dtype = np.intp)
        self._clusters += orphans
        for i, c in enumerate(self._clusters):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self._clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self._clusters])
        self.cluster_sizes_ = np.array([c.size for c in self._clusters])

        return self

    def _cluster_recursive(self, clusters):
        """
        Execute cluster recursively
        clusters (iterable): list of instances of '_Cluster'
        """
        for cluster in clusters:
            if cluster.size <= 3:
                self._clusters.append(cluster)
                continue

            kmeans = KMeans(2, **self.kmeans_args).fit(cluster.data)
            c1, c2 = _Cluster.create(cluster.data, kmeans, cluster.index)
            try:
                beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            except:
                beta = 0.0
            alpha = 0.5 / stats.norm.cdf(beta)
            try:
                bic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df * np.log(cluster.size)
            except:
                bic = float("inf")

            if bic < cluster.bic():
                self._cluster_recursive([c1, c2])
            else:
                self._clusters.append(cluster)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from jsklearn.dataset.scatter import get_scatter_2d

    data = get_scatter_2d()
    x = data[:, 0]
    y = data[:, 1]
    n = x.shape[0]

    if len(x.shape) == 1:
        xdim = 1
    else:
        xdim = x.shape[1]
    print("Dataset: %d samples with %d dimensions" % (n, xdim))

    print("Computing KMeans Clustering...")
    xmeans = XMeans(random_state = 123).fit(np.c_[x,y])
    print("Done.")

    print("Optimal Cluster Num: %d" % len(xmeans.cluster_sizes_))
    for i, s in enumerate(xmeans.cluster_sizes_):
        c = xmeans.cluster_centers_[i]
        print(" * Cluster #%d: %d samples (center: %s)" % (i, s, c))

    plt.scatter(x, y, c=xmeans.labels_, marker='x', s=30)
    plt.scatter(xmeans.cluster_centers_[:,0], xmeans.cluster_centers_[:,1], c="r", marker="*", s=250)
    plt.title("XMeans clustering")
    plt.grid()
    plt.show()
