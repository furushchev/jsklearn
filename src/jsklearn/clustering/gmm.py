#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import numpy as np
from sklearn.mixture import GaussianMixture


class GMM(object):
    def __init__(self, max_cluster_num=10, **gmm_args):
        """
        Args:
          max_cluster_num (int, default: 10): maximum number of clusters

          Other arguments are passed to sklearn.mixture.GaussianMixture
        """

        assert isinstance(max_cluster_num, int) and max_cluster_num > 0
        self.max_cluster_num = max_cluster_num
        self.gmm_args = gmm_args
        self.best_gmm = None

    def fit(self, x):
        """
        Args:
          x: array or sparse matrix whose shape is (n_samples, n_features)
        """
        if "n_components" in self.gmm_args:
            n_clusters = self.gmm_args.pop("n_components")
        else:
            n_clusters = range(1, self.max_cluster_num+1)
        if "covariance_type" in self.gmm_args:
            c_types = self.gmm_args.pop("covariance_type")
        else:
            c_types = ["spherical", "tied", "diag", "full"]
        min_bic = np.inf
        best_gmm = None
        for c_type in c_types:
            for n_cluster in n_clusters:
                gmm = GaussianMixture(n_components=n_cluster,
                                      covariance_type=c_type,
                                      **self.gmm_args)
                gmm.fit(x)
                bic = gmm.bic(x)
                if bic < min_bic:
                    min_bic = bic
                    best_gmm = gmm

        self.best_gmm = best_gmm
        self.cluster_centers_ = self.best_gmm.means_

        return self

    def predict(self, x):
        if not self.best_gmm:
            raise RuntimeError("Model is not yet fit")
        self.cluster_sizes_ = self.best_gmm.means_.shape[0]
        labels = self.best_gmm.predict(x)
        cluster_size = np.empty(self.best_gmm.means_.shape[0], dtype=np.intp)
        for k in range(self.best_gmm.means_.shape[0]):
            cluster_size[k] = len(np.where(labels==k)[0])
        self.cluster_sizes_ = cluster_size
        return labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from jsklearn.dataset import get_scatter_2d

    random_state = 123

    np.random.seed(random_state)

    data = get_scatter_2d()
    X = data[:, :-1]
    t = data[:, -1]
    n, xdim = X.shape

    print("Dataset: %d samples with %d dimensions" % (n, xdim))

    print("Computing GMM Clustering...")
    gmm = GMM(random_state=random_state).fit(X)
    labels = gmm.predict(X)
    print("Done.")

    print("Optimal Cluster Num: %d" % len(gmm.cluster_sizes_))
    for i, s in enumerate(gmm.cluster_sizes_):
        c = gmm.cluster_centers_[i]
        print(" * Cluster #%d: %d samples (center: %s)" % (i, s, c))

    plt.scatter(X[:, 0], X[:, 1], c=labels, marker='x', s=30)
    plt.scatter(gmm.cluster_centers_[:, 0], gmm.cluster_centers_[:, 1],
                c="r", marker="*", s=250)
    plt.title("GMM clustering")
    plt.grid()
    plt.show()
