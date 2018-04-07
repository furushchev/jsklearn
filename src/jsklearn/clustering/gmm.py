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
            n_clusters = [self.gmm_args.pop("n_components")]
        else:
            n_samples = x.shape[0]
            n_clusters = range(1, min(self.max_cluster_num, n_samples)+1)
        if "covariance_type" in self.gmm_args:
            c_types = [self.gmm_args.pop("covariance_type")]
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
            self.fit(x)
        self.cluster_sizes_ = self.best_gmm.means_.shape[0]
        labels = self.best_gmm.predict(x)
        cluster_size = np.empty(self.best_gmm.means_.shape[0], dtype=np.intp)
        for k in range(self.best_gmm.means_.shape[0]):
            cluster_size[k] = len(np.where(labels==k)[0])
        self.cluster_sizes_ = cluster_size
        self.weights_ = self.best_gmm.weights_
        return labels

    def fit_predict(self, x):
        return self.fit(x).predict(x)

    def score(self, x):
        if not self.best_gmm:
            raise RuntimeError("model not fit")
        return self.best_gmm.score(x)

    def score_samples(self, x):
        if not self.best_gmm:
            raise RuntimeError("model not fit")
        return self.best_gmm.score_samples(x)

    @property
    def covariances_(self):
        if not self.best_gmm:
            raise RuntimeError("model not fit")
        cov = self.best_gmm.covariances_
        cov_type = self.best_gmm.covariance_type
        n, dim = self.best_gmm.means_.shape
        if cov_type is "full":
            return cov
        elif cov_type is "tied":
            return np.array([cov for i in range(n)])
        elif cov_type is "diag":
            return np.array([np.diag(m) for m in cov])
        elif cov_type is "spherical":
            return np.array([np.eye(dim) * m for m in cov])

    def __getattr__(self, key):
        params = self.best_gmm.get_params(True)
        if key in params:
            return params[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError


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
    print("Covariance type: %s" % gmm.covariance_type)
    print("Covariance: %s" % gmm.covariances_)
    print("Weights: %s" % gmm.weights_)

    # plot data with predicted class
    plt.scatter(X[:, 0], X[:, 1], c=labels, marker='x', s=30)

    # plot center of clusters
    plt.scatter(gmm.cluster_centers_[:, 0], gmm.cluster_centers_[:, 1],
                c="r", marker="*", s=250)

    # plot ellipse of gaussian
    mx, my = np.meshgrid(np.linspace(*plt.xlim()), np.linspace(*plt.ylim()))
    mf = lambda x, y: np.exp(gmm.best_gmm.score_samples(np.array([[x, y]])))
    mz = np.vectorize(mf)(mx, my)
    plt.pcolor(mx, my, mz, alpha=0.3)
    plt.colorbar()

    plt.title("GMM clustering")
    plt.grid()
    plt.show()
