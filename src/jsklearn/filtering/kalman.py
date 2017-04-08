#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from __future__ import print_function
import math
import numpy as np


class Kalman(object):
    def __init__(self, state_dim=2, meas_dim=2, P_var=1., Q_var=1e-5, R_var=1e-2):
        self.B = np.eye(state_dim)
        self.F = np.eye(state_dim)
        self.H = np.eye(meas_dim, state_dim)
        self.P = np.eye(state_dim) * P_var
        self.Q = np.eye(state_dim) * Q_var
        self.R = np.eye(meas_dim)  * R_var
        self.S = np.eye(state_dim)

        self.u = np.zeros((state_dim,))
        self.z = np.zeros((meas_dim,))
        self.x = self.H.T.dot(self.z)

        self.state_dim = state_dim
        self.meas_dim = meas_dim

    def predict(self):
        self.x = self.F.dot(self.x) + self.B.dot(self.u)
        self.P = self.F.dot(self.P.dot(self.F.T)) + self.Q

    def update(self, z):
        """
        Args:
          z (value or array-like): measurement
        """
        self.z = z  # for evaluate error

        # y = z - Hx
        y = z - self.H.dot(self.x)

        # S = H P H^ + R
        self.S = self.H.dot(self.P.dot(self.H.T)) + self.R
        S_inv = np.linalg.inv(self.S)

        # K = P H^ S-1
        self.K = self.P.dot(self.H.T.dot(S_inv))

        # x = x + K y
        self.x = self.x + self.K.dot(y)

        I = np.eye(self.state_dim)
        # P = (I - K H) P
        self.P = (I - self.K.dot(self.H)).dot(self.P)

    def filter(self, z):
        self.predict()
        self.update(z)
        x = self.H.dot(self.x)
        return x

    @property
    def error(self):
        y = self.z - self.H.dot(self.x)
        S_det = np.linalg.det(self.S)
        S_inv = np.linalg.inv(self.S)
        try:
            err = math.sqrt(S_det) * math.exp(0.5 * y.dot(S_inv).dot(y))
        except OverflowError:
            err = float("inf")
        return err


class AdaptiveKalman(Kalman):
    def __init__(self, **kwargs):
        super(AdaptiveKalman, self).__init__(**kwargs)
        self.ys = []

    def adapt(self, z, m=15):
        y = z - self.H.dot(self.x)
        if len(self.ys) < m:
            self.ys += [y]
        else:
            self.ys = self.ys[1:] + [y]
            cov = np.zeros(self.R.shape)
            for y in self.ys:
                cov += np.outer(y, y)
            cov *= 1.0 / m
            self.R = cov + self.H.dot(self.P.dot(self.H.T))

    def filter(self, z):
        self.predict()
        self.update(z)
        self.adapt(z)
        return self.H.dot(self.x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Dataset Preparation
    n = 20
    sigma = 10

    tx = np.linspace(0.0, 10.0, n)
    ty = tx ** 2
    t = np.dstack((tx, ty))[0]

    ox = tx
    oy = np.random.multivariate_normal(ty, np.eye(ty.shape[0]) * sigma, 1)[0]
    o = np.dstack((ox, oy))[0]

    # Kalman Filter ( position - velocity )
    kalman = Kalman(state_dim=4, meas_dim=2, R_var=1e-3)
    kalman.F = np.array([[1., 0., 1., 0.],
                         [0., 1., 0., 1.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])
    result = np.array(map(lambda z: [kalman.filter(z), kalman.error, kalman.P[1,1]], o))
    z = np.vstack(result[:, 0]).astype(np.float)
    err = result[:, 1]
    cov = result[:, 2]

    fig, (top, bottom) = plt.subplots(nrows=2)

    top.set_title("Kalman Filter (Position-Velocity)")
    top.set_xlim(0, 10)
    top.plot(tx, ty, 'rs-', label="Groundtruth")
    top.plot(ox, oy, 'g^-', label="Measurement")
    top.plot(z[:,0], z[:,1], 'bo-', label="Estimation")
    top.legend(loc=0)
    top.grid()
    splt = top.twinx()
    splt.set_xlim(0, 10)
    splt.set_ylim(0, 4e-3)
    splt.plot(tx, cov, 'y*-', label="P Covariance")
    splt.legend(loc=0)

    # Adaptive Kalman Filter
    kalman = AdaptiveKalman(state_dim=4, meas_dim=2, R_var=1e-3)
    kalman.F = np.array([[1., 0., 1., 0.],
                         [0., 1., 0., 1.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])
    result = np.array(map(lambda z: [kalman.filter(z), kalman.error, kalman.P[1,1]], o))
    z = np.vstack(result[:, 0]).astype(np.float)
    err = result[:, 1]
    cov = result[:, 2]

    bottom.set_title("Adaptive Kalman Filter (Position-Velocity)")
    bottom.set_xlim(0, 10)
    bottom.plot(tx, ty, 'rs-', label="Groundtruth")
    bottom.plot(ox, oy, 'g^-', label="Measurement")
    bottom.plot(z[:, 0], z[:, 1], 'bo-', label="Estimation")
    bottom.legend(loc=2)
    bottom.grid()
    splt = bottom.twinx()
    splt.set_xlim(0, 10)
    splt.set_ylim(0, 4e-3)
    splt.plot(tx, cov, 'y*-', label="P Covariance")
    splt.legend(loc=0)

    fig.tight_layout()
    plt.show()
