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
        err = math.sqrt(S_det) * math.exp(0.5 * y.dot(S_inv).dot(y))
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

    n = 20
    sigma = 0.2

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

    z = []
    e = []
    for ot in o:
        z += [kalman.filter(ot)]
        e += [kalman.error()]
    z = np.array(z)
    e = np.array(e) * 10
    e_lower = z[:,1] - e
    e_upper = z[:,1] + e

    print(z[:,1])
    print(e_lower)
    print(e_upper)

    plt.subplot(2,1,1)  # upper
    plt.plot(tx, ty, 'rs-', label="Groundtruth")
    plt.plot(ox, oy, 'g^-', label="Measurement")
    plt.plot(z[:,0], z[:,1], 'bo-', label="Estimation")
    plt.fill_between(z[:,0], e_lower, e_upper, facecolor='blue', interpolate=True)
    plt.title("Kalman Filter (Position-Velocity)")
    plt.xlim(-1, 11)
    plt.legend(loc=0)


    # Adaptive Kalman Filter
    kalman = AdaptiveKalman(state_dim=4, meas_dim=2, R_var=1e-3)


    plt.subplot(2,1,2)  # bottom
    # TODO: plot
    plt.show()
