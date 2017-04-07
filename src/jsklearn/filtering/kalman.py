#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from __future__ import print_function
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
        z = np.zeros((meas_dim,))

        self.x = self.H.T.dot(z)

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
        return self.H.dot(self.x)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 20
    sigma = 0.2

    tx = np.linspace(0.0, 10.0, n)
    ty = tx ** 2
    t = np.dstack((tx, ty))[0]
    plt.plot(tx, ty, 'rs-', label="Groundtruth")

    ox = np.random.multivariate_normal(tx, np.eye(tx.shape[0]) * sigma, 1)[0]
    oy = np.random.multivariate_normal(ty, np.eye(ty.shape[0]) * sigma, 1)[0]
    o = np.dstack((ox, oy))[0]
    plt.plot(ox, oy, 'g^-', label="Measurement")

    # position - velocity
    kalman = Kalman(state_dim=4, meas_dim=2, R_var=1e-3)
    kalman.F = np.array([[1., 0., 1., 0.],
                         [0., 1., 0., 1.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])

    z = np.array(map(lambda x_t: kalman.filter(x_t), o))
    plt.plot(z[:,0], z[:,1], 'bo-', label="Estimation")

    plt.title("Kalman Filter")
    plt.xlim(-1, 11)
    plt.legend(loc=0)
    plt.show()
