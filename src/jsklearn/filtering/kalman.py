#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from __future__ import print_function
import numpy as np


class Kalman(object):
    def __init__(self, state_dim=2, meas_dim=2, P_var=1e-5, Q_var=1e-5, R_var=1e-2):
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
    t = np.linspace(0, 1, n)
    x_t = np.array([0.0, 0.0])
    y_t = x_t
    u_t = np.array([0.2, 0.2])
    x = [x_t]
    y = [y_t]
    u = np.repeat(u_t, n).reshape((n, 2))
    for i in range(n):
        x_t = x_t + u_t + np.random.multivariate_normal([0., 0.], np.eye(2) * 0.1, 1)[0]
        y_t = x_t + np.random.multivariate_normal([0., 0.], np.eye(2) * 0.1, 1)[0]
        x.append(x_t)
        y.append(y_t)

    x = np.array(x)  # states
    y = np.array(y)  # observations

    kalman = Kalman()

    z = np.array(map(lambda x_t: kalman.filter(x_t), x))

    plt.plot(x[:,0], x[:,1], 'rs-')
    plt.plot(y[:,0], y[:,1], 'g^-')
    plt.plot(z[:,0], z[:,1], 'bo-')
    plt.title("Kalman Filter")
    plt.show()
