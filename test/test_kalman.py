#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>


from jsklearn.util import get_data_path
from jsklearn.filtering.kalman import AdaptiveKalman
import numpy as np
import os
import unittest


class TestKalmanFilter(unittest.TestCase):
    def test_kalman(self):
        data = np.loadtxt(get_data_path("common/sin.txt"))

        kalman = AdaptiveKalman(state_dim=4, meas_dim=2, R_var=1e-3)
        self.assertIsNotNone(kalman)
        kalman.F = np.array([[1., 0., 1., 0.],
                             [0., 1., 0., 1.],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])

        result = np.array(map(lambda x: [kalman.filter(x), kalman.error, kalman.P[1,1]],
                              data))
        z = np.vstack(result[:,0]).astype(np.float)
        e = result[:, 1]
        cov = result[:,2]

        ref = np.loadtxt(get_data_path("sin_adaptive_kalman_result.txt", test=True))

        np.testing.assert_almost_equal(z[:, 1], ref[:, 1], decimal=3,
                                       err_msg="invalid estimated y")

        np.testing.assert_almost_equal(e, ref[:, 2], decimal=3,
                                       err_msg="invalid estimated error")

        np.testing.assert_almost_equal(cov, ref[:, 3], decimal=3,
                                       err_msg="invalid P(1,1)")


if __name__ == '__main__':
    unittest.main()
