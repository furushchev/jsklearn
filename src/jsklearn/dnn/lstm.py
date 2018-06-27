#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import numpy as np
import six
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I


class LSTM(chainer.Chain):
    def __init__(self, hidden_size, forget_bias_init=None):
        super(LSTM, self).__init__()
        if forget_bias_init is None:
            forget_bias_init = 1
        self.forget_bias_init = forget_bias_init

        self.hidden_size = hidden_size

        with self.init_scope():
            self.upward=L.Linear(hidden_size, 4 * hidden_size, initialW=0)
            self.hidden=L.Linear(hidden_size, 4 * hidden_size, initialW=0, nobias=True)
            self._initialize_params()

        self.reset_state()

    def _initialize_params(self):
        forget_bias_init = I._get_initializer(self.forget_bias_init)
        a, i, f, o = F.activation.lstm._extract_gates(
            self.upward.b.data.reshape(1, 4 * self.hidden_size, 1))
        forget_bias_init(f)

    def reset_state(self):
        self.c = None
        self.h = None

    def __call__(self, x):
        batch_size = x.shape[0]
        if self.h is None:
            self.h = chainer.Variable(self.xp.zeros((batch_size, self.hidden_size), dtype=self.xp.float32))
        if self.c is None:
            self.c = chainer.Variable(self.xp.zeros((batch_size, self.hidden_size), dtype=self.xp.float32))

        h = self.upward(x) + self.hidden(self.h)
        self.c, self.h = F.lstm(self.c, h)
        return self.h


class FCLSTM(chainer.Chain):
    def __init__(self, hsize=10, psize=2):
        super(FCLSTM, self).__init__()
        self.hsize = hsize
        self.psize = psize
        with self.init_scope():
            self.l0 = L.Linear(1, hsize)
            for i in range(psize):
                setattr(self, "lstm%d" % i, LSTM(hsize))
            self.o0 = L.Linear(hsize, 1)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        for i in range(self.psize):
            lstm = getattr(self, "lstm%d" % i)
            lstm.reset_state()

    def __call__(self, x):
        h = self.l0(x)
        for i in range(self.psize):
            lstm = getattr(self, "lstm%d" % i)
            h = lstm(F.dropout(h))
        y = self.o0(F.dropout(h))
        return y


class ConvLSTM2D(chainer.Chain):
    """https://arxiv.org/pdf/1607.06450.pdf"""
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=None,
                 hidden_init=None, upward_init=None, bias_init=None, forget_bias_init=None):
        if out_channels is None:
            in_channels, out_channels = None, in_channels
        if pad is None:
            pad = ksize // 2
        super(ConvLSTM2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if bias_init is None:
            bias_init = 0
        if forget_bias_init is None:
            forget_bias_init = 1
        self.hidden_init = hidden_init
        self.upward_init = upward_init
        self.bias_init = bias_init
        self.forget_bias_init = forget_bias_init
        self.c = None
        self.h = None

        with self.init_scope():
            self.upward=L.Convolution2D(in_channels, 4 * out_channels,
                                        ksize=ksize, stride=stride, pad=pad,
                                        initialW=0)
            self.hidden=L.Convolution2D(out_channels, 4 * out_channels,
                                        ksize=ksize, stride=stride, pad=pad,
                                        initialW=0, nobias=True)
            if in_channels is not None:
                self._initialize_params()

    def _initialize_params(self):
        hidden_init = I._get_initializer(self.hidden_init)
        upward_init = I._get_initializer(self.upward_init)
        bias_init = I._get_initializer(self.bias_init)
        forget_bias_init = I._get_initializer(self.forget_bias_init)

        for i in six.moves.range(0, 4 * self.out_channels, self.out_channels):
            hidden_init(self.hidden.W.data[i:i + self.out_channels, :])
            upward_init(self.upward.W.data[i:i + self.out_channels, :])

        a, i, f, o = F.activation.lstm._extract_gates(
            self.upward.b.data.reshape(1, 4 * self.out_channels, 1))

        bias_init(a)
        bias_init(i)
        forget_bias_init(f)
        bias_init(o)

    def reset_state(self):
        self.c = None
        self.h = None

    def set_state(self, c, h):
        assert isinstance(c, chainer.Variable)
        assert isinstance(h, chainer.Variable)

        c_, h_ = c, h
        if self.xp == numpy:
            c_.to_cpu()
            h_.to_cpu()
        else:
            c_.to_gpu(self._device_id)
            h_.to_gpu(self._device_id)
        self.c, self.h = c_, h_

    def to_cpu(self):
        super(ConvLSTM2D, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(ConvLSTM2D, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def __call__(self, x):
        # print "in", self.in_channels, "out", self.out_channels
        batch_size, in_channels, height, width = x.shape

        if self.upward.W.data is None:
            with chainer.cuda.get_device_from_id(self._device_id):
                self.upward._initialize_params(in_channels)
                self._initialize_params()

        h = self.upward(x)

        if self.h is not None:
            hh = self.hidden(self.h)
            h += hh

        if self.c is None:
            xp = self.xp
            with chainer.cuda.get_device_from_id(self._device_id):
                self.c = chainer.Variable(
                    xp.zeros((batch_size, self.out_channels, height, width),
                             dtype=x.dtype))

        self.c, self.h = F.lstm(self.c, h)
        return self.h
