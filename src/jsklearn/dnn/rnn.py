#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import numpy as np
import chainer
from chainer.training import extensions
from chainer import iterators, training
import chainer.links as L
import chainer.functions as F


class RNN(chainer.Chain):
    def __init__(self, hsize=10):
        super(RNN, self).__init__(
            w1=L.Linear( 1, hsize),
            h1=L.Linear(hsize, hsize),
            o =L.Linear(hsize,  1),
        )
        self.hsize = hsize
        self.reset_state()

    def reset_state(self):
        self.last_z = None

    def __call__(self, x):
        bsize = x.shape[0]
        if self.last_z is None:
            self.last_z = self.xp.zeros((bsize, self.hsize), dtype=self.xp.float32)
        z = F.tanh(self.w1(x) + self.h1(self.last_z))
        self.last_z = z
        y = self.o(z)
        return y


class RNN_LSTM(chainer.Chain):
    def __init__(self, hsize=10, psize=2):
        super(RNN_LSTM, self).__init__()
        self.hsize = hsize
        self.psize = psize
        with self.init_scope():
            self.l0 = L.Linear(1, hsize)
            for i in range(psize):
                setattr(self, "lstm%d" % i, L.LSTM(hsize, hsize))
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


class AverageDiff(chainer.Chain):
    def __init__(self, predictor):
        super(AverageDiff, self).__init__(predictor=predictor)

    def reset_state(self):
        self.predictor.reset_state()

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.average(F.absolute(y - t))
        chainer.report({"loss": loss}, self)
        return loss


class TimeSerialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batchsize, repeat=False, allow_duplicates=False):
        super(TimeSerialIterator, self).__init__()
        self.dataset = dataset
        self.batchsize = batchsize
        self.repeat = repeat
        self.allow_duplicates = allow_duplicates
        self.epoch = 0
        self.iteration = 0
        self.is_new_epoch = True

        if self.allow_duplicates:
            self.offsets = np.random.choice(
                np.arange(len(self.dataset)), self.batchsize)
        else:
            length = len(self.dataset) // self.batchsize
            self.offsets = np.asarray([i * length for i in range(self.batchsize)])

    def __next__(self):
        if not self.repeat:
            if self.iteration * self.batchsize >= len(self.dataset):
                raise StopIteration()
        x = self.get_data()
        self.iteration += 1
        t = self.get_data()

        epoch = self.iteration * self.batchsize // len(self.dataset)
        self.is_new_epoch = self.epoch < epoch
        self.epoch = epoch
        return list(zip(x, t))

    @property
    def epoch_detail(self):
        return self.iteration * self.batchsize / len(self.dataset)

    def get_data(self):
        length = len(self.dataset) // self.batchsize
        return np.asarray([self.dataset[(offset + self.iteration) % len(self.dataset)]
                           for offset in self.offsets], dtype=np.float32)[:, np.newaxis]

    def serialize(self, serializer):
        self.iteration = serializer("iteration", self.iteration)
        self.epoch = serializer("epoch", self.epoch)
        self.offsets = serializer("offsets", self.offsets)


class BPTTUpdater(training.updaters.StandardUpdater):
    def __init__(self, train_iter, optimizer, episode_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.episode_len = episode_len

    def update_core(self):
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        model = optimizer.target
        loss = chainer.Variable(model.xp.array(0, dtype=model.xp.float32))

        for i in range(self.episode_len):
            if train_iter.is_new_epoch:
                model.cleargrads()
            batch = next(train_iter)
            x, t = self.converter(batch, self.device)
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
