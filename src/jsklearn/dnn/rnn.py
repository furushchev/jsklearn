#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
import numpy as np
import chainer
from chainer.training import extensions
from chainer import iterators, training
import chainer.links as L
import chainer.functions as F


def get_sin(num=10000, dth=1, test=False):
    x = np.linspace(0, num+1, num+1, dtype=np.float32) * np.radians(dth)
    t = np.sin(x)
    tn = t.copy()
    # if test:
    #     return list(zip(t[:-1], tn[1:])), x[:-1]
    # else:
    #     return list(zip(t[:-1], tn[1:]))
    if test:
        return t, x
    else:
        return t


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
        z = F.relu(self.w1(x) + self.h1(self.last_z))
        self.last_z = z
        y = self.o(z)
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
            batch = next(train_iter)
            x, t = self.converter(batch, self.device)
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    pass


@cli.command()
@click.option("--hidden", type=int, default=10)
@click.option("--step", type=int, default=50)
@click.option("--dth", type=int, default=5)
@click.option("--batch-size", type=int, default=100)
@click.option("--gpu", type=int, default=-1)
@click.option("--max-epoch", type=int, default=2000)
@click.option("--out", type=str, default="rnn_result")
def train(hidden, step, batch_size, gpu, max_epoch, out, dth):
    model = RNN(hsize=hidden)
    model = AverageDiff(model)
    model.reset_state()
    if gpu >= 0:
        model.to_gpu(gpu)

    train_data = get_sin(dth=dth)
    train_iter = TimeSerialIterator(train_data, batch_size, repeat=True, allow_duplicates=True)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = BPTTUpdater(train_iter, optimizer, step, gpu)
    trainer = training.Trainer(updater, (max_epoch, "epoch"), out=out)
    print_interval = 20
    trainer.extend(extensions.LogReport(trigger=(print_interval, "iteration")))
    trainer.extend(extensions.PrintReport(["epoch", "iteration", "main/loss"]),
                   trigger=(print_interval, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.snapshot_object(
        model, "model_iter_{.updater.iteration}"))

    trainer.run()

    chainer.serializers.save_npz("%s.model" % out, model)
    click.echo("Saved model file as %s.model" % out)


@cli.command()
@click.option("--hidden", type=int, default=10)
@click.option("--step", type=int, default=50)
@click.option("--dth", type=int, default=5)
@click.option("--batch-size", type=int, default=100)
@click.option("--gpu", type=int, default=-1)
@click.option("--max-epoch", type=int, default=2000)
@click.option("--out", type=str, default="rnn_result")
def train_custom(hidden, step, batch_size, gpu, max_epoch, out, dth):
    model = RNN(hsize=hidden)
    model = AverageDiff(model)
    model.reset_state()
    if gpu >= 0:
        model.to_gpu(gpu)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_data = get_sin(dth=dth)

    episode_length = step
    num_iteration = len(train_data) // (episode_length * batch_size)
    with click.progressbar(range(max_epoch)) as it:
        for epoch in it:
            for iteration in range(num_iteration):
                episode_start_points = np.random.choice(
                    np.arange(len(train_data) - batch_size),
                    batch_size,
                )

                model.cleargrads()
                model.reset_state()
                loss = chainer.Variable(model.xp.array(0, dtype=model.xp.float32))
                for i in range(episode_length):
                    batch = []
                    for j in range(batch_size):
                        batch.append(train_data[episode_start_points[j] + i])
                    x, t = zip(*batch)
                    x = np.asarray(x, dtype=np.float32)[:, None]
                    t = np.asarray(t, dtype=np.float32)[:, None]
                    if gpu >= 0:
                        x = chainer.cuda.to_gpu(x, gpu)
                        t = chainer.cuda.to_gpu(t, gpu)
                    loss += model(x, t)
                loss.backward()
                optimizer.update()
            print(epoch, iteration, loss)
    chainer.serializers.save_npz("%s.model" % out, model)
    click.echo("Saved model file as %s.model" % out)


@cli.command()
@click.option("--model", type=str, prompt="Model file", default="rnn_result.model")
@click.option("--hidden", type=int, default=10)
@click.option("--dth", type=int, default=5)
@click.option("--step", type=int, default=50)
@click.option("--gpu", type=int, default=-1)
@click.option("--out", type=str, default="graph.png")
def run(model, hidden, dth, step, gpu, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model_path = model
    model = RNN(hsize=hidden)
    train_model = AverageDiff(model)
    train_model.reset_state()
    if gpu >= 0:
        train_model.to_gpu(gpu)

    if not model_path:
        click.echo("No pretrained model specified")
    else:
        click.echo("Loading model from %s" % model_path)
        chainer.serializers.load_npz(model_path, train_model)

    ts, xs = get_sin(num=step, dth=dth, test=True)

    x = chainer.Variable(model.xp.asarray([ts[0]], dtype=model.xp.float32)[:, None])
    ys = []
    if gpu >= 0:
        x.to_gpu(gpu)

    for i in range(len(ts)):
        y = model(x)
        x = y
        ys.append(y.data[0][0])

    plt.plot(xs, ts, linestyle='dashed', label="Ground truth")
    plt.plot(xs, ys, label="Predicted")
    plt.legend()
    plt.savefig(out)
    click.echo("Saved graph to %s" % out)


if __name__ == '__main__':
    cli()
