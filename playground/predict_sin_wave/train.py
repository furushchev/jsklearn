#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
import chainer
from chainer import training
from chainer.training import extensions
from jsklearn.dnn.rnn import RNN, RNN_LSTM
from jsklearn.dnn.lstm import FCLSTM
from jsklearn.dnn.rnn import AverageDiff
from jsklearn.dnn.rnn import BPTTUpdater
from jsklearn.dnn.rnn import TimeSerialIterator
import numpy as np


def get_sin(num=10000, dth=1, test=False):
    x = np.linspace(0, num+1, num+1, dtype=np.float32) * np.radians(dth)
    t = np.sin(x)
    if test:
        return t, x
    else:
        return t


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    pass


@cli.command()
@click.option("--model", type=click.Choice(["rnn", "lstm"]), default="rnn")
@click.option("--hidden-unit", type=int, default=10)
@click.option("--hidden-size", type=int, default=2)
@click.option("--step", type=int, default=72)
@click.option("--dth", type=int, default=5)
@click.option("--batch-size", type=int, default=100)
@click.option("--gpu", type=int, default=-1)
@click.option("--max-epoch", type=int, default=1000)
@click.option("--out", type=str, default="{model}_result")
def train(model, hidden_size, hidden_unit, step, batch_size, gpu, max_epoch, out, dth):
    model_name = model
    out = out.format(model=model_name)
    if model_name == "rnn":
        model = RNN(hsize=hidden_unit)
    elif model_name == "lstm":
        # model = RNN_LSTM(hsize=hidden_unit, psize=hidden_size)
        model = FCLSTM(hsize=hidden_unit, psize=hidden_size)
    else:
        raise click.BadParameter(model_name)

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
        model, "model_iter_{.updater.iteration}"),
                   trigger=(100, "iteration"))

    trainer.run()

    chainer.serializers.save_npz("%s.model" % out, model)
    click.echo("Saved model file as %s.model" % out)


@cli.command()
@click.option("--hidden", type=int, default=10)
@click.option("--step", type=int, default=72)
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

                model.cleargrads()
                # model.reset_state()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
            print(epoch, iteration, loss)
    chainer.serializers.save_npz("%s.model" % out, model)
    click.echo("Saved model file as %s.model" % out)


if __name__ == '__main__':
    cli()
