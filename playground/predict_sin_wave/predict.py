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
from jsklearn.dnn.rnn import TimeSerialIterator
import numpy as np


def get_sin(num=10000, dth=1, test=False):
    x = np.linspace(0, num+1, num+1, dtype=np.float32) * np.radians(dth)
    t = np.sin(x)
    if test:
        return t, x
    else:
        return t


@click.command()
@click.argument("model_path")
@click.option("--model", type=click.Choice(["rnn", "lstm"]), default="rnn")
@click.option("--hidden-unit", type=int, default=10)
@click.option("--hidden-size", type=int, default=2)
@click.option("--dth", type=int, default=5)
@click.option("--step", type=int, default=72)
@click.option("--gpu", type=int, default=-1)
@click.option("--out", type=str, default="{model}_graph.png")
def predict(model_path, model, hidden_unit, hidden_size, dth, step, gpu, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model_type = model
    if model_type == "rnn":
        model = RNN(hsize=hidden_unit)
    elif model_type == "lstm":
        # model = RNN_LSTM(hsize=hidden_unit, psize=hidden_size)
        model = FCLSTM(hsize=hidden_unit, psize=hidden_size)
    else:
        raise click.BadParameter(model_type)

    out = out.format(model=model_type)

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

    start = step // 10
    x = chainer.Variable(model.xp.asarray([ts[start]], dtype=model.xp.float32)[:, None])
    ys = []
    if gpu >= 0:
        x.to_gpu(gpu)

    with chainer.using_config("train", False):
        for i in range(len(ts[start:])):
            y = model(x)
            x = y
            ys.append(y.data[0][0])

    plt.plot(xs[start:], ts[start:], linestyle='dashed', label="Ground truth")
    plt.plot(xs[start:], ys, label="Predicted")
    plt.legend()
    plt.savefig(out)
    click.echo("Saved graph to %s" % out)


if __name__ == '__main__':
    predict()
