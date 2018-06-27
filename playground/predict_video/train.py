#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import cv2
import click
import os
import numpy as np

if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("Agg")

import chainer
from chainer import training
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
from jsklearn.dataset import EpicKitchenActionDataset
from jsklearn.dnn import DEMNet


class DEMNetTrainChain(chainer.Chain):
    def __init__(self, model, loss_func="mse"):
        super(DEMNetTrainChain, self).__init__()
        with self.init_scope():
            self.model = model

        if loss_func not in ["mse"]:
            raise ValueError("loss_func '%s' is unknown" % loss_func)

        self.loss_func = loss_func
        self.episode_size = self.model.episode_size

    def reset_state(self):
        self.model.reset_state()

    def __call__(self, x, t_reconst, t_pred):
        """x, t_reconst, t_pred: (B, C, H, W)"""

        loss = chainer.Variable(self.xp.array(0, dtype=self.xp.float32))
        if self.xp == np:
            loss.to_cpu()
        else:
            loss.to_gpu(self._device_id)

        pred, reconst, hidden = self.model(x)
        if self.loss_func == "mse":
            loss += F.sum(F.mean_squared_error(pred, t_pred))
            loss += F.sum(F.mean_squared_error(reconst, t_reconst))
        else:
            raise ValueError("Invalid loss_func: %s" % self.loss_func)

        chainer.report({"loss": loss}, self)
        return loss


class EpisodeIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, episode_length,
                 repeat=False, shuffle=False, seed=None, resize=(128, 128)):
        super(EpisodeIterator, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.repeat = repeat
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.video_iteration = 0
        self.video = None
        self.video_length = 0
        self.next_image = None
        self.resize = resize
        if shuffle:
            self.offsets = np.random.permutation(len(self.dataset))[:self.batch_size]
        else:
            length = len(self.dataset) // self.batch_size
            self.offsets = np.asarray([i * length for i in range(self.batch_size)])


    @property
    def epoch_detail(self):
        return self.video_iteration * self.batch_size / len(self.dataset)

    def __next__(self):
        min_frames = 0
        episode_length = self.episode_length + 1  # for pred image
        while min_frames < episode_length:
            if not self.repeat and self.video_iteration * self.batch_size >= len(self.dataset):
                raise StopIteration()
            indices = [(offset + self.video_iteration) % len(self.dataset) for offset in self.offsets]
            lengths = [self.dataset.get_length(i) for i in indices]
            min_frames = min(lengths)
            # print "v", indices, lengths, min_frames, episode_length
            self.video_iteration += 1

        data = []
        for i in indices:
            imgs, label = self.dataset[i]
            imgs = imgs[:episode_length]
            data.append(imgs)
        data = np.asarray(data, dtype=np.float32)  # BNCHW
        data = data.transpose((1, 0, 2, 3, 4))     # NBCHW
        return data[:-1], data[:-1], data[1:]  # x, t_reconst, t_pred


class EpisodeUpdater(training.updaters.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(EpisodeUpdater, self).__init__(
            train_iter, optimizer, device=device)

    def update_core(self):
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        model = optimizer.target
        loss = chainer.Variable(model.xp.array(0, dtype=model.xp.float32))
        if model.xp == np:
            loss.to_cpu()
        else:
            loss.to_gpu(model._device_id)

        assert train_iter.episode_length == model.episode_size

        model.reset_state()
        x, t_reconst, t_pred = next(train_iter)  # NBCHW
        n = x.shape[0]
        for i in range(n):
            xi = self.converter(x[i], self.device)
            tri = xi  # self.converter(t_reconst[i], self.device)
            tpi = self.converter(t_pred[i], self.device)
            loss += model(chainer.Variable(xi),
                          chainer.Variable(tri),
                          chainer.Variable(tpi))

        model.cleargrads()
        loss.backward()
        # loss.unchain_backward()
        optimizer.update()


@click.command()
@click.option("--batch-size", type=int, default=2)
@click.option("--max-iter", type=int, default=100000)
@click.option("--gpu", type=int, default=0)
@click.option("--out", type=str, default="results")
@click.option("--fps", type=float, default=4.0)
@click.option("--log-interval", type=int, default=10)
@click.option("--snapshot-interval", type=int, default=100)
def train(batch_size, max_iter, gpu, out, fps, log_interval, snapshot_interval):
    click.echo("Preparing model")
    model = DEMNet(hidden_channels=1000, out_channels=3)
    train_model = DEMNetTrainChain(model)

    model.reset_state()
    if gpu >= 0:
        model.to_gpu(gpu)

    click.echo("Loading dataset")
    train_data = EpicKitchenActionDataset(split="train", fps=fps)
    train_iter = EpisodeIterator(train_data, batch_size, model.episode_size, repeat=True, shuffle=False)

    click.echo("Setting up trainer")
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(train_model)

    updater = EpisodeUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (max_iter, "iteration"), out=out)

    trainer.extend(extensions.LogReport(trigger=(log_interval, "iteration")))
    trainer.extend(extensions.observe_lr(), trigger=(log_interval, "iteration"))
    trainer.extend(extensions.PrintReport(["epoch", "iteration", "lr", "main/loss"]),
                   trigger=(log_interval, "iteration"))
    trainer.extend(extensions.PlotReport(["main/loss"]),
                   trigger=(log_interval, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=log_interval))
    trainer.extend(extensions.dump_graph(root_name="main/loss", out_name="network.dot"))
    trainer.extend(extensions.snapshot_object(model, "model_iter_{.updater.iteration}"),
                   trigger=(snapshot_interval, "iteration"))

    click.echo("Start training")

    trainer.run()

    click.echo("Done")


if __name__ == '__main__':
    train()
