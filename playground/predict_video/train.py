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
from chainer import serializers
from chainer import training
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
from jsklearn.dataset import EpicKitchenActionDataset
from jsklearn.dnn import DEMNet


def grad_diff_loss(x, t, alpha=2.0):
    dx_h = F.absolute_error(x[:,:,:,:-1], x[:,:,:,1:])
    dt_h = F.absolute_error(t[:,:,:,:-1], t[:,:,:,1:])
    dx_v = F.absolute_error(x[:,:,:-1,:], x[:,:,1:,:])
    dt_v = F.absolute_error(t[:,:,:-1,:], t[:,:,1:,:])

    dh = F.absolute_error(dx_h, dt_h) ** alpha
    dv = F.absolute_error(dx_v, dt_v) ** alpha

    return (F.mean(dh) + F.mean(dv)) / 2.0


class DEMNetTrainChain(chainer.Chain):
    def __init__(self, model, loss_func="mse", mse_ratio=0.6):
        super(DEMNetTrainChain, self).__init__()
        with self.init_scope():
            self.model = model

        if "mse" not in loss_func or "gdl" not in loss_func:
            raise ValueError("loss_func '%s' is unknown" % loss_func)
        if mse_ratio < 0.0 or mse_ratio > 1.0:
            raise ValueError("mse_ratio '%s' must be from 0.0 to 1.0" % mse_ratio)

        self.loss_func = loss_func
        self.mse_ratio = mse_ratio
        self.episode_size = self.model.episode_size

    def reset_state(self):
        self.model.reset_state()

    def __call__(self, x, t_reconst, t_pred):
        """x, t_reconst, t_pred: (B, C, H, W)"""

        pred, reconst, hidden = self.model(x)

        mse_loss = 0.
        if "mse" in self.loss_func:
            mse_loss += F.sum(F.mean_squared_error(pred, t_pred))
            mse_loss += F.sum(F.mean_squared_error(reconst, t_reconst))
            chainer.report({"loss/mse": mse_loss}, self)

        gdl_loss = 0.
        if "gdl" in self.loss_func:
            gdl_loss += grad_diff_loss(pred, t_pred, alpha=2.0)
            gdl_loss += grad_diff_loss(reconst, t_reconst, alpha=2.0)
            chainer.report({"loss/gdl": gdl_loss}, self)

        loss = mse_loss * self.mse_ratio + gdl_loss * (1.0 - self.mse_ratio)
        chainer.report({"loss": loss}, self)

        return loss


class EpisodeIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, episode_length,
                 repeat=False, shuffle=False, resize=(128, 128)):
        super(EpisodeIterator, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.repeat = repeat
        self.epoch = 0
        self.iteration = 0
        self.video = None
        self.resize = resize
        self.is_new_epoch = False
        self._previous_epoch_detail = -1.0
        if shuffle:
            self.offsets = np.random.permutation(len(self.dataset))[:self.batch_size]
        else:
            length = len(self.dataset) // self.batch_size
            self.offsets = np.asarray([i * length for i in range(self.batch_size)])

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.iteration = serializer("iteration", self.iteration)
        self.epoch = serializer("epoch", self.epoch)
        try:
            self._previous_epoch_detail = serializer(
                "previous_epoch_detail", self._previous_epoch_detail)
        except KeyError:
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.0)
            else:
                self._previous_epoch_detail = -1.0

    def __next__(self):
        min_frames = 0
        length = len(self.dataset)
        episode_length = self.episode_length + 1  # for pred image
        self._previous_epoch_detail = self.epoch_detail
        while min_frames < episode_length:
            if not self.repeat and self.iteration * self.batch_size >= length:
                raise StopIteration()
            indices = [(offset + self.iteration) % length for offset in self.offsets]
            lengths = [self.dataset.get_length(i) for i in indices]
            min_frames = min(lengths)
            # print "v", indices, lengths, min_frames, episode_length
            self.iteration += 1
        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

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
            train_iter, optimizer,
            device=device,)

    def update_core(self):
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        model = optimizer.target
        assert train_iter.episode_length == model.episode_size

        loss = 0

        model.reset_state()
        x, t_reconst, t_pred = next(train_iter)  # NBCHW
        n = x.shape[0]
        for i in range(n):
            xi = chainer.dataset.to_device(self.device, x[i])
            tri = xi  # self.converter(t_reconst[i], self.device)
            tpi = chainer.dataset.to_device(self.device, t_pred[i])
            loss += model(chainer.Variable(xi),
                          chainer.Variable(tri),
                          chainer.Variable(tpi))

        model.cleargrads()
        loss.backward()
        loss.unchain_backward()  # not necessary?
        optimizer.update()


@click.command()
@click.option("--batch-size", type=int, default=2)
@click.option("--max-iter", type=int, default=100000)
@click.option("--gpu", type=int, default=0)
@click.option("--out", type=str, default="results")
@click.option("--fps", type=float, default=4.0)
@click.option("--loss-func", type=click.Choice(["mse", "gdl", "mse_gdl"]), default="mse_gdl")
@click.option("--log-interval", type=int, default=10)
@click.option("--snapshot-interval", type=int, default=100)
@click.option("--resume", type=str, default="")
def train(batch_size, max_iter, gpu, out, fps, loss_func, log_interval, snapshot_interval, resume):
    click.echo("Preparing model")
    model = DEMNet(hidden_channels=1000, out_channels=3)
    train_model = DEMNetTrainChain(model, loss_func=loss_func)

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
    trainer.extend(extensions.PrintReport(
        ["epoch", "iteration", "lr", "main/loss", "main/loss/mse", "main/loss/gdl"]),
                   trigger=(log_interval, "iteration"))
    trainer.extend(extensions.PlotReport(["main/loss"]),
                   trigger=(log_interval, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=log_interval))
    trainer.extend(extensions.dump_graph(root_name="main/loss", out_name="network.dot"))
    trainer.extend(extensions.snapshot(), trigger=(snapshot_interval, "iteration"))
    trainer.extend(extensions.snapshot_object(model, "model_iter_{.updater.iteration}"),
                   trigger=(snapshot_interval, "iteration"))

    if resume:
        click.echo("Resuming %s" % resume)
        serializers.load_npz(resume, trainer)

    click.echo("Start training")

    trainer.run()

    click.echo("Done")


if __name__ == '__main__':
    train()
