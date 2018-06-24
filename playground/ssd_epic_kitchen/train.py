#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import cv2
cv2.setNumThreads(0)
import copy
import click
import numpy as np
from jsklearn.dataset.epic_kitchen import EpicKitchenObjectDetectionDataset
from jsklearn.dataset.epic_kitchen import epic_kitchen_object_detection_label_names

# chainer
import chainer
from chainer import serializers
from chainer import training
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainer.optimizer import WeightDecay

# chainercv
from chainercv import transforms
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import SSD300, SSD512
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import resize_with_random_interpolation


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labs):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labs, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class Transform(object):
    """Class for augumentation"""

    def __init__(self, coder, size, mean):
        # copy to send to cpu
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        img, bbox, label = in_data

        # 1. Color augumentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            if bbox.size > 0:
                bbox = transforms.translate_bbox(
                    bbox, y_offset=param["y_offset"], x_offset=param["x_offset"])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        if bbox.size > 0:
            bbox, param = transforms.crop_bbox(
                bbox, y_slice=param["y_slice"], x_slice=param["x_slice"],
                allow_outside_center=False, return_param=True)
        if label.size > 0:
            label = label[param["index"]]

        # 4. Resizing with random interpolation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        if bbox.size > 0:
            bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Transformation for SSD network input
        img -= self.mean
        mb_loc, mb_lab = self.coder.encode(bbox, label)

        return img, mb_loc, mb_lab


@click.command()
@click.option("--base-model", type=click.Choice(["ssd300", "ssd512"]), default="ssd300")
@click.option("--base-params", type=str, default="imagenet")
@click.option("--batch-size", type=int, default=16)
@click.option("--max-iter", type=int, default=120000)
@click.option("--gpu", type=int, default=-1)
@click.option("--out", type=str, default="results")
@click.option("--lr", type=float, default=None)  # if None, use Adam
@click.option("--log-interval", type=int, default=10)
@click.option("--snapshot-interval", type=int, default=200)
def train(base_model, base_params, batch_size, max_iter,
          gpu, out, lr,
          log_interval, snapshot_interval):

    click.echo("Preparing dataset")
    dataset = EpicKitchenObjectDetectionDataset()
    seed = 123  # fix seed for tuning hyper parameters
    train_data, test_data = chainer.datasets.split_dataset_random(
        dataset, int(len(dataset) * 0.8), seed=seed)

    click.echo("Preparing model")
    if base_model == "ssd300":
        model = SSD300(pretrained_model=base_params,
                       n_fg_class=len(epic_kitchen_object_detection_label_names))
    else:
        model = SSD512(pretrained_model=base_params,
                       n_fg_class=len(epic_kitchen_object_detection_label_names))

    train_chain = MultiboxTrainChain(model)

    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()

    train_data = TransformDataset(
        train_data, Transform(model.coder, model.insize, model.mean))
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size)

    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size,
        repeat=False, shuffle=False)

    if lr == None:
        optimizer = chainer.optimizers.Adam()
    else:
        optimizer = chainer.optimizers.MomentumSGD(lr=lr)
    optimizer.setup(train_chain)

    for param in train_chain.params():
        if param.name == "b":
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.StandardUpdater(
        train_iter, optimizer, device=gpu)
    trainer = training.Trainer(
        updater, (max_iter, "iteration"), out)

    trainer.extend(extensions.LogReport(trigger=(log_interval, "iteration")))
    trainer.extend(extensions.observe_lr(), trigger=(log_interval, "iteration"))
    trainer.extend(extensions.PrintReport(
        ["epoch", "iteration", "lr",
         "main/loss", "main/loss/loc", "main/loss/conf",
         "validation/main/map"]),
                   trigger=(log_interval, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=log_interval))

    trainer.extend(
        extensions.snapshot_object(model, "model_iter_{.updater.iteration}"),
        trigger=(snapshot_interval, "iteration"))

    click.echo("Start training")

    trainer.run()

    click.echo("Done")


if __name__ == '__main__':
    train()
