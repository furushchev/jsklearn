#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
import copy
import numpy as np
import os
if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("Agg")

import chainer
from chainercv.visualizations import vis_image
import matplotlib.pyplot as plt
from jsklearn.dataset import EpicKitchenActionDataset
from jsklearn.dataset import VideoDataset
from jsklearn.dnn import DEMNet


@click.command()
@click.argument("model_path")
@click.argument("data_dir")
@click.option("--gpu", type=int, default=0)
@click.option("--fps", type=float, default=1.0)
def predict(model_path, data_dir, gpu, fps):
    click.echo("Loading model")
    model = DEMNet(hidden_channels=1000, out_channels=3)
    model.reset_state()
    if gpu >= 0:
        model.to_gpu(gpu)

    if not model_path:
        click.echo("No pretrained model specified")
    else:
        click.echo("Loading model from %s" % model_path)
        chainer.serializers.load_npz(model_path, model)

    click.echo("Loading dataset")
    # train_data = VideoDataset(data_dir, fps=fps)
    train_data = EpicKitchenActionDataset(split="train", fps=fps)

    plt.ion()
    fig = plt.figure()
    tl = fig.add_subplot(221)
    tr = fig.add_subplot(222)
    bl = fig.add_subplot(223)
    br = fig.add_subplot(224)

    plot_init = False

    click.echo("Feeding")
    reconst_imgs, pred_imgs, hiddens = [], [], []
    for imgs, label in train_data:
        img_len = imgs.shape[0]
        if img_len < model.episode_size + 1:
            continue
        imgs = imgs[:model.episode_size+1]
        model.reset_state()
        for i in range(model.episode_size):
            img, next_img = imgs[i], imgs[i+1]
            x = chainer.Variable(img[np.newaxis, :])
            if gpu >= 0:
                x.to_gpu()
            pred, reconst, hidden = model(x)
            if gpu >= 0:
                pred.to_cpu()
                reconst.to_cpu()
                hidden.to_cpu()
            pred, reconst, hidden = pred.data[0], reconst.data[0], hidden.data[0]
            reconst_imgs.append(reconst)
            pred_imgs.append(pred)
            hiddens.append(hidden)

            tl.set_title("Current")
            tl = vis_image(img, ax=tl)
            tr.set_title("Reconst")
            tr = vis_image(reconst, ax=tr)
            bl.set_title("Next")
            bl = vis_image(next_img, ax=bl)
            br.set_title("Pred")
            br = vis_image(pred, ax=br)
            if plot_init is False:
                plt.show()
                plot_init = True
            else:
                fig.canvas.draw()
                fig.canvas.flush_events()
            plt.pause(3)
            tl.clear()
            tr.clear()
            bl.clear()
            br.clear()



if __name__ == '__main__':
    predict()
