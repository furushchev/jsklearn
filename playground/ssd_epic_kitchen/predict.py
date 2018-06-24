#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
from pathlib2 import Path
import numpy as np
import matplotlib.pyplot as plt
from jsklearn.dataset.epic_kitchen import EpicKitchenObjectDetectionDataset
from jsklearn.dataset.epic_kitchen import epic_kitchen_object_detection_label_names

import chainer
from chainercv.links import SSD300, SSD512
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox


@click.command()
@click.argument("model_path")
@click.argument("image_path")
@click.option("--recursive", "-r", is_flag=True)
@click.option("--base-model", type=click.Choice(["ssd300", "ssd512"]), default="ssd300")
@click.option("--gpu", type=int, default=-1)
def predict(model_path, image_path,
            base_model, gpu, recursive):

    click.echo("Preparing model")
    if base_model == "ssd300":
        model = SSD300(pretrained_model=model_path,
                       n_fg_class=len(epic_kitchen_object_detection_label_names))
    else:
        model = SSD512(pretrained_model=model_path,
                       n_fg_class=len(epic_kitchen_object_detection_label_names))

    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()

    if recursive:
        base_dir = Path(image_path)
        image_path = base_dir.rglob("**/*.jpg")
    else:
        image_path = [image_path]

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_init = False

    for p in image_path:
        img = read_image(p)
        bboxes, labels, scores = model.predict([img])

        print p, bboxes, labels, scores
        vis_bbox(img, bboxes[0], labels[0], scores[0],
                 label_names=epic_kitchen_object_detection_label_names, ax=ax)
        fig.canvas.draw()
        fig.canvas.flush_events()
        char = raw_input("?")
        if char.lower() == "q":
            exit(0)
        ax.clear()


if __name__ == '__main__':
    predict()
