#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import numpy as np
import os.path as osp
from chainercv import utils
from epic_kitchen_utils import get_object_detection_images
from epic_kitchen_utils import parse_action_annotation
from epic_kitchen_object_detection_labels import epic_kitchen_object_detection_label_names


class EpicKitchenActionDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir="auto", anno_path="auto", split="train",
                 force_download=False, skip_no_image=True):
        if split not in ["train", "test"]:
            raise ValueError("Split '%s' not available" % split)

        if data_dir == "auto":
            if anno_path == "auto":
                data_dir, anno_path, annotations = get_action_videos(
                    split, force_download=force_download)
            else:
                data_dir = get_action_videos(
                    split, force_download=force_download)[0]
                annotations = parse_action_annotation(anno_path)
        else:
            if anno_path == "auto":
                raise ValueError("auto anno_path with specified data_dir is unsupported")
            else:
                annotations = parse_action_annotation(anno_path)

        self.data_dir = data_dir
        self.split = split
        self.annotations = self._filter_annotations(annotations)

    def __len__(self):
        return len(self.annotations)

    def _filter_annotations(self, annotations):
        return annotations

    def get_example(self, i):
        pass


if __name__ == '__main__':
    from chainercv.visualizations import vis_image
    import matploglib.pyplot as plt

    dataset = EpicKitchenActionDataset(split="train")

    print "Loadded dataset. length:", len(dataset)
    print "Starting slideshow..."

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_init = False
    for i in range(0, len(dataset), 10):
        try:
            img, bbox, label = dataset.get_example(i)
        except Exception as e:
            print i, e
        ax = vis_bbox(img, bbox, label, label_names=epic_kitchen_object_detection_label_names, ax=ax)
        if plot_init is False:
            plt.show()
            plot_init = True
        else:
            fig.canvas.draw()
            fig.canvas.flush_events()
        plt.pause(0.1)
        ax.clear()
