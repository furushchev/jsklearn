#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import numpy as np
import os.path as osp
from chainercv import utils
import imageio
from epic_kitchen_utils import get_action_videos
from epic_kitchen_utils import parse_action_annotation
from epic_kitchen_action_labels import epic_kitchen_action_label_names


class EpicKitchenActionDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir="auto", anno_path="auto", split="train",
                 fps=1.0,
                 force_download=False, download_timeout=None,
                 skip_no_image=True):
        if split not in ["train", "test"]:
            raise ValueError("Split '%s' not available" % split)

        if data_dir == "auto":
            if anno_path == "auto":
                data_dir, anno_path, annotations = get_action_videos(
                    split, force_download=force_download, download_timeout=download_timeout)
            else:
                data_dir = get_action_videos(
                    split, force_download=force_download, download_timeout=download_timeout)[0]
                annotations = parse_action_annotation(anno_path)
        else:
            if anno_path == "auto":
                raise ValueError("auto anno_path with specified data_dir is unsupported")
            else:
                annotations = parse_action_annotation(anno_path)

        self.data_dir = data_dir
        self.split = split
        self.annotations = self._filter_annotations(annotations)
        self.fps = fps

    def __len__(self):
        return len(self.annotations)

    def _filter_annotations(self, annotations, skip_no_image=True):
        valid_annos = []
        for anno in annotations:
            if skip_no_image:
                video_path = self.get_video_path(anno)
                if not osp.exists(video_path):
                    continue
            valid_annos.append(anno)
        return valid_annos

    def get_video_path(self, annotation):
        pid, vid = annotation["participant_id"], annotation["video_id"]
        video_path = osp.join(self.data_dir, self.split, pid, vid + ".MP4")
        return video_path

    def get_images(self, annotation):
        video_path = self.get_video_path(annotation)
        video = imageio.get_reader(video_path)
        meta = video.get_meta_data()
        video_frames, video_fps = meta["nframes"], meta["fps"]
        nskips = int(video_fps // self.fps)

        start_frame = annotation["start_frame"]
        stop_frame = annotation["stop_frame"]

        images = []
        for i in range(start_frame, stop_frame, nskips):
            img = video.get_data(i)
            if img.ndim == 2:
                img = img[np.newaxis]  # 1HW
            else:
                img = img.transpose((2, 0, 1))  # CHW
            images.append(img)
        return np.asarray(images, dtype=np.float32)

    def get_example(self, i):
        annotation = self.annotations[i]
        images = self.get_images(annotation)
        verb = annotation["verb_class"]
        nouns = annotation["all_noun_classes"]
        label = np.asarray([verb] + nouns, dtype=np.int32)
        return images, label


if __name__ == '__main__':
    from chainercv.visualizations import vis_bbox
    import matplotlib.pyplot as plt
    import traceback

    dataset = EpicKitchenActionDataset(split="train", fps=2.0)

    print "Loadded dataset. length:", len(dataset)
    print "Starting slideshow..."

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_init = False
    for i in range(0, len(dataset), 10):
        try:
            images, label = dataset.get_example(i)
        except Exception as e:
            print i, e
            traceback.print_exc()
        frames, channels, height, width = images.shape
        dummy_bbox = np.asarray([[0, 0, height, width]])
        for i in range(frames):
            img = images[i]
            ax = vis_bbox(img, dummy_bbox, [label[0]], label_names=epic_kitchen_action_label_names, ax=ax)
            if plot_init is False:
                plt.show()
                plot_init = True
            else:
                fig.canvas.draw()
                fig.canvas.flush_events()
            plt.pause(0.1)
            ax.clear()
