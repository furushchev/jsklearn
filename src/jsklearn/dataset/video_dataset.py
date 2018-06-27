#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import cv2
import chainer
import glob
import numpy as np
import os.path as osp
import imageio


def _check_video(vp):
    video = None
    try:
        video = imageio.get_reader(vp)
        meta = video.get_meta_data()
        return True
    except:
        return False
    finally:
        if video is not None:
            video.close()


def _find_videos(dir_path, recursive=False):
    videos = []
    if recursive:
        for d in os.listdir(dir_path):
            videos += _find_videos(d, True)
    fns = []
    exts = ["mp4", "avi"]
    for ext in exts:
        ext = "*.%s" % ext
        fns += glob.glob(osp.join(dir_path, ext.lower()))
        fns += glob.glob(osp.join(dir_path, ext.upper()))
    return [osp.abspath(fn) for fn in fns]


class VideoDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root_dir, recursive=False, fps=1.0, resize=(128, 128)):
        if not osp.exists(root_dir):
            raise IOError("%s does not exist" % root_dir)


        video_paths = _find_videos(root_dir, recursive=recursive)
        self.video_paths = filter(_check_video, video_paths)
        self.fps = fps
        self.resize = resize

    def __len__(self):
        return len(self.video_paths)

    def get_example(self, i):
        video_path = self.video_paths[i]
        video = None
        images = []
        try:
            video = imageio.get_reader(video_path)
            meta = video.get_meta_data()
            nframes, fps = meta["nframes"], meta["fps"]
            nskips = int(fps // self.fps)
            for frame in range(0, nframes, nskips):
                try:
                    img = video.get_data(frame)
                    if self.resize:
                        img = cv2.resize(img, self.resize)
                except IndexError as e:
                    import traceback
                    print traceback.format_exc()
                    print meta
                    print i, nframes, nskips
                    raise e
                if img.ndim == 2:
                    img = img[np.newaxis]  # 1HW
                else:
                    img = img.transpose((2, 0, 1))  # CHW
                images.append(img)
        finally:
            if video is not None:
                video.close()
        return np.asarray(images, dtype=np.float32)


if __name__ == '__main__':
    from chainercv.visualizations import vis_image
    import matplotlib.pyplot as plt
    from jsklearn.util import get_data_path
    import traceback

    root_dir = get_data_path("video")
    dataset = VideoDataset(root_dir, fps=1.0)

    print "Loadded dataset. length:", len(dataset)
    print "Starting slideshow..."

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_init = False
    for i in range(0, len(dataset), 1):
        try:
            images = dataset.get_example(i)
        except Exception as e:
            print i, e
            traceback.print_exc()
        frames, channels, height, width = images.shape
        for i in range(frames):
            img = images[i]
            ax = vis_image(img, ax=ax)
            if plot_init is False:
                plt.show()
                plot_init = True
            else:
                fig.canvas.draw()
                fig.canvas.flush_events()
            plt.pause(0.1)
            ax.clear()
