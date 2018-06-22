#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import numpy as np
import imageio
from pathlib2 import Path
import chainer


class VideoDataset(chainer.dataset.DatasetMixin):
    """Time serialized Image Dataset from Videos

    Args:
        data_dir (string): Path to the root of the training data.
        split ({'train', 'val', 'test'}): Select from dataset splits
    """

    def __init__(self, data_dir, split='train', dtype=np.float32):
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise IOError("data_dir '%s' does not exist or not a directory" % data_dir)
        split_candidates = ["train", "val", "test"]
        if split not in split_candidates:
            raise ValueError(
                "Please pick split from %s" % split_candidates)

        video_paths = []
        index = data_dir / ("%s.txt" % split)
        if index.exists():
            with index.open() as f:
                for p in f.readlines():
                    p = Path(p.strip())
                    if not p.is_absolute():
                        p = data_dir / p
                    if not p.exists():
                        print "'%s' does not exists" % p
                        continue
                    video_paths.append(p)
        else:
            print "All videos are used"
            for ext in ["mp4", "mov", "avi", "mpg", "mpeg", "mkv", "wmv"]:
                video_paths.extend(list(data_dir.rglob("*.%s" % ext)))

        self.dtype = dtype
        self.data_dir = data_dir
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)

    def get_example(self, i):
        if i >= len(self):
            raise IndexError("index is too large")

        video_path = self.video_paths[i].absolute()
        try:
            reader = imageio.get_reader(str(video_path))
        except imageio.plugins.ffmpeg.NeedDownloadError:
            imageio.plugins.ffmpeg.download()
            reader = imageio.get_reader(str(video_path))

        frames = []
        for i in range(reader.get_length()):
            try:
                frame = reader.get_data(i)
                frames.append(frame)
            except imageio.core.format.CannotReadFrameError:
                print "Format error at frame %d of %s. Skiped." % (i, video_path.name)
        frames = np.asarray(frames, dtype=self.dtype)
        return frames.transpose(3, 0, 2, 1)  # CHW
