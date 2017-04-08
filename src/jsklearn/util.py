#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
import rospkg

_rospack = rospkg.RosPack()

def get_data_path(rel_path, test=False):
    path = _rospack.get_path("jsklearn")
    if test:
        path = os.path.join(path, "test", "data")
    else:
        path = os.path.join(path, "data")
    path = os.path.join(path, rel_path)
    if os.path.exists(path):
        return path
    else:
        raise IOError("%s not found" % path)
