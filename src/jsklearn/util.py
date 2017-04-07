#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
import rospkg

_rospack = rospkg.RosPack()

def get_data_path(rel_path):
    p = os.path.join(_rospack.get_path("jsklearn"), "data", rel_path)
    if os.path.exists(p):
        return p
    else:
        raise IOError("%s not found" % p)
