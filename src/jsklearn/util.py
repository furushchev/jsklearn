#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from pathlib2 import Path

_rospack = None


def get_data_path(rel_path, test=False):
    global _rospack
    # first search from relative to this file
    rel_path = Path(rel_path)
    base_path = Path(__file__).parent.absolute().parent.parent
    if test:
        path = base_path / "test" / "data" / rel_path
    else:
        path = base_path / "data" / rel_path
    if path.exists():
        return str(path)
    # fallback to use from ros package
    import rospkg
    if _rospack is None:
        _rospack = rospkg.RosPack()
    base_path = Path(_rospack.get_path("jsklearn"))
    if test:
        path = base_path / "test" / "data" / rel_path
    else:
        path = base_path / "data" / rel_path
    if path.exists():
        return str(path)
    else:
        raise IOError("%s not found" % path)
