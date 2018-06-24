#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from collections import OrderedDict
import csv
import json
import os.path as osp
from chainer.dataset import download
import multiprocessing as mp
from chainercv import utils


root = "pfnet/chainercv/epic_kitchen"
data_url_base = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"
version = "1.2.2"
anno_url = "https://github.com/epic-kitchens/annotations/archive/v{version}.tar.gz".format(version=version)


def get_videos(split):
    raise NotImplementedError()


def get_frames(split):
    raise NotImplementedError()


def _get_image(url, destination, force_download, extract=True):
    ext = osp.splitext(url)[1]

    if not force_download:
        if osp.exists(destination):
            return str(destination)
        else:
            archive = destination + ".tar"
            if osp.exists(archive):
                utils.extractall(archive, destination, ext)
                return str(destination)

    if extract:
        cache_path = utils.cached_download(url)
        utils.extractall(cache_path, destination, ext)
    else:
        from six.moves.urllib import request
        import tempfile
        import filelock
        import os
        import shutil
        from chainercv.utils.download import _reporthook
        temp_root = tempfile.mkdtemp()
        try:
            temp_path = osp.join(temp_root, osp.basename(url))
            data_dir = download.get_dataset_directory(root)
            lock_path = osp.join(data_dir, "_dl_lock")
            print('Downloading ...')
            print('From: {:s}'.format(url))
            print('To: {:s}'.format(temp_path))
            request.urlretrieve(url, temp_path, _reporthook)
            with filelock.FileLock(lock_path):
                os.makedirs(destination)
                shutil.move(temp_path, destination)
        finally:
            shutil.rmtree(temp_root)

    return str(destination)


def _get_image_map_func(prop):
    try:
        return _get_image(**prop)
    except Exception as e:
        return e


def parse_object_detection_annotation(in_path):
    with open(in_path) as f:
        reader = csv.reader(f)
        keys = next(reader)
        imgfns = OrderedDict()
        for values in reader:
            dic = dict(zip(keys, values))
            key = dic["participant_id"] + "__" + dic["video_id"] + "__" + dic["frame"]
            anno = {
                "noun": dic["noun"],
                "noun_class": int(dic["noun_class"]),
                "bounding_boxes": eval(dic["bounding_boxes"]),
            }
            if key in imgfns:
                imgfns[key]["annotations"].append(anno)
            else:
                imgfns[key] = {
                    "participant_id": dic["participant_id"],
                    "video_id": dic["video_id"],
                    "frame": int(dic["frame"]),
                    "annotations": [anno],
                }
    return imgfns.values()


def get_object_detection_images(split, download_parallel_num=None, download_timeout=None, force_download=False):
    if split != "train":
        raise ValueError("split '%s' not available" % split)

    data_dir = download.get_dataset_directory(root)
    images_root = osp.join(data_dir, "object_detection_images")
    annos_root = osp.join(data_dir, "annotations-{version}".format(version=version))
    anno_fn = "EPIC_{split}_object_labels.csv".format(split=split)

    # download annotation
    if not osp.exists(osp.join(annos_root, anno_fn)):
        download_file_path = utils.cached_download(anno_url)
        ext = osp.splitext(anno_url)[1]
        utils.extractall(download_file_path,
                         data_dir, ext)
    anno_path = osp.join(annos_root, anno_fn)
    annotations = parse_object_detection_annotation(anno_path)
    image_archive_fns = set()
    for anno in annotations:
        image_archive_fn = osp.join(anno["participant_id"], anno["video_id"])
        image_archive_fns.add(image_archive_fn)

    # download images
    if download_parallel_num is None:
        download_parallel_num = mp.cpu_count()
    if download_timeout is None:
        download_timeout = 9999999

    dl_props = []
    for fn in image_archive_fns:
        dl_props.append({
            "url": osp.join(data_url_base, "object_detection_images", split, fn + ".tar"),
            "destination": osp.join(images_root, split, fn),
            "force_download": force_download,
        })
    dl_pool = mp.Pool(download_parallel_num)
    results = dl_pool.map_async(_get_image_map_func, dl_props).get(download_timeout)
    dl_pool.close()

    # raise error if exists
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print i, dl_props[i]
            raise result

    return images_root, anno_path, annotations


def parse_action_annotation(in_path):
    import pandas as pd
    from datetime import datetime
    df = pd.read_csv(in_path)
    df = df.sort_values("uid")
    annotations = []
    for i, row in df.iterrows():
        for eval_key in ["all_nouns", "all_noun_classes"]:
            row[eval_key] = eval(row[eval_key])
        annotations.append(row.to_dict())
    return annotations


def get_action_videos(split="train", download_parallel_num=None, download_timeout=None, force_download=False):
    if split != "train":
        raise ValueError("split '%s' not available" % split)

    data_dir = download.get_dataset_directory(root)
    videos_root = osp.join(data_dir, "videos")
    annos_root = osp.join(data_dir, "annotations-{version}".format(version=version))
    anno_fn = "EPIC_{split}_action_labels.csv".format(split=split)

    # download annotation
    if not osp.exists(osp.join(annos_root, anno_fn)):
        download_file_path = utils.cached_download(anno_url)
        ext = osp.splitext(anno_url)[1]
        utils.extractall(download_file_path,
                         data_dir, ext)
    anno_path = osp.join(annos_root, anno_fn)
    annotations = parse_action_annotation(anno_path)

    # download videos
    if download_parallel_num is None:
        download_parallel_num = mp.cpu_count()
    if download_timeout is None:
        download_timeout = 9999999

    video_ids = set([a["video_id"] for a in annotations])
    dl_props = []
    for video_id in video_ids:
        pid = video_id.split("_")[0]
        dl_props.append({
            "url": osp.join(data_url_base, "videos", split, pid, video_id + ".MP4"),
            "destination": osp.join(videos_root, split, pid, video_id),
            "force_download": force_download,
            "extract": False,
        })
    dl_pool = mp.Pool(download_parallel_num)
    results = dl_pool.map_async(_get_image_map_func, dl_props).get(download_timeout)
    dl_pool.close()

    # raise error if exists
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print i, dl_props[i]
            raise result

    return videos_root, anno_path, annotations


def gen_labels(in_fn, out_fn, prefix, index_key, value_key):
    import pandas as pd
    df = pd.read_csv(in_fn)
    df = df[[index_key, value_key]].drop_duplicates().sort_values(index_key)

    fmtstr = "\n    ({0.%s}, '{0.%s}')," % (index_key, value_key)
    with open(out_fn, "w") as f:
        f.write("#!/usr/bin/env python\n")
        f.write("# -*- coding: utf-8 -*-\n\n\n")
        f.write("from collections import OrderedDict\n\n\n")
        f.write("%s_labels = OrderedDict((" % prefix)
        for index, row in df.iterrows():
            f.write(fmtstr.format(row))
        f.write("\n))")
        f.write("\n\n{0}_label_names = {0}_labels.values()\n".format(prefix))
    print "label dict was written to", out_fn


def gen_object_detection_label(out_fn="epic_kitchen_object_detection_labels.py"):
    data_dir = download.get_dataset_directory(root)
    images_root = osp.join(data_dir, "object_detection_images")
    annos_root = osp.join(data_dir, "annotations-{version}".format(version=version))
    anno_fn = "EPIC_noun_classes.csv"

    gen_labels(osp.join(annos_root, anno_fn),
               out_fn,
               "epic_kitchen_object_detection",
               "noun_id", "class_key")

def gen_action_labels(out_fn="epic_kitchen_action_labels.py"):
    data_dir = download.get_dataset_directory(root)
    images_root = osp.join(data_dir, "object_detection_images")
    annos_root = osp.join(data_dir, "annotations-{version}".format(version=version))
    anno_fn = "EPIC_verb_classes.csv"

    gen_labels(osp.join(annos_root, anno_fn),
               out_fn,
               "epic_kitchen_action",
               "verb_id", "class_key")


if __name__ == '__main__':
    # get_object_detection_images("train", 1)
    get_action_videos("train", 1)
    # gen_object_detection_label()
    # gen_action_labels()
