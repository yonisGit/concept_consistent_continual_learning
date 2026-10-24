"""
Benchmark builders (Sec. 4.1 and Appendix).

Expected layouts under ``data.root`` (download instructions are in README.md):

* CUB-200-2011 ...... ``CUB_200_2011/{images,segmentations}/ images.txt classes.txt
                        image_class_labels.txt train_test_split.txt``
* ImageNet-R ........ ``imagenet-r/<wnid>/*.jpg``
* ImageNet-A ........ ``imagenet-a/<wnid>/*.jpg``
* ObjectNet ......... ``objectnet-1.0/images/<folder>/*.png`` + ``objectnet-1.0/mappings/*.json``
* DomainNet ......... ``domainnet/<domain>/<class>/*.jpg`` + ``domainnet/<domain>_{train,test}.txt``
* PartImageNet ...... ``PartImageNet/images/{train,val,test}/*.JPEG`` + ``PartImageNet/annotations/{train,val,test}.json``
* Fitzpatrick17k .... ``fitzpatrick17k/fitzpatrick17k.csv`` + ``fitzpatrick17k/images/<md5hash>.jpg``
* ImageNet-1K ....... ``imagenet/{train,val}/<wnid>/*.JPEG`` (Split ImageNet-100C and the RQ4 subset)

Readable class names (needed for LLM concept discovery) for WordNet-id datasets
are read from ``<root>/LOC_synset_mapping.txt`` (the Kaggle ImageNet file) if
present, otherwise from NLTK WordNet, otherwise the wnid is used.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np

from c3l.data.base import (
    DatasetSplits,
    ImageListDataset,
    ImageRecord,
    default_loader,
    hold_out_per_class,
    scan_class_folders,
    stratified_split,
)

SPLIT_SEED = 1993  # fixed train/test split seed (independent of the run seed)


# --------------------------------------------------------------------------- #
# Class-name helpers
# --------------------------------------------------------------------------- #
_SYNSET_CACHE: Optional[Dict[str, str]] = None


def wnid_to_name(wnid: str, root: str) -> str:
    global _SYNSET_CACHE
    if _SYNSET_CACHE is None:
        _SYNSET_CACHE = {}
        path = os.path.join(root, "LOC_synset_mapping.txt")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    w, names = line.strip().split(" ", 1)
                    _SYNSET_CACHE[w] = names.split(",")[0].strip()
    if wnid in _SYNSET_CACHE:
        return _SYNSET_CACHE[wnid]
    try:
        from nltk.corpus import wordnet as wn  # optional

        syn = wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
        return syn.lemma_names()[0].replace("_", " ")
    except Exception:
        return wnid


def imagenet_wnids(root: str) -> List[str]:
    """The 1000 ImageNet-1K wnids in standard (sorted) index order."""
    train_dir = os.path.join(root, "imagenet", "train")
    if os.path.isdir(train_dir):
        return sorted(os.listdir(train_dir))
    path = os.path.join(root, "LOC_synset_mapping.txt")
    if os.path.exists(path):
        with open(path) as f:
            return [line.split(" ", 1)[0] for line in f]
    raise FileNotFoundError("Need imagenet/train or LOC_synset_mapping.txt to recover ImageNet-1K indices.")


# --------------------------------------------------------------------------- #
# CUB-200-2011
# --------------------------------------------------------------------------- #
def build_cub(cfg) -> DatasetSplits:
    root = os.path.join(cfg.data.root, "CUB_200_2011")

    def read(fn):
        with open(os.path.join(root, fn)) as f:
            return [line.strip().split(" ", 1) for line in f if line.strip()]

    images = dict(read("images.txt"))
    labels = {k: int(v) - 1 for k, v in read("image_class_labels.txt")}
    is_train = {k: v == "1" for k, v in read("train_test_split.txt")}
    classes = [v for _, v in read("classes.txt")]
    class_names = [c.split(".", 1)[-1].replace("_", " ") for c in classes]

    train, test = [], []
    for img_id, rel in images.items():
        seg = os.path.join(root, "segmentations", rel.rsplit(".", 1)[0] + ".png")
        rec = ImageRecord(os.path.join(root, "images", rel), labels[img_id], {"segmentation": seg})
        (train if is_train[img_id] else test).append(rec)
    train, val = hold_out_per_class(train, cfg.data.val_per_class, SPLIT_SEED)
    return DatasetSplits(
        ImageListDataset(train, class_names, "cub_train"),
        ImageListDataset(test, class_names, "cub_test"),
        class_names,
        val=ImageListDataset(val, class_names, "cub_val") if val else None,
        info={"has_segmentation": True},
    )


# --------------------------------------------------------------------------- #
# ImageNet-R / ImageNet-A (wnid folders, no official split -> 80/20)
# --------------------------------------------------------------------------- #
def _build_wnid_folder_dataset(cfg, folder: str, name: str) -> DatasetSplits:
    root = os.path.join(cfg.data.root, folder)
    records, wnids = scan_class_folders(root)
    class_names = [wnid_to_name(w, cfg.data.root) for w in wnids]
    train, test = stratified_split(records, cfg.data.train_test_split, SPLIT_SEED)
    train, val = hold_out_per_class(train, cfg.data.val_per_class, SPLIT_SEED)
    return DatasetSplits(
        ImageListDataset(train, class_names, f"{name}_train"),
        ImageListDataset(test, class_names, f"{name}_test"),
        class_names,
        val=ImageListDataset(val, class_names, f"{name}_val") if val else None,
        info={"wnids": wnids},
    )


def build_imagenet_r(cfg) -> DatasetSplits:
    return _build_wnid_folder_dataset(cfg, "imagenet-r", "imagenet_r")


def build_imagenet_a(cfg) -> DatasetSplits:
    return _build_wnid_folder_dataset(cfg, "imagenet-a", "imagenet_a")


# --------------------------------------------------------------------------- #
# ObjectNet (113 classes overlapping ImageNet-1K)
# --------------------------------------------------------------------------- #
def objectnet_loader(path: str):
    """ObjectNet images carry a 1-pixel red border that must be removed."""
    img = default_loader(path)
    w, h = img.size
    return img.crop((2, 2, w - 2, h - 2))


def _objectnet_meta(root: str):
    on_root = os.path.join(root, "objectnet-1.0")
    with open(os.path.join(on_root, "mappings", "folder_to_objectnet_label.json")) as f:
        folder_to_label = json.load(f)
    with open(os.path.join(on_root, "mappings", "objectnet_to_imagenet_1k.json")) as f:
        on_to_in = json.load(f)  # objectnet label -> "imagenet name; imagenet name"
    with open(os.path.join(on_root, "mappings", "imagenet_to_label_2012_v2")) as f:
        in_names = [line.strip() for line in f]
    name_to_in_idx = {n: i for i, n in enumerate(in_names)}
    folders = sorted(fd for fd, lbl in folder_to_label.items() if lbl in on_to_in)
    in_indices = []
    for fd in folders:
        names = [n.strip() for n in on_to_in[folder_to_label[fd]].split(";")]
        in_indices.append(sorted(name_to_in_idx[n] for n in names if n in name_to_in_idx))
    return on_root, folders, folder_to_label, in_indices


def build_objectnet(cfg) -> DatasetSplits:
    on_root, folders, folder_to_label, in_indices = _objectnet_meta(cfg.data.root)
    records, _ = scan_class_folders(os.path.join(on_root, "images"), folders)
    class_names = [folder_to_label[f] for f in folders]
    train, test = stratified_split(records, cfg.data.train_test_split, SPLIT_SEED)
    train, val = hold_out_per_class(train, cfg.data.val_per_class, SPLIT_SEED)
    mk = lambda recs, n: ImageListDataset(recs, class_names, n, image_loader=objectnet_loader)
    return DatasetSplits(
        mk(train, "objectnet_train"), mk(test, "objectnet_test"), class_names,
        val=mk(val, "objectnet_val") if val else None,
        info={"imagenet_indices": in_indices},
    )


# --------------------------------------------------------------------------- #
# DomainNet (345 classes x 6 domains, official split files)
# --------------------------------------------------------------------------- #
def build_domainnet(cfg) -> DatasetSplits:
    root = os.path.join(cfg.data.root, "domainnet")
    train, test = [], []
    class_names: Dict[int, str] = {}
    for dom in cfg.data.domainnet_domains:
        for split, bucket in (("train", train), ("test", test)):
            fn = os.path.join(root, f"{dom}_{split}.txt")
            with open(fn) as f:
                for line in f:
                    rel, lbl = line.strip().rsplit(" ", 1)
                    lbl = int(lbl)
                    class_names[lbl] = rel.split("/")[1].replace("_", " ")
                    bucket.append(ImageRecord(os.path.join(root, rel), lbl, {"domain": dom}))
    names = [class_names[i] for i in range(len(class_names))]
    train, val = hold_out_per_class(train, cfg.data.val_per_class, SPLIT_SEED)
    return DatasetSplits(
        ImageListDataset(train, names, "domainnet_train"),
        ImageListDataset(test, names, "domainnet_test"),
        names,
        val=ImageListDataset(val, names, "domainnet_val") if val else None,
    )


# --------------------------------------------------------------------------- #
# PartImageNet (Appendix "Performance without foundation models")
# --------------------------------------------------------------------------- #
def build_partimagenet(cfg) -> DatasetSplits:
    """
    PartImageNet's official splits are class-disjoint, so all splits are pooled, 150 of
    the 158 classes are kept (8 dropped at random) and each class is split 80/20.
    Expected layout: ``PartImageNet/images/<split>/<file>`` + ``PartImageNet/annotations/<split>.json``
    (COCO format with part polygons).
    """
    root = os.path.join(cfg.data.root, "PartImageNet")
    raw = []
    for split in ("train", "val", "test"):
        ann_path = os.path.join(root, "annotations", f"{split}.json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            coco = json.load(f)
        for im in coco["images"]:
            wnid = os.path.basename(im["file_name"]).split("_")[0]
            raw.append((os.path.join(root, "images", split, im["file_name"]), wnid, im["id"], ann_path))
    wnids = sorted({w for _, w, _, _ in raw})
    rng = np.random.RandomState(SPLIT_SEED)
    n_keep = cfg.data.partimagenet["num_classes"]
    keep = sorted(rng.choice(len(wnids), size=min(n_keep, len(wnids)), replace=False))
    kept = [wnids[i] for i in keep]
    w2l = {w: i for i, w in enumerate(kept)}
    records = [ImageRecord(p, w2l[w], {"coco_image_id": iid, "coco_ann": ap}) for p, w, iid, ap in raw if w in w2l]
    train, test = stratified_split(records, cfg.data.train_test_split, SPLIT_SEED)
    names = [wnid_to_name(w, cfg.data.root) for w in kept]
    return DatasetSplits(
        ImageListDataset(train, names, "partimagenet_train"),
        ImageListDataset(test, names, "partimagenet_test"),
        names,
        info={"wnids": kept, "has_part_masks": True},
    )


# --------------------------------------------------------------------------- #
# Fitzpatrick17k (spurious-correlation protocol, RQ2)
# --------------------------------------------------------------------------- #
def build_fitz(cfg) -> DatasetSplits:
    root = os.path.join(cfg.data.root, "fitzpatrick17k")
    rows = []
    with open(os.path.join(root, "fitzpatrick17k.csv")) as f:
        for row in csv.DictReader(f):
            path = os.path.join(root, "images", row["md5hash"] + ".jpg")
            if os.path.exists(path):
                rows.append((path, row["label"], int(row.get("fitzpatrick_scale", -1) or -1)))
    names = sorted({r[1] for r in rows})
    n2l = {n: i for i, n in enumerate(names)}
    records = [ImageRecord(p, n2l[n], {"fitzpatrick": s}) for p, n, s in rows]
    train, test = stratified_split(records, cfg.data.train_test_split, SPLIT_SEED)
    return DatasetSplits(
        ImageListDataset(train, names, "fitz_train"),
        ImageListDataset(test, names, "fitz_test"),
        names,
    )


# --------------------------------------------------------------------------- #
# Split ImageNet-100C (Lee et al. 2024)
# --------------------------------------------------------------------------- #
def build_imagenet100c(cfg) -> DatasetSplits:
    """
    100 random ImageNet-1K classes in 10 tasks. In every task one randomly chosen
    class gets ``skew_ratio`` of its training images corrupted with Gaussian noise
    while every other class gets ``base_corrupt_rate`` (1%) -> the noise becomes a
    spurious cue for that class. Test images are clean.
    """
    root = cfg.data.root
    wnids_all = imagenet_wnids(root)
    rng = np.random.RandomState(SPLIT_SEED)
    chosen = sorted(rng.choice(len(wnids_all), size=100, replace=False))
    wnids = [wnids_all[i] for i in chosen]
    train, _ = scan_class_folders(os.path.join(root, "imagenet", "train"), wnids)
    test, _ = scan_class_folders(os.path.join(root, "imagenet", "val"), wnids)
    names = [wnid_to_name(w, root) for w in wnids]
    params = cfg.data.imagenet100c

    def inject_bias(splits: DatasetSplits, scenario, seed: int) -> None:
        r = np.random.RandomState(seed + 777)
        biased_orig = set()
        for t in range(scenario.num_tasks):
            new_label = r.choice(scenario.task_classes(t))
            biased_orig.add(scenario.class_order[new_label])
        for rec in splits.train.records:
            rate = params["skew_ratio"] if rec.label in biased_orig else params["base_corrupt_rate"]
            if r.rand() < rate:
                rec.extras["gaussian_noise_std"] = params["noise_std"]
        splits.info["biased_classes"] = sorted(int(c) for c in biased_orig)

    return DatasetSplits(
        ImageListDataset(train, names, "imagenet100c_train"),
        ImageListDataset(test, names, "imagenet100c_test"),
        names,
        post_scenario_hook=inject_bias,
    )


# --------------------------------------------------------------------------- #
# RQ4 robustness: 500 ImageNet classes x 5 images (+40 validation images/class)
# --------------------------------------------------------------------------- #
def build_imagenet_subset(cfg, n_classes: int = 500, n_per_class: int = 5, n_val: int = 40) -> DatasetSplits:
    """Labels are ImageNet-1K indices so the pretrained 1000-way head is reused."""
    root = cfg.data.root
    wnids = imagenet_wnids(root)
    names = [wnid_to_name(w, root) for w in wnids]
    rng = np.random.RandomState(SPLIT_SEED + cfg.seed)
    chosen = sorted(rng.choice(len(wnids), size=n_classes, replace=False))
    train, val = [], []
    for c in chosen:
        cdir = os.path.join(root, "imagenet", "train", wnids[c])
        files = sorted(os.listdir(cdir))
        perm = rng.permutation(len(files))
        train += [ImageRecord(os.path.join(cdir, files[i]), int(c)) for i in perm[:n_per_class]]
        val += [ImageRecord(os.path.join(cdir, files[i]), int(c)) for i in perm[n_per_class:n_per_class + n_val]]
    return DatasetSplits(
        ImageListDataset(train, names, "imagenet_subset_train"),
        ImageListDataset([], names, "imagenet_subset_test"),
        names,
        val=ImageListDataset(val, names, "imagenet_subset_val"),
        info={"chosen_classes": [int(c) for c in chosen], "wnids": wnids},
    )


def build_imagenet_a_eval(cfg) -> tuple[ImageListDataset, List[List[int]]]:
    """Whole ImageNet-A as an evaluation set; returns per-class allowed ImageNet indices."""
    root = os.path.join(cfg.data.root, "imagenet-a")
    records, wnids = scan_class_folders(root)
    all_wnids = imagenet_wnids(cfg.data.root)
    in_idx = [[all_wnids.index(w)] for w in wnids]
    names = [wnid_to_name(w, cfg.data.root) for w in wnids]
    return ImageListDataset(records, names, "imagenet_a_eval"), in_idx


def build_objectnet_eval(cfg) -> tuple[ImageListDataset, List[List[int]]]:
    on_root, folders, folder_to_label, in_indices = _objectnet_meta(cfg.data.root)
    records, _ = scan_class_folders(os.path.join(on_root, "images"), folders)
    names = [folder_to_label[f] for f in folders]
    return ImageListDataset(records, names, "objectnet_eval", image_loader=objectnet_loader), in_indices


BUILDERS = {
    "cub": build_cub,
    "imagenet_r": build_imagenet_r,
    "imagenet_a": build_imagenet_a,
    "objectnet": build_objectnet,
    "domainnet": build_domainnet,
    "partimagenet": build_partimagenet,
    "fitz": build_fitz,
    "imagenet100c": build_imagenet100c,
}


def build_dataset(cfg) -> DatasetSplits:
    name = cfg.data.name
    if name not in BUILDERS:
        raise ValueError(f"Unknown dataset {name}. Choose from {sorted(BUILDERS)}")
    return BUILDERS[name](cfg)
