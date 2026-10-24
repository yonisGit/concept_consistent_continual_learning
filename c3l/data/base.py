"""
Dataset primitives shared by every benchmark.

Every benchmark is reduced to an :class:`ImageListDataset` (a list of
``ImageRecord``s + class names). Class-incremental splitting, transforms and
mask alignment are handled generically in ``scenario.py`` / ``transforms.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ImageRecord:
    path: str
    label: int                      # label in the *original* dataset label space
    extras: Dict[str, Any] = field(default_factory=dict)


class ImageListDataset(Dataset):
    """A list of image paths with integer labels; returns PIL RGB images."""

    def __init__(
        self,
        records: List[ImageRecord],
        class_names: List[str],
        name: str = "dataset",
        image_loader: Optional[Callable[[str], Image.Image]] = None,
    ):
        self.records = records
        self.class_names = class_names
        self.name = name
        self._loader = image_loader or default_loader

    def __len__(self) -> int:
        return len(self.records)

    @property
    def targets(self) -> List[int]:
        return [r.label for r in self.records]

    def load_image(self, idx: int) -> Image.Image:
        return self._loader(self.records[idx].path)

    def corrupt(self, idx: int, img: torch.Tensor) -> torch.Tensor:
        """Optional synthetic corruption applied on the [0,1] tensor (Split ImageNet-100C)."""
        std = self.records[idx].extras.get("gaussian_noise_std")
        if not std:
            return img
        g = torch.Generator().manual_seed(int(idx) * 7919 + 17)  # deterministic per sample
        return (img + torch.randn(img.shape, generator=g) * std).clamp(0.0, 1.0)

    def __getitem__(self, idx: int):
        return self.load_image(idx), self.records[idx].label

    def sample_key(self, idx: int) -> str:
        """Stable identifier used for caching masks / concepts."""
        return f"{self.name}::{os.path.normpath(self.records[idx].path)}"


def default_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


@dataclass
class DatasetSplits:
    """Train / test (/ optional validation) sets of one benchmark."""

    train: ImageListDataset
    test: ImageListDataset
    class_names: List[str]
    val: Optional[ImageListDataset] = None
    info: Dict[str, Any] = field(default_factory=dict)
    # Called once the class-incremental scenario is known (e.g. I-100C bias injection).
    post_scenario_hook: Optional[Callable[["DatasetSplits", Any, int], None]] = None


def stratified_split(
    records: List[ImageRecord], train_frac: float, seed: int
) -> tuple[List[ImageRecord], List[ImageRecord]]:
    """Per-class random split (used for datasets without an official split)."""
    rng = np.random.RandomState(seed)
    by_cls: Dict[int, List[ImageRecord]] = {}
    for r in records:
        by_cls.setdefault(r.label, []).append(r)
    train, test = [], []
    for c in sorted(by_cls):
        items = by_cls[c]
        perm = rng.permutation(len(items))
        n_tr = int(round(train_frac * len(items)))
        n_tr = min(max(n_tr, 1), max(len(items) - 1, 1))
        train += [items[i] for i in perm[:n_tr]]
        test += [items[i] for i in perm[n_tr:]]
    return train, test


def hold_out_per_class(
    records: List[ImageRecord], n_per_class: int, seed: int
) -> tuple[List[ImageRecord], List[ImageRecord]]:
    """Remove ``n_per_class`` images per class (held-out C_c^VAL for concept filtering)."""
    if n_per_class <= 0:
        return records, []
    rng = np.random.RandomState(seed + 12345)
    by_cls: Dict[int, List[ImageRecord]] = {}
    for r in records:
        by_cls.setdefault(r.label, []).append(r)
    keep, held = [], []
    for c in sorted(by_cls):
        items = by_cls[c]
        perm = rng.permutation(len(items))
        k = min(n_per_class, max(len(items) - 1, 0))
        held += [items[i] for i in perm[:k]]
        keep += [items[i] for i in perm[k:]]
    return keep, held


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG")


def scan_class_folders(root: str, class_dirs: Optional[List[str]] = None) -> tuple[List[ImageRecord], List[str]]:
    """``root/<class_dir>/<image>`` layout -> records, class dir names (sorted)."""
    if class_dirs is None:
        class_dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    records = []
    for label, d in enumerate(class_dirs):
        cdir = os.path.join(root, d)
        for dirpath, _, files in os.walk(cdir):
            for fn in sorted(files):
                if fn.endswith(IMG_EXTS):
                    records.append(ImageRecord(os.path.join(dirpath, fn), label))
    return records, class_dirs
