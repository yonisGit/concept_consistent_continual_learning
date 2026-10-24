"""
Class-incremental scenario (Sec. 3.1 / 4.1).

``T`` tasks with disjoint, equally sized class sets. The class order is a
random permutation controlled by the run seed (the paper averages over 5 seeds
*and* task orderings). Labels are remapped so that task ``t`` owns the
contiguous label range ``[t*K, (t+1)*K)`` - this makes logit masking for
"classes seen so far" trivial. Task boundaries are known at train time but not
at test time (single head over all seen classes).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from c3l.data.base import DatasetSplits, ImageListDataset
from c3l.data.transforms import EvalTransform, PairedTrainTransform, to_base_tensor
from c3l.utils import normalize_image


class ClassIncrementalScenario:
    def __init__(self, splits: DatasetSplits, num_tasks: int, seed: int = 0, shuffle: bool = True):
        self.splits = splits
        self.num_tasks = num_tasks
        n_cls = len(splits.class_names)
        order = np.arange(n_cls)
        if shuffle:
            order = np.random.RandomState(seed).permutation(n_cls)
        if n_cls % num_tasks != 0:
            # Drop the remainder so all tasks have the same number of classes.
            order = order[: n_cls - (n_cls % num_tasks)]
        self.class_order: List[int] = [int(c) for c in order]       # new label -> original label
        self.orig_to_new: Dict[int, int] = {c: i for i, c in enumerate(self.class_order)}
        self.num_classes = len(self.class_order)
        self.classes_per_task = self.num_classes // num_tasks
        self.class_names = [splits.class_names[c] for c in self.class_order]

        self._train_idx = self._index_by_task(splits.train)
        self._test_idx = self._index_by_task(splits.test)
        self._val_idx = self._index_by_task(splits.val) if splits.val is not None else None

        if splits.post_scenario_hook is not None:
            splits.post_scenario_hook(splits, self, seed)

    # ------------------------------------------------------------------ #
    def _index_by_task(self, ds: ImageListDataset) -> List[List[int]]:
        per_task: List[List[int]] = [[] for _ in range(self.num_tasks)]
        for i, r in enumerate(ds.records):
            new = self.orig_to_new.get(r.label)
            if new is None:
                continue
            per_task[new // self.classes_per_task].append(i)
        return per_task

    def task_classes(self, t: int) -> List[int]:
        k = self.classes_per_task
        return list(range(t * k, (t + 1) * k))

    def num_seen_classes(self, t: int) -> int:
        return (t + 1) * self.classes_per_task

    def task_of_label(self, label: int) -> int:
        return label // self.classes_per_task

    def train_indices(self, t: int) -> List[int]:
        return self._train_idx[t]

    def test_indices(self, t: int) -> List[int]:
        return self._test_idx[t]

    def val_indices(self, t: int) -> List[int]:
        if self._val_idx is None:
            return []
        return self._val_idx[t]

    def new_label(self, orig_label: int) -> int:
        return self.orig_to_new[orig_label]

    def class_name_of(self, new_label: int) -> str:
        return self.class_names[new_label]


class TaskDataset(Dataset):
    """
    A view of a base dataset restricted to ``indices``, returning dicts::

        {"image": normalised (3,crop,crop), "label": new label, "index": base index,
         "mask": (crop,crop) uint8 concept mask (only if ``mask_fn`` is given)}

    ``mask_fn(base_index) -> (base,base) uint8`` supplies concept masks in the
    base frame (used by C3L-NoReplay / C3L++ where current-task samples are regularised).
    """

    def __init__(
        self,
        base: ImageListDataset,
        indices: List[int],
        label_map: Dict[int, int],
        train: bool,
        base_size: int = 256,
        crop_size: int = 224,
        mask_fn: Optional[Callable[[int], torch.Tensor]] = None,
        flip: bool = True,
    ):
        self.base = base
        self.indices = list(indices)
        self.label_map = label_map
        self.base_size = base_size
        self.mask_fn = mask_fn
        self.tf = PairedTrainTransform(crop_size, flip=flip) if train else EvalTransform(crop_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img = to_base_tensor(self.base.load_image(idx), self.base_size)
        img = self.base.corrupt(idx, img)
        mask = self.mask_fn(idx) if self.mask_fn is not None else None
        img, mask = self.tf(img, mask)
        out = {
            "image": normalize_image(img[None])[0],
            "label": self.label_map[self.base.records[idx].label],
            "index": idx,
        }
        if mask is not None:
            out["mask"] = mask.to(torch.uint8)
        return out


def load_base_crop(base: ImageListDataset, idx: int, base_size: int, crop_size: int) -> torch.Tensor:
    """Deterministic (centre-cropped, un-normalised, [0,1]) image used for replay storage."""
    img = to_base_tensor(base.load_image(idx), base_size)
    img = base.corrupt(idx, img)
    img, _ = EvalTransform(crop_size)(img)
    return img
