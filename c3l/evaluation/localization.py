"""
Semantic localisation of explanations (RQ2, Appendix "Explanation Quality"),
following Chefer et al. (2021): saliency maps are binarised at their mean
value and compared with ground-truth object masks (CUB segmentations).

* pixAcc : correctly classified pixels / all pixels
* mIoU   : mean over {background, foreground} of dataset-level IoU (Chefer et al.)
* mAP    : Chefer et al.'s per-image average precision of the raw saliency scores
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from c3l.data.transforms import center_crop
from c3l.explanations.factory import Explainer


def load_cub_segmentation(path: str, base_size: int, crop_size: int) -> torch.Tensor:
    m = Image.open(path).convert("L").resize((base_size, base_size), Image.NEAREST)
    m = torch.from_numpy((np.asarray(m) > 127).astype(np.uint8))
    return center_crop(m, crop_size)


class LocalizationMeter:
    def __init__(self):
        self.inter = np.zeros(2)
        self.union = np.zeros(2)
        self.correct = 0
        self.pixels = 0
        self.ious = []
        self.pixel_aps = []

    def update(self, sal: torch.Tensor, gt: torch.Tensor) -> None:
        """sal, gt: (B,H,W)."""
        from sklearn.metrics import average_precision_score

        sal = sal.detach().float().cpu()
        gt = gt.bool().cpu()
        thr = sal.flatten(1).mean(1).view(-1, 1, 1)
        pred = sal > thr
        for p, g, s in zip(pred, gt, sal):
            self.correct += int((p == g).sum())
            self.pixels += g.numel()
            for c, (pc, gc) in enumerate(((~p, ~g), (p, g))):
                self.inter[c] += int((pc & gc).sum())
                self.union[c] += int((pc | gc).sum())
            u = int((p | g).sum())
            self.ious.append(int((p & g).sum()) / u if u else 0.0)
            if g.any() and (~g).any():
                self.pixel_aps.append(average_precision_score(g.view(-1).numpy(), s.view(-1).numpy()))

    def compute(self) -> Dict[str, float]:
        ious = np.asarray(self.ious)
        return {
            "mIoU": float(100 * np.mean(self.inter / np.maximum(self.union, 1))),
            "fg_IoU": float(100 * ious.mean()) if len(ious) else 0.0,
            "pixAcc": float(100 * self.correct / max(self.pixels, 1)),
            "mAP": float(100 * np.mean(self.pixel_aps)) if self.pixel_aps else 0.0,
        }


def evaluate_localization(model, loader, explainer: Explainer, gt_fn, device,
                          max_batches: Optional[int] = None) -> Dict[str, float]:
    """
    ``loader`` yields TaskDataset batches (with ``index``); ``gt_fn(index) -> (crop,crop)`` mask.
    Explanations are computed w.r.t. the ground-truth class.
    """
    model.eval()
    meter = LocalizationMeter()
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = batch["image"].to(device), batch["label"].to(device)
        sal = explainer(model, x, y, create_graph=False)
        gt = torch.stack([gt_fn(int(i)) for i in batch["index"]])
        meter.update(sal, gt)
    return meter.compute()
