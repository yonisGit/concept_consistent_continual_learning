"""
Geometry-aligned transforms.

C3L needs pixel-accurate alignment between an image, its concept mask
``M_concept`` and its reference saliency ``S_ref``. We therefore define one
canonical *base frame*: every image is resized to ``base_size x base_size``.
Concept masks are computed (and cached) in this frame. Training crops/flips are
applied jointly to the image and its mask; evaluation and replay samples use the
deterministic centre crop.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def to_base_tensor(img: Image.Image, base_size: int) -> torch.Tensor:
    """PIL -> float tensor in [0,1], shape (3, base, base)."""
    img = img.resize((base_size, base_size), Image.BILINEAR)
    return TF.to_tensor(img)


def to_base_pil(img: Image.Image, base_size: int) -> Image.Image:
    return img.resize((base_size, base_size), Image.BILINEAR)


def resize_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest-neighbour resize of a (H,W) {0,1} mask to (size,size)."""
    if mask.shape[-1] == size and mask.shape[-2] == size:
        return mask
    m = F.interpolate(mask[None, None].float(), size=(size, size), mode="nearest")[0, 0]
    return m.to(mask.dtype)


def center_crop_offsets(base: int, crop: int) -> Tuple[int, int]:
    off = (base - crop) // 2
    return off, off


def center_crop(t: torch.Tensor, crop: int) -> torch.Tensor:
    """Centre-crop the last two dims."""
    h, w = t.shape[-2:]
    top, left = (h - crop) // 2, (w - crop) // 2
    return t[..., top:top + crop, left:left + crop]


class PairedTrainTransform:
    """Random crop + horizontal flip applied identically to image and optional mask."""

    def __init__(self, crop: int, flip: bool = True, scale: Tuple[float, float] = (1.0, 1.0)):
        self.crop = crop
        self.flip = flip
        self.scale = scale  # (1,1) -> plain random crop; <1 -> random-resized-crop style

    def __call__(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None):
        _, h, w = img.shape
        s = random.uniform(*self.scale)
        side = max(self.crop, int(round(min(h, w) * s))) if s < 1.0 else self.crop
        side = min(side, h, w)
        top = random.randint(0, h - side)
        left = random.randint(0, w - side)
        img = img[:, top:top + side, left:left + side]
        if mask is not None:
            mask = mask[top:top + side, left:left + side]
        if side != self.crop:
            img = F.interpolate(img[None], size=(self.crop, self.crop), mode="bilinear", align_corners=False)[0]
            if mask is not None:
                mask = resize_mask(mask, self.crop)
        if self.flip and random.random() < 0.5:
            img = img.flip(-1)
            if mask is not None:
                mask = mask.flip(-1)
        return img, mask


class EvalTransform:
    def __init__(self, crop: int):
        self.crop = crop

    def __call__(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None):
        img = center_crop(img, self.crop)
        if mask is not None:
            mask = center_crop(mask, self.crop)
        return img, mask
