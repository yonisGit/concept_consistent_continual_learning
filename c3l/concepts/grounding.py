"""
Spatial concept grounding (Sec. 3.2, Eq. 1):

    M_concept(x) = U_{psi in Psi_y*}  U_{i : s_i(psi) >= tau_conf}  B_i(psi)

Grounders return *detections* (mask, score, concept index); ``concept_mask``
aggregates them with the confidence threshold. Implemented grounders:

* ``SAM3Grounder``        SAM 3 via Ultralytics, all concepts in a single batched query (default)
* ``GroundedSAMGrounder`` Grounding-DINO boxes -> SAM masks, one query per concept (ablation)
* ``ObjectGrounder``      SAM 3 prompted with the class name -> object-level masks (ablation)
* ``PartImageNetGTGrounder`` ground-truth part masks, no foundation model (C3L-PG, Appendix)

``MaskProvider`` adds on-disk caching (masks are computed once, offline, in the
base frame).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
from PIL import Image

from c3l.data.base import ImageListDataset
from c3l.data.transforms import to_base_pil
from c3l.utils import get_logger

logger = get_logger()


@dataclass
class Detections:
    masks: np.ndarray        # (N, H, W) bool
    scores: np.ndarray       # (N,)
    concept_ids: np.ndarray  # (N,) index into the queried concept list

    @staticmethod
    def empty(h: int, w: int) -> "Detections":
        return Detections(np.zeros((0, h, w), bool), np.zeros(0, np.float32), np.zeros(0, np.int64))


def union_mask(det: Detections, tau: float, h: int, w: int) -> np.ndarray:
    keep = det.scores >= tau
    if not keep.any():
        return np.zeros((h, w), np.uint8)
    return det.masks[keep].any(0).astype(np.uint8)


class Grounder:
    confidence_threshold: float = 0.3

    def detect(self, image: Image.Image, prompts: List[str]) -> Detections:
        raise NotImplementedError

    def concept_mask(self, image: Image.Image, concepts: List[str]) -> np.ndarray:
        w, h = image.size
        if not concepts:
            return np.zeros((h, w), np.uint8)
        return union_mask(self.detect(image, concepts), self.confidence_threshold, h, w)


class SAM3Grounder(Grounder):
    """SAM 3 with batched text prompts (Ultralytics implementation)."""

    def __init__(self, weights: str = "sam3.pt", confidence_threshold: float = 0.3, half: bool = True):
        from ultralytics.models.sam import SAM3SemanticPredictor

        self.confidence_threshold = confidence_threshold
        # Let the predictor return everything above a low floor; thresholding is ours (tau_conf).
        overrides = dict(conf=min(0.05, confidence_threshold), task="segment", mode="predict",
                         model=weights, half=half, verbose=False, save=False)
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def detect(self, image: Image.Image, prompts: List[str]) -> Detections:
        w, h = image.size
        arr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()  # Ultralytics expects BGR numpy
        self.predictor.set_image(arr)
        results = self.predictor(text=list(prompts))
        if not results or results[0].masks is None or len(results[0].masks) == 0:
            return Detections.empty(h, w)
        r = results[0]
        masks = r.masks.data.cpu().numpy() > 0.5
        if masks.shape[-2:] != (h, w):
            t = torch.from_numpy(masks[:, None].astype(np.float32))
            masks = torch.nn.functional.interpolate(t, size=(h, w), mode="nearest")[:, 0].numpy() > 0.5
        if r.boxes is not None and len(r.boxes) == len(masks):
            scores = r.boxes.conf.cpu().numpy().astype(np.float32)
            cls = r.boxes.cls.cpu().numpy().astype(np.int64)
        else:  # defensive: no per-mask metadata -> treat as confident, prompt unknown (single prompt only)
            scores = np.ones(len(masks), np.float32)
            cls = np.zeros(len(masks), np.int64)
        return Detections(masks, scores, cls)


class GroundedSAMGrounder(Grounder):
    """Grounding-DINO (HF transformers) boxes + SAM masks; concepts are queried sequentially."""

    def __init__(self, dino_id: str = "IDEA-Research/grounding-dino-base", sam_weights: str = "sam2.1_b.pt",
                 confidence_threshold: float = 0.3, text_threshold: float = 0.25, device: str = "cuda"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from ultralytics import SAM

        self.confidence_threshold = confidence_threshold
        self.text_threshold = text_threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device).eval()
        self.sam = SAM(sam_weights)

    @torch.no_grad()
    def detect(self, image: Image.Image, prompts: List[str]) -> Detections:
        w, h = image.size
        all_m, all_s, all_c = [], [], []
        for ci, concept in enumerate(prompts):
            inputs = self.processor(images=image, text=concept.lower().rstrip(".") + ".", return_tensors="pt").to(self.device)
            out = self.dino(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=min(0.05, self.confidence_threshold),
                text_threshold=self.text_threshold, target_sizes=[(h, w)])[0]
            if len(res["boxes"]) == 0:
                continue
            boxes = res["boxes"].cpu().numpy()
            seg = self.sam(np.asarray(image)[:, :, ::-1].copy(), bboxes=boxes.tolist(), verbose=False)[0]
            if seg.masks is None:
                continue
            m = seg.masks.data.cpu().numpy() > 0.5
            all_m.append(m)
            all_s.append(res["scores"].cpu().numpy()[: len(m)])
            all_c.append(np.full(len(m), ci))
        if not all_m:
            return Detections.empty(h, w)
        return Detections(np.concatenate(all_m), np.concatenate(all_s).astype(np.float32), np.concatenate(all_c))


class ObjectGrounder(Grounder):
    """Object-level masks: SAM 3 prompted with the class name instead of the concept list."""

    def __init__(self, sam3: SAM3Grounder):
        self.sam3 = sam3
        self.confidence_threshold = sam3.confidence_threshold

    def detect(self, image: Image.Image, prompts: List[str]) -> Detections:
        return self.sam3.detect(image, prompts)


class PartImageNetGTGrounder:
    """Union of ground-truth part masks (C3L-PG). Requires pycocotools."""

    def __init__(self):
        from pycocotools.coco import COCO  # noqa: F401

        self._cocos = {}

    def mask_for_record(self, record) -> np.ndarray:
        from pycocotools.coco import COCO

        ann_path = record.extras["coco_ann"]
        if ann_path not in self._cocos:
            self._cocos[ann_path] = COCO(ann_path)
        coco = self._cocos[ann_path]
        info = coco.loadImgs(record.extras["coco_image_id"])[0]
        m = np.zeros((info["height"], info["width"]), np.uint8)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=info["id"])):
            m |= coco.annToMask(ann).astype(np.uint8)
        return m


def build_grounder(cfg) -> Grounder | PartImageNetGTGrounder:
    g = cfg.grounding
    if g.method in ("sam3", "object"):
        sam3 = SAM3Grounder(g.sam3_weights, g.confidence_threshold, g.half)
        return sam3 if g.method == "sam3" else ObjectGrounder(sam3)
    if g.method == "grounded_sam":
        return GroundedSAMGrounder(g.grounding_dino_id, g.sam_weights, g.confidence_threshold)
    if g.method == "gt_parts":
        return PartImageNetGTGrounder()
    raise ValueError(f"Unknown grounding method {g.method}")


# --------------------------------------------------------------------------- #
# Cached mask provider
# --------------------------------------------------------------------------- #
class MaskProvider:
    """
    ``provider(idx) -> (base, base) uint8`` concept mask for sample ``idx`` of ``dataset``.

    ``prompt_fn(orig_label) -> list[str]`` gives the concept list (or ``[class_name]``
    for object-level grounding). Masks are cached as packed bits under
    ``cache_dir/<config-hash>/``.
    """

    def __init__(self, dataset: ImageListDataset, grounder, prompt_fn: Callable[[int], List[str]],
                 base_size: int, cache_dir: str, tag: str):
        self.dataset = dataset
        self.grounder = grounder
        self.prompt_fn = prompt_fn
        self.base_size = base_size
        self.dir = os.path.join(cache_dir, tag)
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, idx: int) -> str:
        key = hashlib.md5(self.dataset.sample_key(idx).encode()).hexdigest()
        return os.path.join(self.dir, key + ".npy")

    def compute(self, idx: int) -> np.ndarray:
        rec = self.dataset.records[idx]
        if isinstance(self.grounder, PartImageNetGTGrounder):
            m = self.grounder.mask_for_record(rec)
            m = np.asarray(Image.fromarray(m * 255).resize((self.base_size, self.base_size), Image.NEAREST)) > 0
            return m.astype(np.uint8)
        img = to_base_pil(self.dataset.load_image(idx), self.base_size)
        return self.grounder.concept_mask(img, self.prompt_fn(rec.label))

    def get_numpy(self, idx: int) -> np.ndarray:
        path = self._path(idx)
        if os.path.exists(path):
            packed = np.load(path)
            m = np.unpackbits(packed)[: self.base_size * self.base_size].reshape(self.base_size, self.base_size)
        else:
            m = self.compute(idx).astype(np.uint8)
            np.save(path, np.packbits(m.reshape(-1)))
        return m

    def __call__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.get_numpy(idx).astype(np.uint8))

    def precompute(self, indices) -> None:
        from tqdm import tqdm

        for i in tqdm(list(indices), desc="Grounding concepts", leave=False):
            if not os.path.exists(self._path(i)):
                self.get_numpy(i)


def grounding_tag(cfg, concept_source: str) -> str:
    """Cache key: dataset + grounding method + threshold + concept bank."""
    g = cfg.grounding
    raw = f"{cfg.data.name}|{g.method}|{g.confidence_threshold}|{cfg.data.base_size}|{concept_source}"
    return f"{cfg.data.name}_{g.method}_tau{g.confidence_threshold}_" + hashlib.md5(raw.encode()).hexdigest()[:8]
