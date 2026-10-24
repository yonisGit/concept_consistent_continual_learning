"""
Optional concept filtering (Sec. 3.2, Appendix C "Concept Validation Details").

For every class ``c`` and held-out images ``C_c^VAL``:

* Occurrence rate   nu_occ(psi) = |{x : Seg_psi(x) != {}}| / |C_c^VAL|      (per concept)
* Spatial coverage  nu_cov      = mean_x IoU( U_psi Seg_psi(x), Obj(x) )     (per concept set)

where ``Obj(x)`` is the SAM 3 segment obtained with the class name as prompt.
Concepts with ``nu_occ < tau_occ`` are discarded (default 0.15).

The paper defines coverage on the *union* of concept masks, i.e. per class,
while also saying concepts failing "either threshold" are dropped. We therefore
support two modes (``concepts.coverage_mode``):

* ``set_iou`` (default, literal reading): coverage is computed for the retained
  set. If it is below tau_cov, pruned concepts are re-added greedily (highest
  occurrence first) while that increases coverage, so the set is not "too sparse".
* ``concept_in_object``: per-concept coverage = mean fraction of the concept's
  mask lying on the object. Concepts below tau_cov (i.e. grounded mostly on the
  background) are discarded.

Both modes record all statistics so Table "concept_stats" can be reproduced.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from c3l.concepts.grounding import Detections, union_mask
from c3l.data.base import ImageListDataset
from c3l.data.transforms import to_base_pil
from c3l.utils import get_logger

logger = get_logger()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


class ConceptValidator:
    def __init__(self, grounder, occurrence_threshold: float = 0.15, coverage_threshold: float = 0.2,
                 coverage_mode: str = "set_iou", base_size: int = 256):
        self.grounder = grounder
        self.tau_occ = occurrence_threshold
        self.tau_cov = coverage_threshold
        self.mode = coverage_mode
        self.base_size = base_size

    def validate_class(self, class_name: str, concepts: List[str], dataset: ImageListDataset,
                       indices: List[int]) -> Dict:
        tau = self.grounder.confidence_threshold
        n = len(indices)
        if n == 0 or not concepts:
            return {"kept": list(concepts), "occurrence": {}, "coverage": None, "n_images": n}
        per_img: List[Dict[int, np.ndarray]] = []
        objs: List[np.ndarray] = []
        occ = np.zeros(len(concepts))
        in_obj = [[] for _ in concepts]
        for idx in indices:
            img = to_base_pil(dataset.load_image(idx), self.base_size)
            det: Detections = self.grounder.detect(img, concepts)
            obj = union_mask(self.grounder.detect(img, [class_name]), tau, self.base_size, self.base_size).astype(bool)
            masks = {}
            for k in range(len(concepts)):
                sel = (det.concept_ids == k) & (det.scores >= tau)
                if sel.any():
                    m = det.masks[sel].any(0)
                    masks[k] = m
                    occ[k] += 1
                    if m.sum() > 0:
                        in_obj[k].append(float(np.logical_and(m, obj).sum() / m.sum()))
            per_img.append(masks)
            objs.append(obj)
        occ_rate = occ / n
        kept = [k for k in range(len(concepts)) if occ_rate[k] >= self.tau_occ]

        def set_cov(ks: List[int]) -> float:
            vals = []
            for masks, obj in zip(per_img, objs):
                u = np.zeros_like(obj)
                for k in ks:
                    if k in masks:
                        u |= masks[k]
                vals.append(_iou(u, obj))
            return float(np.mean(vals)) if vals else 0.0

        if self.mode == "concept_in_object":
            cov_k = {k: (float(np.mean(in_obj[k])) if in_obj[k] else 0.0) for k in range(len(concepts))}
            kept = [k for k in kept if cov_k[k] >= self.tau_cov]
            coverage = set_cov(kept)
        else:  # set_iou
            coverage = set_cov(kept)
            if coverage < self.tau_cov:
                pruned = sorted(set(range(len(concepts))) - set(kept), key=lambda k: -occ_rate[k])
                for k in pruned:
                    c_new = set_cov(kept + [k])
                    if c_new > coverage:
                        kept.append(k)
                        coverage = c_new
                    if coverage >= self.tau_cov:
                        break
                if coverage < self.tau_cov:
                    logger.warning(f"{class_name}: concept-set coverage {coverage:.2f} < {self.tau_cov}")
        kept = sorted(kept)
        return {
            "kept": [concepts[k] for k in kept],
            "occurrence": {concepts[k]: float(occ_rate[k]) for k in range(len(concepts))},
            "coverage": coverage,
            "n_images": n,
        }


def validate_bank(bank: Dict[str, Dict], validator: ConceptValidator, dataset: ImageListDataset,
                  images_per_class: int, seed: int = 0) -> Dict[str, Dict]:
    """Adds ``validated`` and ``validation`` fields to every bank entry."""
    rng = np.random.RandomState(seed)
    by_cls: Dict[int, List[int]] = {}
    for i, r in enumerate(dataset.records):
        by_cls.setdefault(r.label, []).append(i)
    name_to_label = {n.lower(): i for i, n in enumerate(dataset.class_names)}
    for key, entry in bank.items():
        label = int(key) if key.isdigit() else name_to_label.get(entry["name"].lower())
        if label is None or label not in by_cls:
            entry["validated"] = entry["concepts"]
            continue
        idx = by_cls[label]
        idx = list(rng.choice(idx, size=min(images_per_class, len(idx)), replace=False))
        res = validator.validate_class(entry["name"], entry["concepts"], dataset, idx)
        entry["validated"] = res["kept"]
        entry["validation"] = {k: v for k, v in res.items() if k != "kept"}
        logger.info(f"{entry['name']}: kept {len(res['kept'])}/{len(entry['concepts'])} "
                    f"(coverage={res['coverage']})")
    return bank


def bank_statistics(bank: Dict[str, Dict]) -> Dict[str, float]:
    """Generated / retained / avg occurrence / avg coverage (Table concept_stats)."""
    gen = [len(e["concepts"]) for e in bank.values()]
    ret = [len(e.get("validated", e["concepts"])) for e in bank.values()]
    occ, cov = [], []
    for e in bank.values():
        v = e.get("validation")
        if not v:
            continue
        occ += [v["occurrence"][c] for c in e.get("validated", []) if c in v["occurrence"]]
        if v.get("coverage") is not None:
            cov.append(v["coverage"])
    return {
        "generated": float(np.mean(gen)) if gen else 0.0,
        "retained": float(np.mean(ret)) if ret else 0.0,
        "avg_occurrence": float(np.mean(occ)) if occ else 0.0,
        "avg_coverage": float(np.mean(cov)) if cov else 0.0,
    }
