"""
Appendix "Prompt Complexity Robustness Analysis": SAM 3 vs Grounded-SAM on verbose concepts.

50 random CUB classes, their concepts with more than two words, 10 random images
per class; every concept is used as a prompt on every image of its class.

* hit rate        : % of prompts yielding a mask with confidence >= 0.3
* avg confidence  : mean confidence over all predicted masks

    python scripts/prompt_complexity.py --config configs/cub_vit.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c3l.concepts.discovery import ConceptLookup, load_concept_bank  # noqa: E402
from c3l.concepts.grounding import GroundedSAMGrounder, SAM3Grounder  # noqa: E402
from c3l.config import load_config  # noqa: E402
from c3l.data.datasets import build_dataset  # noqa: E402
from c3l.data.transforms import to_base_pil  # noqa: E402
from c3l.utils import save_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--classes", type=int, default=50)
    p.add_argument("--images", type=int, default=10)
    a = p.parse_args()
    cfg = load_config(a.config)
    splits = build_dataset(cfg)
    ds = splits.test
    lookup = ConceptLookup(load_concept_bank(cfg.concepts.bank_path), splits.class_names)
    rng = np.random.RandomState(cfg.seed)
    classes = rng.choice(len(splits.class_names), size=a.classes, replace=False)
    by_cls = {}
    for i, r in enumerate(ds.records):
        by_cls.setdefault(r.label, []).append(i)
    tau = 0.3
    grounders = {
        "sam3": SAM3Grounder(cfg.grounding.sam3_weights, tau, cfg.grounding.half),
        "grounded_sam": GroundedSAMGrounder(cfg.grounding.grounding_dino_id, cfg.grounding.sam_weights, tau),
    }
    out = {}
    n_concepts = 0
    for name, g in grounders.items():
        hits, confs = [], []
        for c in classes:
            concepts = [x for x in lookup(int(c)) if len(x.split()) > 2]
            n_concepts += len(concepts) if name == "sam3" else 0
            imgs = rng.choice(by_cls[int(c)], size=min(a.images, len(by_cls[int(c)])), replace=False)
            for i in imgs:
                img = to_base_pil(ds.load_image(int(i)), cfg.data.base_size)
                for concept in concepts:
                    det = g.detect(img, [concept])
                    hits.append(bool((det.scores >= tau).any()))
                    confs += det.scores.tolist()
        out[name] = {"hit_rate": 100 * float(np.mean(hits)) if hits else 0.0,
                     "avg_confidence": float(np.mean(confs)) if confs else 0.0, "n_prompts": len(hits)}
    out["n_concepts"] = n_concepts
    print(out)
    save_json(out, os.path.join(cfg.output_dir, "prompt_complexity.json"))


if __name__ == "__main__":
    main()
