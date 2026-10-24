"""
Appendix "Isolating SAM 3's inherent capabilities" (Table sam3_zeroshot).

SAM 3 as a stand-alone zero-shot classifier, without any continual learning:

* Variant 1 (category names): prompt with every class name; predict the class of the
  highest-confidence mask.
* Variant 2 (attribute prompts): prompt with the filtered concepts of every class; a
  class's score is the mean over its concepts of the best detection confidence (0 if
  not detected, tau_conf applied); predict the arg-max class.

    python scripts/sam3_zero_shot.py --config configs/cub_vit.yaml --variant 2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c3l.concepts.discovery import ConceptLookup, load_concept_bank  # noqa: E402
from c3l.concepts.grounding import SAM3Grounder  # noqa: E402
from c3l.config import load_config  # noqa: E402
from c3l.data.datasets import build_dataset  # noqa: E402
from c3l.data.transforms import to_base_pil  # noqa: E402
from c3l.utils import save_json  # noqa: E402


def best_scores(g: SAM3Grounder, img, prompts, chunk: int, tau: float) -> np.ndarray:
    """Max confidence per prompt (prompts are queried in chunks to bound memory)."""
    out = np.zeros(len(prompts), np.float32)
    for s in range(0, len(prompts), chunk):
        det = g.detect(img, prompts[s:s + chunk])
        for cid, sc in zip(det.concept_ids, det.scores):
            if sc >= tau:
                out[s + cid] = max(out[s + cid], sc)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--variant", type=int, choices=[1, 2], default=1)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--max_images", type=int, default=None)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    splits = build_dataset(cfg)
    test, names = splits.test, splits.class_names
    tau = cfg.grounding.confidence_threshold
    g = SAM3Grounder(cfg.grounding.sam3_weights, tau, cfg.grounding.half)
    if a.variant == 2:
        lookup = ConceptLookup(load_concept_bank(cfg.concepts.validated_bank_path or cfg.concepts.bank_path), names,
                               field="validated" if cfg.concepts.validated_bank_path else "concepts")
        concepts = [lookup(c) for c in range(len(names))]
        flat = sorted({c for cs in concepts for c in cs})
        pos = {c: i for i, c in enumerate(flat)}
    rng = np.random.RandomState(cfg.seed)
    order = rng.permutation(len(test))[: a.max_images]
    correct = 0
    for n, i in enumerate(order):
        img = to_base_pil(test.load_image(int(i)), cfg.data.base_size)
        if a.variant == 1:
            pred = int(best_scores(g, img, names, a.chunk, tau).argmax())
        else:
            s = best_scores(g, img, flat, a.chunk, tau)
            cls_scores = [np.mean([s[pos[c]] for c in cs]) if cs else 0.0 for cs in concepts]
            pred = int(np.argmax(cls_scores))
        correct += int(pred == test.records[int(i)].label)
    acc = 100.0 * correct / max(len(order), 1)
    print(f"SAM 3 zero-shot variant {a.variant} on {cfg.data.name}: {acc:.2f}%")
    save_json({"variant": a.variant, "dataset": cfg.data.name, "acc": acc, "n": len(order)},
              os.path.join(cfg.output_dir, f"sam3_zero_shot_{cfg.data.name}_v{a.variant}_seed{cfg.seed}.json"))


if __name__ == "__main__":
    main()
