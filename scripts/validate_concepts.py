"""
Step 2 of C3L (optional): concept filtering by occurrence rate and spatial coverage.

    python scripts/validate_concepts.py --config configs/cub_vit.yaml
    # reads concepts.bank_path, writes concepts.validated_bank_path (+ *_stats.json)

Held-out images: the dataset's validation split if ``data.val_per_class > 0``,
otherwise ``concepts.validation_images_per_class`` random training images per class.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c3l.concepts.discovery import load_concept_bank  # noqa: E402
from c3l.concepts.grounding import build_grounder  # noqa: E402
from c3l.concepts.validation import ConceptValidator, bank_statistics, validate_bank  # noqa: E402
from c3l.config import load_config  # noqa: E402
from c3l.data.datasets import build_dataset, build_imagenet_subset  # noqa: E402
from c3l.utils import get_logger, save_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--dataset", default=None, help="imagenet_subset for the RQ4 experiment")
    p.add_argument("--occ", type=float, default=None, help="override tau_occ")
    p.add_argument("--cov", type=float, default=None, help="override tau_cov")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    logger = get_logger()
    splits = build_imagenet_subset(cfg) if a.dataset == "imagenet_subset" else build_dataset(cfg)
    held_out = splits.val if splits.val is not None and len(splits.val) else splits.train
    bank = load_concept_bank(cfg.concepts.bank_path)
    cfg.grounding.method = "sam3"  # validation always uses SAM 3 (concept + class-name prompts)
    validator = ConceptValidator(
        build_grounder(cfg),
        occurrence_threshold=a.occ if a.occ is not None else cfg.concepts.occurrence_threshold,
        coverage_threshold=a.cov if a.cov is not None else cfg.concepts.coverage_threshold,
        coverage_mode=cfg.concepts.coverage_mode,
        base_size=cfg.data.base_size,
    )
    tic = time.time()
    bank = validate_bank(bank, validator, held_out, cfg.concepts.validation_images_per_class, seed=cfg.seed)
    out = a.out or cfg.concepts.validated_bank_path
    save_json(bank, out)
    stats = bank_statistics(bank)
    stats["minutes"] = (time.time() - tic) / 60
    save_json(stats, out.replace(".json", "_stats.json"))
    logger.info(f"Validated bank -> {out} | stats: {stats}")


if __name__ == "__main__":
    main()
