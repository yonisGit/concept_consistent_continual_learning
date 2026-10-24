"""
Step 1 of C3L (offline, before any training): LLM concept discovery (LF-CBM style).

    export OPENAI_API_KEY=...
    python scripts/discover_concepts.py --config configs/cub_vit.yaml
    # -> writes concepts.bank_path (e.g. concept_banks/cub_raw.json)

For the RQ4 ImageNet subset use ``--dataset imagenet_subset``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c3l.concepts.discovery import discover_concept_bank  # noqa: E402
from c3l.config import load_config  # noqa: E402
from c3l.data.datasets import build_dataset, build_imagenet_subset  # noqa: E402
from c3l.utils import get_logger  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--dataset", default=None, help="imagenet_subset for the RQ4 experiment")
    p.add_argument("--out", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    logger = get_logger()
    splits = build_imagenet_subset(cfg) if a.dataset == "imagenet_subset" else build_dataset(cfg)
    class_names = splits.class_names
    if a.dataset == "imagenet_subset":
        chosen = splits.info["chosen_classes"]
        class_names = [n if i in set(chosen) else None for i, n in enumerate(class_names)]
    out = a.out or cfg.concepts.bank_path
    tic = time.time()
    names = [n for n in class_names if n is not None]
    if os.path.exists(out) and not a.overwrite:
        logger.info(f"{out} exists; use --overwrite to regenerate.")
        return
    bank = discover_concept_bank(names, cfg, out, overwrite=True)
    if a.dataset == "imagenet_subset":
        # re-key by ImageNet index so ConceptLookup can match labels directly
        idx = [i for i, n in enumerate(class_names) if n is not None]
        bank = {str(i): bank[str(j)] for j, i in enumerate(idx)}
        from c3l.utils import save_json
        save_json(bank, out)
    logger.info(f"Concept discovery for {len(names)} classes took {(time.time() - tic) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
