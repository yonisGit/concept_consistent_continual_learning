"""
Builds everything an experiment needs from a config: dataset, class-incremental
scenario, model, explainer, concept bank, grounder and cached mask providers.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, List, Optional

from c3l.concepts.discovery import ConceptLookup, load_concept_bank
from c3l.concepts.grounding import MaskProvider, build_grounder, grounding_tag
from c3l.data.base import ImageListDataset
from c3l.data.datasets import build_dataset
from c3l.data.scenario import ClassIncrementalScenario
from c3l.utils import get_logger

logger = get_logger()


def _file_hash(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return "none"
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:10]


class ConceptResources:
    """Lazily creates the grounder and per-dataset ``MaskProvider``s."""

    def __init__(self, cfg, class_names: List[str]):
        self.cfg = cfg
        self.class_names = class_names
        self._grounder = None
        self._providers = {}
        g = cfg.grounding.method
        if g in ("object", "gt_parts"):
            self.lookup = None
            self.source = g
        else:
            path = cfg.concepts.validated_bank_path if cfg.concepts.use_validated else None
            path = path or cfg.concepts.bank_path
            if not path:
                raise ValueError("concepts.bank_path is required (run scripts/discover_concepts.py first).")
            bank = load_concept_bank(path)
            field = "validated" if (cfg.concepts.use_validated and cfg.concepts.validated_bank_path) else "concepts"
            self.lookup = ConceptLookup(bank, class_names, field=field)
            self.source = f"{os.path.basename(path)}:{_file_hash(path)}:{field}"

    @property
    def grounder(self):
        if self._grounder is None:
            self._grounder = build_grounder(self.cfg)
        return self._grounder

    def prompt_fn(self) -> Callable[[int], List[str]]:
        if self.cfg.grounding.method == "object":
            return lambda lbl: [self.class_names[lbl]]
        if self.lookup is None:
            return lambda lbl: []
        return self.lookup

    def provider(self, dataset: ImageListDataset) -> MaskProvider:
        key = dataset.name
        if key not in self._providers:
            self._providers[key] = _LazyProvider(self, dataset)
        return self._providers[key]


class _LazyProvider(MaskProvider):
    """MaskProvider that only instantiates the (heavy) grounder on a cache miss."""

    def __init__(self, res: ConceptResources, dataset: ImageListDataset):
        cfg = res.cfg
        self._res = res
        super().__init__(dataset, None, res.prompt_fn(), cfg.data.base_size, cfg.grounding.cache_dir,
                         grounding_tag(cfg, res.source))

    def __getstate__(self):
        # DataLoader workers only read the (pre-computed) cache: never pickle the foundation models.
        state = self.__dict__.copy()
        state["grounder"] = None
        state["_res"] = None
        state["prompt_fn"] = None
        return state

    def compute(self, idx: int):
        if self._res is None:
            raise RuntimeError("Concept mask missing from cache inside a DataLoader worker; "
                               "call provider.precompute(indices) before building the loader.")
        if self.grounder is None:
            self.grounder = self._res.grounder
        return super().compute(idx)


def build_scenario(cfg):
    splits = build_dataset(cfg)
    n_tasks = cfg.data.num_tasks
    scenario = ClassIncrementalScenario(splits, n_tasks, seed=cfg.seed, shuffle=cfg.data.shuffle_class_order)
    logger.info(f"{cfg.data.name}: {scenario.num_classes} classes, {n_tasks} tasks x "
                f"{scenario.classes_per_task} classes, {len(splits.train)} train / {len(splits.test)} test images")
    return splits, scenario
