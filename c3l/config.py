"""
Experiment configuration.

Configs are plain YAML files (see ``configs/``). A config may ``inherit`` from
another YAML file (paths are relative to the inheriting file) and any field can
be overridden from the command line with dotted keys, e.g.::

    python main.py --config configs/imagenet_r_vit.yaml \
        --set method.lambda_concept=0.5 replay.buffer_size=200 seed=1

All defaults below follow Appendix "Implementation Details" of the paper.
"""

from __future__ import annotations

import ast
import copy
import os
from typing import Any, Dict, Iterable, Optional

import yaml

# --------------------------------------------------------------------------- #
# Defaults (paper, Appendix A)
# --------------------------------------------------------------------------- #
DEFAULTS: Dict[str, Any] = {
    "experiment_name": "c3l",
    "output_dir": "./outputs",
    "seed": 0,
    "device": "auto",
    "num_workers": 4,
    "amp": False,  # mixed precision is OFF by default: double-backprop through LRP is fp16-fragile

    "data": {
        "name": "imagenet_r",       # cub | imagenet_r | imagenet_a | objectnet | domainnet | partimagenet | fitz | imagenet100c
        "root": "./data",
        "num_tasks": 10,            # 10 for most datasets, 5 for DomainNet
        "shuffle_class_order": True,  # class order is permuted with the run seed ("different task orderings")
        "base_size": 256,           # images are resized to base_size x base_size; masks live in this frame
        "crop_size": 224,
        "train_test_split": 0.8,    # used by datasets without an official split (ImageNet-R/A, ObjectNet)
        "val_per_class": 0,         # held-out images per class reserved for concept validation (0 = use train)
        "domainnet_domains": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        "imagenet100c": {"skew_ratio": 0.5, "noise_std": 0.18, "base_corrupt_rate": 0.01},
        "partimagenet": {"num_classes": 150},
    },

    "model": {
        "arch": "vit",              # resnet18 | vit
        "pretrained": True,
        # ResNet-18: ImageNet-1K weights (torchvision). ViT-B/16: ImageNet-21K weights (timm).
        "vit_name": "vit_base_patch16_224.augreg_in21k",
        "freeze_backbone": False,
    },

    "train": {
        "epochs": 20,               # 20 for ImageNet-R/A, ObjectNet, DomainNet; 50 for CUB
        "batch_size": 128,
        "replay_batch_size": 32,
        "optimizer": "sgd",
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "grad_clip": None,
        "log_every": 50,
        "eval_batch_size": 256,
    },

    "method": {
        # c3l | c3l_noreplay | c3l_fullreplay | c3l_pp | er | finetune | joint
        "name": "c3l",
        "lambda_concept": 0.8,      # 1.0 for CNNs, 0.8 for ViTs
        "lambda_stab": 1.2,         # 1.5 for CNNs, 1.2 for ViTs
        "concept_loss": "bce",      # bce (paper) | l1 | l2
        "stab_loss": "cosine",      # cosine (paper) | l1 | l2
        "background": "stabilize",  # stabilize (paper) | suppress | none
        "concept_on_current": False,  # True == C3L++ (L_concept also on current-task samples)
    },

    "explainer": {
        "name": "auto",             # auto -> lrp (resnet18) / attnlrp (vit); gradcam | gradcam_vit | tattr | iia
        "epsilon": 1.0e-6,
        "aggregate": "sum",         # how to collapse colour channels of Input x Gradient: sum | abs | pos
        "normalize": "minmax",      # saliency maps are min-max normalised to [0, 1]
        "iia_steps": 8,
        "eval_name": "auto_gradcam",  # explainer used for localisation evaluation (Grad-CAM for every method)
    },

    "replay": {
        "buffer_size": 1000,
        "balanced": True,           # each task contributes equally
        "flip": False,              # never augment replay samples: S_ref was computed on the exact input
    },

    "concepts": {
        "bank_path": None,          # JSON {class_name: [concepts]}; generated with scripts/discover_concepts.py
        "validated_bank_path": None,  # output of scripts/validate_concepts.py
        "use_validated": True,      # Appendix C: filtering is optional
        "llm_model": "gpt-4o-mini",
        "prompts": ["important", "superclass"],  # LF-CBM prompts without the "around" prompt
        "max_concept_length": 30,
        "class_sim_cutoff": 0.85,
        "concept_sim_cutoff": 0.9,
        "occurrence_threshold": 0.15,   # nu_occ
        "coverage_threshold": 0.20,     # nu_cov
        "coverage_mode": "set_iou",     # see concepts/validation.py
        "validation_images_per_class": 40,
    },

    "grounding": {
        "method": "sam3",           # sam3 | grounded_sam | object (class-name prompt) | gt_parts
        "confidence_threshold": 0.3,  # tau_conf
        "sam3_weights": "sam3.pt",
        "grounding_dino_id": "IDEA-Research/grounding-dino-base",
        "sam_weights": "sam2.1_b.pt",
        "half": True,
        "cache_dir": "./cache/masks",
    },

    "analysis": {
        "crowding_tasks": [],       # e.g. [1, 3, 5, 8, 10] -> Appendix "Crowding-out" analysis
        "localization": False,      # evaluate semantic localisation (CUB) after the last task
        "save_checkpoints": False,
    },
}


class Config(dict):
    """A dict with attribute access (recursively)."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict) and not isinstance(obj, Config):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o
        return unwrap(self)


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        parent_path = os.path.join(os.path.dirname(os.path.abspath(path)), parent)
        merged = _load_yaml(parent_path)
        return _deep_update(merged, cfg)
    return cfg


def _parse_value(v: str) -> Any:
    low = v.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def load_config(path: Optional[str] = None, overrides: Optional[Iterable[str]] = None) -> Config:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        _deep_update(cfg, _load_yaml(path))
    apply_overrides(cfg, overrides)
    _apply_arch_defaults(cfg, path, overrides)
    return Config.wrap(cfg)


def _apply_arch_defaults(cfg: Dict[str, Any], path, overrides) -> None:
    """If lambdas were not set explicitly, use the per-architecture values from the paper."""
    explicit = set()
    if path:
        raw = _load_yaml(path)
        explicit |= set((raw.get("method") or {}).keys())
    for o in overrides or []:
        if o.startswith("method.lambda_"):
            explicit.add(o.split("=")[0].split(".")[1])
    if cfg["model"]["arch"].startswith("resnet"):
        if "lambda_concept" not in explicit:
            cfg["method"]["lambda_concept"] = 1.0
        if "lambda_stab" not in explicit:
            cfg["method"]["lambda_stab"] = 1.5
