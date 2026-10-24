"""
C3L - Concept-Consistent Continual Learning: main entry point.

    python main.py --config configs/imagenet_r_vit.yaml
    python main.py --config configs/cub_resnet18.yaml --set seed=3 method.name=er

Runs one class-incremental experiment (one seed) and writes
``<output_dir>/<experiment_name>/seed_<seed>/{config.yaml,results.json,train.log}``.
Use ``scripts/run_sweep.py`` for multiple seeds / ablation grids.
"""

from __future__ import annotations

import argparse
import os

import yaml

from c3l.config import load_config
from c3l.experiment import build_scenario
from c3l.trainer import ContinualTrainer
from c3l.utils import get_device, get_logger, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True, help="YAML config file")
    p.add_argument("--set", nargs="*", default=[], help="Overrides, e.g. method.lambda_stab=1.0 seed=2")
    return p.parse_args(argv)


def run(cfg) -> dict:
    run_dir = os.path.join(cfg.output_dir, cfg.experiment_name, f"seed_{cfg.seed}")
    os.makedirs(run_dir, exist_ok=True)
    logger = get_logger(log_file=os.path.join(run_dir, "train.log"))
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger.info(f"Run dir: {run_dir} | device: {device}")
    splits, scenario = build_scenario(cfg)
    trainer = ContinualTrainer(cfg, splits, scenario, device, run_dir)
    return trainer.run()


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    run(cfg)


if __name__ == "__main__":
    main()
