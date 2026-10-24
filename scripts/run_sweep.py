"""
Run a sweep (multiple seeds and/or ablation settings) and aggregate the results.

Sweep file format (see configs/ablations/)::

    base: ../imagenet_r_vit.yaml          # relative to the sweep file (or pass --base)
    seeds: [0, 1, 2, 3, 4]
    runs:                                 # named override lists ...
      c3l: []
      wo_stab: [method.lambda_stab=0.0]
    grid:                                 # ... and/or a cartesian grid
      replay.buffer_size: [200, 500, 1000]

    python scripts/run_sweep.py configs/ablations/components.yaml
    python scripts/run_sweep.py configs/ablations/main_results.yaml --base configs/cub_vit.yaml
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c3l.config import load_config  # noqa: E402
from c3l.utils import get_logger  # noqa: E402
from main import run  # noqa: E402
from scripts.aggregate_results import aggregate  # noqa: E402


def expand(sweep: dict):
    runs = dict(sweep.get("runs") or {})
    grid = sweep.get("grid") or {}
    if grid:
        keys = list(grid)
        for values in itertools.product(*[grid[k] for k in keys]):
            name = "__".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, values))
            runs[name] = [f"{k}={v}" for k, v in zip(keys, values)]
    return runs or {"default": []}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep")
    p.add_argument("--base", default=None, help="Override the base config of the sweep")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--only", nargs="*", default=None, help="Run only these named runs")
    p.add_argument("--set", nargs="*", default=[], help="Extra overrides applied to every run")
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()

    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)
    if not args.base and not sweep.get("base"):
        p.error("this sweep has no 'base' config; pass --base configs/<dataset>_<arch>.yaml")
    base = args.base or os.path.join(os.path.dirname(os.path.abspath(args.sweep)), sweep["base"])
    seeds = args.seeds if args.seeds is not None else sweep.get("seeds", [0])
    sweep_name = os.path.splitext(os.path.basename(args.sweep))[0]
    base_name = os.path.splitext(os.path.basename(base))[0]
    logger = get_logger()

    exp_names = []
    for run_name, overrides in expand(sweep).items():
        if args.only and run_name not in args.only:
            continue
        exp = f"{base_name}/{sweep_name}/{run_name}"
        exp_names.append(exp)
        for seed in seeds:
            cfg = load_config(base, list(overrides) + list(args.set) + [f"seed={seed}", f"experiment_name={exp}"])
            out = os.path.join(cfg.output_dir, exp, f"seed_{seed}", "results.json")
            if os.path.exists(out):
                logger.info(f"skip (exists): {out}")
                continue
            logger.info(f"==> {exp} seed={seed} overrides={overrides}")
            if not args.dry:
                run(cfg)
    if not args.dry:
        cfg = load_config(base)
        aggregate(os.path.join(cfg.output_dir, base_name, sweep_name))


if __name__ == "__main__":
    main()
