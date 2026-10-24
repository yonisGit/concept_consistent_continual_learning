"""
Aggregate ``results.json`` files: mean +- std over seeds for ACC / F / AIA (and
localisation metrics when present), plus paired t-tests of every run against a
reference run (default: the one whose name starts with "c3l").

    python scripts/aggregate_results.py outputs/imagenet_r_vit/components [--ref c3l]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c3l.evaluation.stats import format_pm, mean_std, paired_ttest  # noqa: E402
from c3l.utils import save_json  # noqa: E402


def collect(root: str):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "seed_*", "results.json"), recursive=True)):
        run_dir = os.path.dirname(os.path.dirname(path))
        name = os.path.relpath(run_dir, root)
        seed = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        with open(path) as f:
            res = json.load(f)
        flat = dict(res.get("metrics", {}))
        for k, v in (res.get("localization") or {}).items():
            flat[f"loc_{k}"] = v
        runs.setdefault(name, {})[seed] = flat
    return runs


def aggregate(root: str, ref: str | None = None) -> dict:
    runs = collect(root)
    if not runs:
        print(f"No results under {root}")
        return {}
    if ref is None:
        ref = next((n for n in runs if n.split("/")[-1].startswith("c3l")), None)
    summary = {}
    for name, by_seed in runs.items():
        metrics = sorted({k for r in by_seed.values() for k in r})
        summary[name] = {m: mean_std([r[m] for r in by_seed.values() if m in r]) for m in metrics}
        if ref and name != ref and ref in runs:
            common = sorted(set(by_seed) & set(runs[ref]))
            for m in ("ACC", "F", "AIA"):
                if all(m in by_seed[s] and m in runs[ref][s] for s in common):
                    summary[name][f"ttest_vs_{ref}_{m}"] = paired_ttest(
                        [runs[ref][s][m] for s in common], [by_seed[s][m] for s in common])
    print(f"\n{'run':45s} " + " ".join(f"{m:>16s}" for m in ("ACC", "F", "AIA")))
    for name, by_seed in runs.items():
        cells = []
        for m in ("ACC", "F", "AIA"):
            vals = [r[m] for r in by_seed.values() if m in r]
            cells.append(f"{format_pm(vals):>16s}" if vals else f"{'-':>16s}")
        print(f"{name:45s} " + " ".join(cells) + f"   (n={len(by_seed)})")
    save_json(summary, os.path.join(root, "summary.json"))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--ref", default=None)
    a = p.parse_args()
    aggregate(a.root, a.ref)
