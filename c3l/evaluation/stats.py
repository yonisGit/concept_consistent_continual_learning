"""Aggregation over runs: mean +- std and paired t-tests (p < 0.05), Sec. 4.1."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired t-test over matched runs (same seed / task order)."""
    from scipy.stats import ttest_rel

    if len(a) != len(b) or len(a) < 2:
        return {"t": float("nan"), "p": float("nan")}
    t, p = ttest_rel(a, b)
    return {"t": float(t), "p": float(p), "significant": bool(p < 0.05)}


def format_pm(values: List[float], digits: int = 2) -> str:
    s = mean_std(values)
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"
