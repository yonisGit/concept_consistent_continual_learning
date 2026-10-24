"""
Relevance-mass flow analysis (Appendix "Quantitative Evidence for the Crowding-Out Effect").

For test samples of task T_1 and region R in {C (concept), B (background), T (total)}:

    R_total(R) = sum_{i in R} S_x(i),   R_rel(R) = R_total(R) / f_theta(x)_y,   CR = R_rel(C) / R_rel(B)

computed with the *un-normalised* AttnLRP / LRP relevance after selected tasks.
Also reports the conservation gap  |sum_i S_x(i) - f_theta(x)_y| / |f_theta(x)_y|  (Eq. 4).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from c3l.explanations.factory import Explainer


def crowding_analysis(model, loader, explainer: Explainer, device) -> Dict[str, float]:
    model.eval()
    rc, rb, rt, cons = [], [], [], []
    for batch in loader:
        x, y, m = batch["image"].to(device), batch["label"].to(device), batch["mask"].to(device).float()
        s = explainer.raw(model, x, y, create_graph=False)
        with torch.no_grad():
            logit = model(x).gather(1, y.view(-1, 1)).squeeze(1)
        denom = torch.where(logit.abs() < 1e-6, torch.full_like(logit, 1e-6), logit)
        c = (s * m).flatten(1).sum(1) / denom
        b = (s * (1 - m)).flatten(1).sum(1) / denom
        t = s.flatten(1).sum(1) / denom
        rc += c.tolist()
        rb += b.tolist()
        rt += t.tolist()
        cons += (t - 1).abs().tolist()
    return {
        "R_rel_C": float(np.mean(rc)),
        "R_rel_B": float(np.mean(rb)),
        "R_rel_T": float(np.mean(rt)),
        "CR": float(np.mean(rc) / max(np.mean(rb), 1e-12)),
        "conservation_gap": float(np.median(cons)),
    }
