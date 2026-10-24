"""
Continual-learning metrics (Sec. 4.1).

``A[i, j]`` = accuracy (%) on the test set of task ``i`` after training on task ``j``
(class-incremental: prediction over all classes seen so far).

* ACC = 1/T sum_i A[i, T]
* F   = 1/(T-1) sum_{i<T} ( max_{j in {i..T}} A[i, j] - A[i, T] )        (paper's definition)
* AIA = 1/T sum_t ( 1/t sum_{i<=t} A[i, t] )
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def accuracy_matrix_metrics(A: np.ndarray) -> Dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    T = A.shape[0]
    acc = float(A[:, T - 1].mean())
    if T > 1:
        forgetting = float(np.mean([A[i, i:].max() - A[i, T - 1] for i in range(T - 1)]))
    else:
        forgetting = 0.0
    aia = float(np.mean([A[: t + 1, t].mean() for t in range(T)]))
    return {"ACC": acc, "F": forgetting, "AIA": aia}


@torch.no_grad()
def evaluate_tasks(model, loaders: List, num_seen: int, device, amp: bool = False) -> List[float]:
    """Accuracy (%) of ``model`` on each task test loader, predicting over ``num_seen`` classes."""
    model.eval()
    accs = []
    for loader in loaders:
        correct, total = 0, 0
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                logits = model(x)[:, :num_seen]
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        accs.append(100.0 * correct / max(total, 1))
    return accs


@torch.no_grad()
def restricted_accuracy(model, loader, allowed: List[List[int]], device) -> float:
    """
    Accuracy of a 1000-way ImageNet model on a dataset whose class ``c`` corresponds
    to the ImageNet indices ``allowed[c]`` (ImageNet-A: 1 index, ObjectNet: >= 1).
    Logits are restricted to the union of allowed indices.
    """
    model.eval()
    union = sorted({i for a in allowed for i in a})
    idx_to_cls = {}
    for c, a in enumerate(allowed):
        for i in a:
            idx_to_cls[i] = c
    union_t = torch.tensor(union, device=device)
    correct, total = 0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = model(x)[:, union_t]
        pred_in = union_t[logits.argmax(1)]
        pred = torch.tensor([idx_to_cls[int(i)] for i in pred_in], device=device)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)
