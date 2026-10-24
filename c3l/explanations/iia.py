"""
Iterated Integrated Attributions, "IIA" (Barkan et al., ICCV 2023) - C3L-IIA ablation.

IIA integrates gradient x activation of internal representations along an
interpolation path from a baseline to the input. We implement the single-level
(non-nested) path integral, which is what makes it affordable inside a training
loop:

* ViT: for alpha in (0,1], accumulate E_h[(A_alpha * dS/dA_alpha)^+] for every block on
  the scaled input alpha*x, then roll out over blocks and read the [CLS] row.
* CNN: integrated Grad-CAM on the layer4 activations along the same path.

This is a simplified version of the original (which iterates the integral over
several nested paths); the README documents the deviation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from c3l.explanations.attnlrp_vit import patch_grid, vit_forward
from c3l.explanations.base import upsample_map
from c3l.explanations.lrp_resnet import resnet_forward


def iia_vit(vit, x: torch.Tensor, y: torch.Tensor, steps: int, create_graph: bool) -> torch.Tensor:
    acc = None
    for k in range(1, steps + 1):
        alpha = k / steps
        attns = []
        logits = vit_forward(vit, x * alpha, lrp=False, attn_store=attns)
        score = logits.gather(1, y.view(-1, 1)).sum()
        grads = torch.autograd.grad(score, attns, create_graph=create_graph)
        maps = [(a * g).clamp(min=0).mean(1) for a, g in zip(attns, grads)]
        acc = maps if acc is None else [m0 + m for m0, m in zip(acc, maps)]
    B, N, _ = acc[0].shape
    R = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, N, N)
    for m in acc:
        R = R + torch.bmm(m / steps, R)
    n_pref = getattr(vit, "num_prefix_tokens", 1)
    gh, gw = patch_grid(vit, x)
    return upsample_map(R[:, 0, n_pref:].reshape(B, gh, gw), x.shape[-2:])


def iia_resnet(net, x: torch.Tensor, y: torch.Tensor, steps: int, create_graph: bool) -> torch.Tensor:
    cam = 0.0
    for k in range(1, steps + 1):
        alpha = k / steps
        feats = {}

        def keep(h):
            feats["a"] = h
            return h

        logits = resnet_forward(net, x * alpha, eps=0.0, feature_hook=keep)
        score = logits.gather(1, y.view(-1, 1)).sum()
        (g,) = torch.autograd.grad(score, feats["a"], create_graph=create_graph)
        cam = cam + F.relu((g * feats["a"]).sum(1))
    return upsample_map(cam / steps, x.shape[-2:])
