"""
Grad-CAM (Selvaraju et al., 2017) for CNNs and the Grad-CAM variant for ViTs used
by Chefer et al. (2021). Used (i) as the unified *evaluation* explainer for all
methods (Sec. 4.1) and (ii) for the C3L-GradCAM ablation (Table abl_xai).
Both are differentiable (``create_graph=True``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from c3l.explanations.attnlrp_vit import patch_grid, vit_forward
from c3l.explanations.base import upsample_map
from c3l.explanations.lrp_resnet import resnet_forward


def gradcam_resnet(net, x: torch.Tensor, y: torch.Tensor, create_graph: bool) -> torch.Tensor:
    feats = {}

    def keep(h):
        feats["a"] = h
        return h

    logits = resnet_forward(net, x, eps=0.0, feature_hook=keep)
    score = logits.gather(1, y.view(-1, 1)).sum()
    (g,) = torch.autograd.grad(score, feats["a"], create_graph=create_graph)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats["a"]).sum(1))
    return upsample_map(cam, x.shape[-2:])


def gradcam_vit(vit, x: torch.Tensor, y: torch.Tensor, create_graph: bool) -> torch.Tensor:
    """Chefer et al.'s Grad-CAM for ViT: last-block attention weighted by its mean gradient."""
    attns = []
    logits = vit_forward(vit, x, lrp=False, attn_store=attns)
    score = logits.gather(1, y.view(-1, 1)).sum()
    a = attns[-1]  # (B, heads, N, N)
    (g,) = torch.autograd.grad(score, a, create_graph=create_graph)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).mean(1))  # (B, N, N)
    n_pref = getattr(vit, "num_prefix_tokens", 1)
    gh, gw = patch_grid(vit, x)
    cls_row = cam[:, 0, n_pref:].reshape(-1, gh, gw)
    return upsample_map(cls_row, x.shape[-2:])
