"""
Transformer Attribution, "T-Attr" (Chefer et al., CVPR 2021) - C3L-T-Attr ablation.

    A_bar^(b) = E_h[ (grad A^(b)  *  R^(A^(b)))^+ ],     R <- R + A_bar^(b) R

The relevance of each attention map ``R^(A)`` is taken from the AttnLRP forward
(``A * dS/dA`` under the LRP-modified gradients). For a single, differentiable
forward we use the same LRP-modified gradient as ``grad A`` (the original method
uses the plain gradient there) - a minor deviation noted in the README.
"""

from __future__ import annotations

import torch

from c3l.explanations.attnlrp_vit import patch_grid, vit_forward
from c3l.explanations.base import upsample_map


def transformer_attribution(vit, x: torch.Tensor, y: torch.Tensor, eps: float, create_graph: bool) -> torch.Tensor:
    attns = []
    logits = vit_forward(vit, x, lrp=True, eps=eps, attn_store=attns)
    score = logits.gather(1, y.view(-1, 1)).sum()
    grads = torch.autograd.grad(score, attns, create_graph=create_graph)
    B, _, N, _ = attns[0].shape
    R = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, N, N)
    for a, g in zip(attns, grads):
        rel = a * g                                  # LRP relevance of the attention map
        a_bar = (g * rel).clamp(min=0).mean(1)       # E_h[(grad A . R^A)^+]
        R = R + torch.bmm(a_bar, R)
    n_pref = getattr(vit, "num_prefix_tokens", 1)
    gh, gw = patch_grid(vit, x)
    cls_row = R[:, 0, n_pref:].reshape(B, gh, gw)
    return upsample_map(cls_row, x.shape[-2:])
