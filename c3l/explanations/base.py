"""Shared utilities for explanation methods."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def stabilize(z: torch.Tensor, eps: float) -> torch.Tensor:
    """
    LRP epsilon-rule expressed as a *modified gradient*.

    Returns a tensor with the same value as ``z`` whose local gradient is
    ``z / (z + eps * sign(z))``. If every affine op of a network is wrapped
    with this function, Input x Gradient of the network equals epsilon-LRP
    (Montavon et al., 2019; Ancona et al., 2018). Because the scale factor is a
    detached multiplier, the graph stays differentiable w.r.t. the weights, which
    is what allows ``create_graph=True`` (double back-prop) during training.
    """
    if eps <= 0:
        return z
    sign = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    scale = (z / (z + eps * sign)).detach()
    return z * scale + (z * (1.0 - scale)).detach()


def halve_gradient(t: torch.Tensor) -> torch.Tensor:
    """Same value, half the gradient (AttnLRP *uniform rule* for bilinear matmuls)."""
    return 0.5 * t + 0.5 * t.detach()


def identity_rule(fn_out: torch.Tensor, fn_in: torch.Tensor) -> torch.Tensor:
    """
    AttnLRP identity rule for element-wise non-linearities (GELU, ...):
    relevance passes through unchanged, i.e. gradient := out / in.
    """
    ratio = torch.where(fn_in.abs() > 1e-6, fn_out / fn_in, torch.full_like(fn_in, 0.5)).detach()
    return fn_in * ratio


def aggregate_channels(attr: torch.Tensor, how: str = "sum") -> torch.Tensor:
    """(B,C,H,W) attribution -> (B,H,W) pixel relevance."""
    if how == "sum":
        return attr.sum(1)
    if how == "abs":
        return attr.abs().sum(1)
    if how == "pos":
        return attr.sum(1).clamp(min=0)
    raise ValueError(f"Unknown aggregation {how}")


def minmax_normalize(s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max normalisation to [0,1] (differentiable)."""
    flat = s.flatten(1)
    mn = flat.min(1, keepdim=True).values
    mx = flat.max(1, keepdim=True).values
    return ((flat - mn) / (mx - mn + eps)).view_as(s)


def upsample_map(m: torch.Tensor, size) -> torch.Tensor:
    """(B,h,w) -> (B,H,W) bilinear."""
    return F.interpolate(m[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]


def target_scores(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, y.view(-1, 1)).squeeze(1)
