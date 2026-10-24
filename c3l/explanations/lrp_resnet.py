"""
Differentiable epsilon-LRP for torchvision ResNets (Sec. 3.3, eps = 1e-6).

We re-implement the ResNet forward pass explicitly so that every affine
operation (conv, batch-norm, residual addition, average pooling, fc) is wrapped
with :func:`stabilize`. Input x Gradient of this forward then equals the
epsilon-rule relevance, and - unlike hook-based LRP libraries - it supports
``create_graph=True`` so ``L_concept`` / ``L_stab`` can be back-propagated into the
weights.

Rules used: epsilon-rule for conv / BN / fc / residual-add / avg-pool, winner-
take-all for max-pool (= its gradient), pass-through for ReLU (= its gradient).
Batch-norm always uses running statistics during explanation.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from c3l.explanations.base import stabilize


def _bn(x: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
    return F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)


def _block(block: nn.Module, x: torch.Tensor, eps: float) -> torch.Tensor:
    s = lambda t: stabilize(t, eps)
    identity = x
    if hasattr(block, "conv3"):  # Bottleneck (ResNet-50 style), supported for completeness
        out = F.relu(s(_bn(s(block.conv1(x)), block.bn1)))
        out = F.relu(s(_bn(s(block.conv2(out)), block.bn2)))
        out = s(_bn(s(block.conv3(out)), block.bn3))
    else:  # BasicBlock (ResNet-18)
        out = F.relu(s(_bn(s(block.conv1(x)), block.bn1)))
        out = s(_bn(s(block.conv2(out)), block.bn2))
    if block.downsample is not None:
        identity = s(_bn(s(block.downsample[0](x)), block.downsample[1]))
    return F.relu(s(out + identity))


def resnet_forward(
    net: nn.Module,
    x: torch.Tensor,
    eps: float = 0.0,
    feature_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Explicit ResNet forward (BN in inference mode).

    ``eps > 0``  -> LRP-epsilon modified gradients.
    ``feature_hook`` is applied to the layer4 output (used by Grad-CAM / IIA).
    """
    s = lambda t: stabilize(t, eps)
    h = F.relu(s(_bn(s(net.conv1(x)), net.bn1)))
    h = net.maxpool(h)
    for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
        for block in layer:
            h = _block(block, h, eps)
    if feature_hook is not None:
        h = feature_hook(h)
    h = s(torch.flatten(net.avgpool(h), 1))
    return s(net.fc(h))


def lrp_resnet(net: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, create_graph: bool) -> torch.Tensor:
    """Returns the (B,C,H,W) epsilon-LRP relevance of the target logits ``y``."""
    x = x.detach().requires_grad_(True)
    logits = resnet_forward(net, x, eps=eps)
    score = logits.gather(1, y.view(-1, 1)).sum()
    (grad,) = torch.autograd.grad(score, x, create_graph=create_graph)
    return x * grad
