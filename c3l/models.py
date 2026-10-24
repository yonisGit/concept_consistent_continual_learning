"""
Backbones (Sec. 4.1): ResNet-18 (ImageNet-1K, torchvision) and ViT-B/16 (ImageNet-21K, timm).

A single linear head over *all* classes of the scenario is used; logits of
classes that have not been seen yet are masked out (class-incremental, single head).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContinualClassifier(nn.Module):
    """Wraps a backbone + head and exposes ``arch`` so explainers can dispatch."""

    def __init__(self, net: nn.Module, arch: str):
        super().__init__()
        self.net = net
        self.arch = arch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def mask_unseen(logits: torch.Tensor, num_seen: int) -> torch.Tensor:
        """Keep only the logits of classes seen so far."""
        return logits[:, :num_seen]


def build_model(cfg, num_classes: int) -> ContinualClassifier:
    arch = cfg.model.arch.lower()
    if arch in ("resnet18", "resnet"):
        import torchvision

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if cfg.model.pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        if num_classes != net.fc.out_features:
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        if cfg.model.freeze_backbone:
            for n, p in net.named_parameters():
                p.requires_grad_(n.startswith("fc."))
        return ContinualClassifier(net, "resnet18")

    if arch in ("vit", "vit_b16", "vit-b"):
        import timm

        net = timm.create_model(cfg.model.vit_name, pretrained=cfg.model.pretrained, num_classes=num_classes)
        if cfg.model.freeze_backbone:
            for n, p in net.named_parameters():
                p.requires_grad_(n.startswith("head"))
        return ContinualClassifier(net, "vit")

    raise ValueError(f"Unknown architecture: {cfg.model.arch}")
