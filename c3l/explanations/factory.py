"""
Single entry point for computing saliency maps ``S_x = E_theta(x, y)``.

``Explainer(name, ...)(model, x, y, create_graph)`` returns a (B,H,W) map,
min-max normalised to [0,1] by default (Appendix A). Names:

* ``lrp``          epsilon-LRP (ResNet)                 - C3L default for CNNs
* ``attnlrp``      AttnLRP (ViT)                        - C3L default for ViTs
* ``gradcam``      Grad-CAM (ResNet)                    - evaluation protocol / ablation
* ``gradcam_vit``  Grad-CAM for ViT (Chefer et al.)     - evaluation protocol / ablation
* ``tattr``        Transformer Attribution (ViT)        - ablation
* ``iia``          Iterated Integrated Attributions     - ablation
* ``auto``         lrp / attnlrp depending on the architecture
* ``auto_gradcam`` gradcam / gradcam_vit depending on the architecture
"""

from __future__ import annotations

import torch

from c3l.explanations.attnlrp_vit import attnlrp_vit
from c3l.explanations.base import aggregate_channels, minmax_normalize
from c3l.explanations.gradcam import gradcam_resnet, gradcam_vit
from c3l.explanations.iia import iia_resnet, iia_vit
from c3l.explanations.lrp_resnet import lrp_resnet
from c3l.explanations.transformer_attribution import transformer_attribution


def resolve_name(name: str, arch: str) -> str:
    is_vit = arch.startswith("vit")
    if name == "auto":
        return "attnlrp" if is_vit else "lrp"
    if name == "auto_gradcam":
        return "gradcam_vit" if is_vit else "gradcam"
    if name == "gradcam" and is_vit:
        return "gradcam_vit"
    if name == "lrp" and is_vit:
        return "attnlrp"
    return name


class Explainer:
    def __init__(self, name: str = "auto", epsilon: float = 1e-6, aggregate: str = "sum",
                 normalize: str = "minmax", iia_steps: int = 8):
        self.name = name
        self.epsilon = epsilon
        self.aggregate = aggregate
        self.normalize = normalize
        self.iia_steps = iia_steps

    @classmethod
    def from_cfg(cls, ecfg, name: str | None = None) -> "Explainer":
        return cls(name or ecfg.name, ecfg.epsilon, ecfg.aggregate, ecfg.normalize, ecfg.iia_steps)

    def raw(self, model, x: torch.Tensor, y: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        """Un-normalised (B,H,W) relevance (signed for LRP methods)."""
        net, arch = model.net, model.arch
        name = resolve_name(self.name, arch)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            if name == "lrp":
                s = aggregate_channels(lrp_resnet(net, x, y, self.epsilon, create_graph), self.aggregate)
            elif name == "attnlrp":
                s = aggregate_channels(attnlrp_vit(net, x, y, self.epsilon, create_graph), self.aggregate)
            elif name == "gradcam":
                s = gradcam_resnet(net, x, y, create_graph)
            elif name == "gradcam_vit":
                s = gradcam_vit(net, x, y, create_graph)
            elif name == "tattr":
                s = transformer_attribution(net, x, y, self.epsilon, create_graph)
            elif name == "iia":
                s = (iia_vit if arch.startswith("vit") else iia_resnet)(net, x, y, self.iia_steps, create_graph)
            else:
                raise ValueError(f"Unknown explainer {name}")
        return s if create_graph else s.detach()

    def __call__(self, model, x: torch.Tensor, y: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        s = self.raw(model, x, y, create_graph)
        if self.normalize == "minmax":
            s = minmax_normalize(s)
        return s
