"""
Qualitative figures (Appendix "Qualitative Visualizations"): image, concept mask and
saliency maps of one or more checkpoints (e.g. before / after continual learning).

Checkpoints are written by the trainer when ``analysis.save_checkpoints=true``.

    python scripts/visualize_explanations.py --config configs/imagenet_r_vit.yaml \
        --ckpt outputs/.../model_T1.pt outputs/.../model_T10.pt --n 8 --out figs/
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c3l.config import load_config  # noqa: E402
from c3l.data.scenario import load_base_crop  # noqa: E402
from c3l.data.transforms import center_crop  # noqa: E402
from c3l.experiment import ConceptResources, build_scenario  # noqa: E402
from c3l.explanations.factory import Explainer  # noqa: E402
from c3l.models import build_model  # noqa: E402
from c3l.utils import get_device, normalize_image  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--explainer", default=None, help="defaults to explainer.eval_name (Grad-CAM)")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default="figs")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    device = get_device(cfg.device)
    splits, scenario = build_scenario(cfg)
    expl = Explainer.from_cfg(cfg.explainer, name=a.explainer or cfg.explainer.eval_name)
    models = []
    for c in a.ckpt:
        m = build_model(cfg, scenario.num_classes).to(device).eval()
        m.load_state_dict(torch.load(c, map_location=device))
        models.append(m)
    try:
        prov = ConceptResources(cfg, splits.class_names).provider(splits.test)
    except Exception:
        prov = None
    idx = scenario.test_indices(0)[: a.n]
    os.makedirs(a.out, exist_ok=True)
    for i in idx:
        x = load_base_crop(splits.test, i, cfg.data.base_size, cfg.data.crop_size)
        y = torch.tensor([scenario.new_label(splits.test.records[i].label)], device=device)
        cols = 2 + len(models)
        fig, axs = plt.subplots(1, cols, figsize=(3 * cols, 3))
        axs[0].imshow(x.permute(1, 2, 0).numpy())
        axs[0].set_title(scenario.class_name_of(int(y)))
        if prov is not None:
            axs[1].imshow(center_crop(prov(i), cfg.data.crop_size).numpy(), cmap="gray")
        axs[1].set_title("concept mask")
        for k, m in enumerate(models):
            s = expl(m, normalize_image(x[None].to(device)), y)[0].cpu().numpy()
            axs[2 + k].imshow(x.permute(1, 2, 0).numpy())
            axs[2 + k].imshow(s, cmap="jet", alpha=0.5)
            axs[2 + k].set_title(os.path.basename(a.ckpt[k]))
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"sample_{i}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
