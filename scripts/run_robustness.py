"""
RQ4 - robustness to OOD samples (Sec. 4.5, Appendix "Robustness Tests Further Details").

1. Sample 5 images from each of 500 random ImageNet-1K classes (2,500 images) and
   40 further images per class for concept validation.
2. Run the full C3L preprocessing on this subset: concept discovery + validation
   (scripts/discover_concepts.py / validate_concepts.py with --dataset imagenet_subset),
   SAM 3 grounding, and LRP / AttnLRP reference explanations of the pretrained model.
3. Fine-tune the ImageNet-1K pretrained model on the subset with
   CE + lambda_concept * L_concept + lambda_stab * L_stab (all samples carry masks and S_ref).
4. Report accuracy and explanation mIoU on ImageNet-A and ObjectNet (classes overlapping
   ImageNet-1K) before and after fine-tuning. mIoU compares Grad-CAM maps with SAM 3
   object masks obtained by prompting with the ground-truth class name.

    python scripts/run_robustness.py --config configs/robustness_vit.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c3l.config import Config, load_config  # noqa: E402
from c3l.data.datasets import build_imagenet_a_eval, build_imagenet_subset, build_objectnet_eval  # noqa: E402
from c3l.data.scenario import TaskDataset, load_base_crop  # noqa: E402
from c3l.data.transforms import center_crop  # noqa: E402
from c3l.evaluation.localization import LocalizationMeter  # noqa: E402
from c3l.evaluation.metrics import restricted_accuracy  # noqa: E402
from c3l.experiment import ConceptResources  # noqa: E402
from c3l.explanations.factory import Explainer  # noqa: E402
from c3l.losses import C3LRegularizer  # noqa: E402
from c3l.models import build_model  # noqa: E402
from c3l.utils import get_device, get_logger, normalize_image, save_json, set_seed  # noqa: E402


class SupervisedSubset(Dataset):
    """Deterministic crops with aligned concept masks and reference saliency."""

    def __init__(self, ds, masks, s_ref, base, crop):
        self.ds, self.masks, self.s_ref, self.base, self.crop = ds, masks, s_ref, base, crop

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x = load_base_crop(self.ds, i, self.base, self.crop)
        return {"image": normalize_image(x[None])[0], "label": self.ds.records[i].label,
                "mask": self.masks[i], "s_ref": self.s_ref[i]}


def evaluate(model, cfg, device, explainer, max_loc_images: int):
    out = {}
    for name, builder in (("imagenet_a", build_imagenet_a_eval), ("objectnet", build_objectnet_eval)):
        ds, allowed = builder(cfg)
        ident = {c: c for c in range(len(ds.class_names))}
        loader = DataLoader(TaskDataset(ds, list(range(len(ds))), ident, train=False,
                                        base_size=cfg.data.base_size, crop_size=cfg.data.crop_size),
                            batch_size=cfg.train.eval_batch_size, num_workers=cfg.num_workers)
        acc = restricted_accuracy(model, loader, allowed, device)
        # explanation mIoU vs SAM 3 object masks (class-name prompt)
        ocfg = Config.wrap(copy.deepcopy(cfg.to_dict()))
        ocfg.grounding.method = "object"
        prov = ConceptResources(ocfg, ds.class_names).provider(ds)
        idx = list(range(min(len(ds), max_loc_images)))
        loc_loader = DataLoader(TaskDataset(ds, idx, ident, train=False, base_size=cfg.data.base_size,
                                            crop_size=cfg.data.crop_size),
                                batch_size=32, num_workers=cfg.num_workers)
        meter = LocalizationMeter()
        for batch in loc_loader:
            x = batch["image"].to(device)
            y_in = torch.tensor([allowed[int(c)][0] for c in batch["label"]], device=device)
            sal = explainer(model, x, y_in)
            gt = torch.stack([center_crop(prov(int(i)), cfg.data.crop_size) for i in batch["index"]])
            meter.update(sal, gt)
        out[name] = {"acc": acc, **{k: v for k, v in meter.compute().items()}}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--max_loc_images", type=int, default=2000)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = os.path.join(cfg.output_dir, cfg.experiment_name, f"seed_{cfg.seed}")
    logger = get_logger(log_file=os.path.join(run_dir, "robustness.log"))

    model = build_model(cfg, 1000).to(device).eval()
    expl = Explainer.from_cfg(cfg.explainer)
    eval_expl = Explainer.from_cfg(cfg.explainer, name=cfg.explainer.eval_name)
    results = {"before": evaluate(model, cfg, device, eval_expl, a.max_loc_images)}
    logger.info(f"before: {results['before']}")

    splits = build_imagenet_subset(cfg)
    train = splits.train
    res = ConceptResources(cfg, splits.class_names)
    prov = res.provider(train)
    prov.precompute(range(len(train)))
    masks = [center_crop(prov(i), cfg.data.crop_size) for i in range(len(train))]
    s_ref = []
    for s in range(0, len(train), 32):
        idx = list(range(s, min(s + 32, len(train))))
        x = torch.stack([load_base_crop(train, i, cfg.data.base_size, cfg.data.crop_size) for i in idx])
        y = torch.tensor([train.records[i].label for i in idx], device=device)
        s_ref += list(expl(model, normalize_image(x.to(device)), y).cpu())
    loader = DataLoader(SupervisedSubset(train, masks, s_ref, cfg.data.base_size, cfg.data.crop_size),
                        batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)

    reg = C3LRegularizer(cfg.method)
    opt = torch.optim.SGD([q for q in model.parameters() if q.requires_grad], lr=cfg.train.lr,
                          momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    for epoch in range(cfg.train.epochs):
        model.train()
        tot, logs = 0.0, {}
        for batch in loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            ce = F.cross_entropy(model(x), y)
            s = expl(model, x, y, create_graph=True)
            r, logs = reg(s, batch["mask"].to(device), batch["s_ref"].to(device))
            loss = ce + r
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        logger.info(f"epoch {epoch + 1}: loss {tot / max(len(loader), 1):.4f} {logs}")

    model.eval()
    results["after"] = evaluate(model, cfg, device, eval_expl, a.max_loc_images)
    logger.info(f"after: {results['after']}")
    save_json(results, os.path.join(run_dir, "robustness_results.json"))


if __name__ == "__main__":
    main()
