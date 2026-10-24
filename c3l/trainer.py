"""
Class-incremental training loop for C3L and its variants (Sec. 3, Appendix A).

For every task t:
  1. train on D_t (+ replay batches from M) with
         L_total = CE + lambda_concept * L_concept + lambda_stab * L_stab
     where the C3L terms use differentiable LRP / AttnLRP maps (double back-prop);
  2. evaluate on the test sets of tasks 1..t  -> column t of the accuracy matrix;
  3. insert samples of D_t into M together with their concept masks and their
     reference saliency S_ref computed by the current model (frozen afterwards);
  4. optional analyses (crowding-out after selected tasks).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from c3l.data.scenario import TaskDataset
from c3l.evaluation.crowding import crowding_analysis
from c3l.evaluation.localization import evaluate_localization, load_cub_segmentation
from c3l.evaluation.metrics import accuracy_matrix_metrics, evaluate_tasks
from c3l.experiment import ConceptResources
from c3l.explanations.factory import Explainer, resolve_name
from c3l.losses import C3LRegularizer
from c3l.methods import get_method_spec, needs_concepts
from c3l.models import build_model
from c3l.replay import ReplayBuffer
from c3l.utils import get_logger, save_json

logger = get_logger()


class ContinualTrainer:
    def __init__(self, cfg, splits, scenario, device: torch.device, run_dir: str):
        self.cfg = cfg
        self.splits = splits
        self.scenario = scenario
        self.device = device
        self.run_dir = run_dir
        self.spec = get_method_spec(cfg.method)
        self.model = build_model(cfg, scenario.num_classes).to(device)
        self.explainer = Explainer.from_cfg(cfg.explainer)
        self.eval_explainer = Explainer.from_cfg(cfg.explainer, name=cfg.explainer.eval_name)
        self.regularizer = C3LRegularizer(cfg.method)
        self.label_map = scenario.orig_to_new

        wants_concepts = needs_concepts(self.spec) or bool(cfg.analysis.crowding_tasks)
        self.concepts = ConceptResources(cfg, splits.class_names) if wants_concepts else None
        self.buffer = (
            ReplayBuffer(cfg.replay.buffer_size, self.spec.replay_masks, self.spec.replay_ref, seed=cfg.seed)
            if self.spec.uses_replay else None
        )
        T = scenario.num_tasks
        self.acc_matrix = np.zeros((T, T))
        self.history: Dict[str, list] = {"loss": [], "task_time_s": [], "crowding": {}}
        logger.info(f"Method {self.spec.name} | explainer {resolve_name(self.explainer.name, self.model.arch)} | "
                    f"lambda_concept={cfg.method.lambda_concept} lambda_stab={cfg.method.lambda_stab}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def _loader(self, dataset, indices, train: bool, with_masks: bool = False, batch_size: Optional[int] = None):
        mask_fn = self.concepts.provider(dataset) if with_masks else None
        if mask_fn is not None:
            mask_fn.precompute(indices)  # offline grounding, before training on the task
        ds = TaskDataset(dataset, indices, self.label_map, train=train, base_size=self.cfg.data.base_size,
                         crop_size=self.cfg.data.crop_size, mask_fn=mask_fn)
        bs = batch_size or (self.cfg.train.batch_size if train else self.cfg.train.eval_batch_size)
        return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=self.cfg.num_workers,
                          pin_memory=self.device.type == "cuda", drop_last=False)

    def _test_loaders(self, upto: int) -> List[DataLoader]:
        return [self._loader(self.splits.test, self.scenario.test_indices(i), train=False) for i in range(upto + 1)]

    # ------------------------------------------------------------------ #
    # Optimisation
    # ------------------------------------------------------------------ #
    def _optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        t = self.cfg.train
        if t.optimizer == "sgd":
            return torch.optim.SGD(params, lr=t.lr, momentum=t.momentum, weight_decay=t.weight_decay)
        if t.optimizer == "adam":
            return torch.optim.Adam(params, lr=t.lr, weight_decay=t.weight_decay)
        raise ValueError(t.optimizer)

    def _step(self, batch, num_seen: int, optimizer, scaler) -> Dict[str, float]:
        dev = self.device
        x = batch["image"].to(dev, non_blocking=True)
        y = batch["label"].to(dev, non_blocking=True)
        rb = None
        if self.buffer is not None and len(self.buffer) > 0:
            rb = self.buffer.sample(self.cfg.train.replay_batch_size, dev)

        use_amp = self.cfg.amp and dev.type == "cuda"
        with torch.autocast(dev.type, enabled=use_amp):
            xs = torch.cat([x, rb["image"]]) if rb is not None else x
            ys = torch.cat([y, rb["label"]]) if rb is not None else y
            logits = self.model(xs)[:, :num_seen]
            l_task = F.cross_entropy(logits, ys)

        reg = torch.zeros((), device=dev)
        logs = {"l_concept": 0.0, "l_stab": 0.0}
        with torch.autocast(dev.type, enabled=False):
            if self.spec.reg_on_replay and rb is not None:
                s = self.explainer(self.model, rb["image"], rb["label"], create_graph=True)
                r, lg = self.regularizer(s, rb["mask"], rb.get("s_ref"))
                reg = reg + r
                logs = {k: logs[k] + lg[k] for k in logs}
            if self.spec.concept_on_current:
                s = self.explainer(self.model, x, y, create_graph=True)
                r, lg = self.regularizer(s, batch["mask"].to(dev), None,
                                         apply_concept=True, apply_stab=self.spec.stab_on_current)
                reg = reg + r
                logs = {k: logs[k] + lg[k] for k in logs}

        total = l_task + reg
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total).backward()
            if self.cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if self.cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            optimizer.step()
        return {"l_task": float(l_task.detach()), **logs, "total": float(total.detach())}

    def train_on(self, indices: List[int], num_seen: int, task_id: int) -> None:
        loader = self._loader(self.splits.train, indices, train=True, with_masks=self.spec.concept_on_current)
        optimizer = self._optimizer()
        scaler = torch.cuda.amp.GradScaler() if (self.cfg.amp and self.device.type == "cuda") else None
        step = 0
        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            running: Dict[str, float] = {}
            for batch in loader:
                logs = self._step(batch, num_seen, optimizer, scaler)
                for k, v in logs.items():
                    running[k] = running.get(k, 0.0) + v
                step += 1
                if step % self.cfg.train.log_every == 0:
                    logger.info(f"task {task_id + 1} ep {epoch + 1} step {step} | "
                                + " ".join(f"{k}={v:.4f}" for k, v in logs.items()))
            n = max(len(loader), 1)
            epoch_log = {k: v / n for k, v in running.items()}
            self.history["loss"].append({"task": task_id + 1, "epoch": epoch + 1, **epoch_log})
            logger.info(f"task {task_id + 1} epoch {epoch + 1}/{self.cfg.train.epochs} | "
                        + " ".join(f"{k}={v:.4f}" for k, v in epoch_log.items()))

    # ------------------------------------------------------------------ #
    # Replay buffer update
    # ------------------------------------------------------------------ #
    def update_buffer(self, task_id: int) -> None:
        if self.buffer is None:
            return
        self.model.eval()
        mask_fn = self.concepts.provider(self.splits.train) if self.spec.replay_masks else None
        ref_fn = (lambda x, y: self.explainer(self.model, x, y, create_graph=False)) if self.spec.replay_ref else None
        self.buffer.add_task(
            task_id, self.splits.train, self.scenario.train_indices(task_id), self.label_map,
            self.cfg.data.base_size, self.cfg.data.crop_size, mask_fn=mask_fn, ref_fn=ref_fn, device=self.device,
        )
        logger.info(f"buffer: {len(self.buffer)} samples | " +
                    " ".join(f"{k}={v:.1f}" for k, v in self.buffer.memory_report().items()))

    # ------------------------------------------------------------------ #
    # Analyses
    # ------------------------------------------------------------------ #
    def run_crowding(self, task_id: int) -> None:
        idx = self.scenario.test_indices(0)
        loader = self._loader(self.splits.test, idx, train=False, with_masks=True, batch_size=32)
        analysis_explainer = Explainer.from_cfg(self.cfg.explainer)  # AttnLRP / LRP
        res = crowding_analysis(self.model, loader, analysis_explainer, self.device)
        self.history["crowding"][f"after_T{task_id + 1}"] = res
        logger.info(f"crowding after T{task_id + 1}: {res}")

    def run_localization(self) -> Dict[str, float]:
        if not self.splits.info.get("has_segmentation"):
            logger.warning("Localization requested but dataset has no segmentations; skipping.")
            return {}
        idx = [i for t in range(self.scenario.num_tasks) for i in self.scenario.test_indices(t)]
        loader = self._loader(self.splits.test, idx, train=False, batch_size=32)
        recs = self.splits.test.records
        gt_fn = lambda i: load_cub_segmentation(recs[i].extras["segmentation"], self.cfg.data.base_size,
                                                self.cfg.data.crop_size)
        res = evaluate_localization(self.model, loader, self.eval_explainer, gt_fn, self.device)
        logger.info(f"localization ({self.eval_explainer.name}): {res}")
        return res

    # ------------------------------------------------------------------ #
    def run(self) -> Dict:
        T = self.scenario.num_tasks
        t_start = time.time()
        if self.spec.joint:
            return self._run_joint()
        for t in range(T):
            tic = time.time()
            n_seen = self.scenario.num_seen_classes(t)
            logger.info(f"===== Task {t + 1}/{T}: classes {self.scenario.task_classes(t)[0]}..{n_seen - 1} =====")
            self.train_on(self.scenario.train_indices(t), n_seen, t)
            accs = evaluate_tasks(self.model, self._test_loaders(t), n_seen, self.device, self.cfg.amp)
            self.acc_matrix[: t + 1, t] = accs
            logger.info(f"after T{t + 1}: per-task acc {[round(a, 2) for a in accs]} | avg {np.mean(accs):.2f}")
            if t < T - 1 or self.cfg.analysis.crowding_tasks:
                self.update_buffer(t)
            self.history["task_time_s"].append(time.time() - tic)
            if (t + 1) in list(self.cfg.analysis.crowding_tasks or []):
                self.run_crowding(t)
            if self.cfg.analysis.save_checkpoints:
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, f"model_T{t + 1}.pt"))
        results = {
            "metrics": accuracy_matrix_metrics(self.acc_matrix),
            "acc_matrix": self.acc_matrix.tolist(),
            "history": self.history,
            "wall_clock_s": time.time() - t_start,
            "buffer_memory_mb": self.buffer.memory_report() if self.buffer is not None else None,
        }
        if self.cfg.analysis.localization:
            results["localization"] = self.run_localization()
        logger.info(f"FINAL {results['metrics']}")
        save_json(results, os.path.join(self.run_dir, "results.json"))
        return results

    def _run_joint(self) -> Dict:
        """Joint training: i.i.d. training on the union of all tasks."""
        T = self.scenario.num_tasks
        tic = time.time()
        idx = [i for t in range(T) for i in self.scenario.train_indices(t)]
        self.train_on(idx, self.scenario.num_classes, task_id=T - 1)
        accs = evaluate_tasks(self.model, self._test_loaders(T - 1), self.scenario.num_classes, self.device, self.cfg.amp)
        results = {"metrics": {"ACC": float(np.mean(accs))}, "per_task_acc": accs, "wall_clock_s": time.time() - tic}
        if self.cfg.analysis.localization:
            results["localization"] = self.run_localization()
        save_json(results, os.path.join(self.run_dir, "results.json"))
        return results
