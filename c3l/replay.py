"""
Replay buffer M (Sec. 3.1 / Appendix A).

* fixed capacity |M| (1000 by default), uniform random sampling, every past task
  contributes equally (class-balanced reservoir per task);
* when a sample *first* enters M after task T_j we store, once and for all:
    - the deterministic (centre-cropped) image,
    - its concept mask M_concept (offline grounding, model-state independent),
    - its reference saliency S_ref computed with the model at the end of T_j.
  S_ref is never refreshed: it is what L_stab anchors the background pattern to.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from c3l.data.base import ImageListDataset
from c3l.data.scenario import load_base_crop
from c3l.data.transforms import center_crop
from c3l.utils import normalize_image, tensor_nbytes


class ReplayBuffer:
    def __init__(self, capacity: int, store_masks: bool, store_ref: bool, seed: int = 0):
        self.capacity = capacity
        self.store_masks = store_masks
        self.store_ref = store_ref
        self.rng = np.random.RandomState(seed)
        self.images = torch.empty(0, dtype=torch.uint8)
        self.labels = torch.empty(0, dtype=torch.long)
        self.tasks = torch.empty(0, dtype=torch.long)
        self.indices = torch.empty(0, dtype=torch.long)
        self.masks = torch.empty(0, dtype=torch.uint8)
        self.s_ref = torch.empty(0, dtype=torch.float16)

    def __len__(self) -> int:
        return int(self.labels.numel())

    # ------------------------------------------------------------------ #
    def _keep(self, keep: torch.Tensor) -> None:
        self.images, self.labels, self.tasks, self.indices = (
            self.images[keep], self.labels[keep], self.tasks[keep], self.indices[keep])
        if self.store_masks:
            self.masks = self.masks[keep]
        if self.store_ref:
            self.s_ref = self.s_ref[keep]

    def _shrink_old_tasks(self, quota: int) -> None:
        keep = []
        for t in self.tasks.unique().tolist():
            idx = (self.tasks == t).nonzero(as_tuple=True)[0]
            keep.append(idx[:quota])  # samples were inserted in random order -> first `quota` is a uniform subset
        if keep:
            self._keep(torch.cat(keep).sort().values)

    @torch.no_grad()
    def add_task(
        self,
        task_id: int,
        dataset: ImageListDataset,
        candidate_indices: List[int],
        label_map: Dict[int, int],
        base_size: int,
        crop_size: int,
        mask_fn: Optional[Callable[[int], torch.Tensor]] = None,
        ref_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
    ) -> None:
        """Rebalance M to ``capacity // (#tasks)`` per task, then insert samples of ``task_id``."""
        n_tasks = int(task_id) + 1
        quota = self.capacity // n_tasks
        self._shrink_old_tasks(quota)
        n_new = min(self.capacity - len(self), len(candidate_indices))
        chosen = self.rng.choice(candidate_indices, size=n_new, replace=False).tolist()

        imgs, labels, masks, refs = [], [], [], []
        for start in range(0, len(chosen), batch_size):
            chunk = chosen[start:start + batch_size]
            x = torch.stack([load_base_crop(dataset, i, base_size, crop_size) for i in chunk])
            y = torch.tensor([label_map[dataset.records[i].label] for i in chunk], dtype=torch.long)
            x_u8 = (x * 255).round().to(torch.uint8)
            x = x_u8.float() / 255.0  # S_ref must be computed on exactly the input that will be replayed
            imgs.append(x_u8)
            labels.append(y)
            if self.store_masks:
                m = torch.stack([center_crop(mask_fn(i), crop_size) for i in chunk]).to(torch.uint8)
                masks.append(m)
            if self.store_ref:
                s = ref_fn(normalize_image(x.to(device)), y.to(device))
                refs.append(s.detach().to("cpu", torch.float16))

        if not chosen:
            return
        self.images = torch.cat([self.images.view(-1, 3, crop_size, crop_size)] + imgs)
        self.labels = torch.cat([self.labels] + labels)
        self.tasks = torch.cat([self.tasks, torch.full((len(chosen),), task_id, dtype=torch.long)])
        self.indices = torch.cat([self.indices, torch.tensor(chosen, dtype=torch.long)])
        if self.store_masks:
            self.masks = torch.cat([self.masks.view(-1, crop_size, crop_size)] + masks)
        if self.store_ref:
            self.s_ref = torch.cat([self.s_ref.view(-1, crop_size, crop_size)] + refs)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = torch.from_numpy(self.rng.choice(len(self), size=min(batch_size, len(self)), replace=False))
        out = {
            "image": normalize_image(self.images[idx].float().div(255.0)).to(device, non_blocking=True),
            "label": self.labels[idx].to(device),
            "task": self.tasks[idx].to(device),
        }
        if self.store_masks:
            out["mask"] = self.masks[idx].to(device)
        if self.store_ref:
            out["s_ref"] = self.s_ref[idx].float().to(device)
        return out

    def memory_report(self) -> Dict[str, float]:
        """Megabytes used by each component (Appendix 'Computational Analysis')."""
        mb = lambda t: tensor_nbytes(t) / 2 ** 20
        rep = {"images_mb": mb(self.images), "concept_masks_mb": 0.0, "reference_saliency_mb": 0.0}
        if self.store_masks:
            # Masks are binary: report bit-packed size, the way they would be stored on disk.
            rep["concept_masks_mb"] = self.masks.numel() / 8 / 2 ** 20
        if self.store_ref:
            rep["reference_saliency_mb"] = mb(self.s_ref)
        rep["total_mb"] = sum(rep.values())
        return rep

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in ("images", "labels", "tasks", "indices", "masks", "s_ref")}
