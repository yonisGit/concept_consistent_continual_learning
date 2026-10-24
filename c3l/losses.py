"""
C3L objectives (Sec. 3.3).

    L_total   = L_task + lambda_concept * L_concept + lambda_stab * L_stab                (Eq. 2)
    L_concept = 1/|M| sum BCE(S_x * M_concept, M_concept)                                 (Eq. 3)
    L_stab    = 1/|M| sum ( 1 - <S_x*M_bg, S_ref*M_bg> / (||S_x*M_bg|| ||S_ref*M_bg|| + eps) )  (Eq. 5)

plus the variants used in the ablations:

* C3L-NoReplay (Appendix D):  L_concept on current samples, L_stab = ||M_bg * S_x||^2  (suppression)
* C3L-FullReplay (Appendix D): L_concept = BCE(M*S_x, M*S_ref),  L_stab = ||M_bg * (S_x - S_ref)||^2
* background = suppress / none (Table abl_background)

All maps are (B,H,W) and min-max normalised to [0,1]. Per-sample losses are
averaged over pixels, then over the batch (= 1/|M| sum over buffer samples
in expectation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_BCE_EPS = 1e-6


def concept_overlap_loss(s: torch.Tensor, m_concept: torch.Tensor, kind: str = "bce",
                         target: torch.Tensor | None = None) -> torch.Tensor:
    """
    Eq. 3. ``target`` defaults to ``M_concept`` (C3L); FullReplay passes ``M * S_ref``.
    Outside the concept region the input is 0 and the target is 0, which
    contributes ~0 to the BCE, so the loss acts on concept pixels only.
    """
    m = m_concept.float()
    pred = s * m
    tgt = m if target is None else target
    if kind == "bce":
        per_pix = F.binary_cross_entropy(pred.clamp(_BCE_EPS, 1 - _BCE_EPS), tgt, reduction="none")
    elif kind == "l1":
        per_pix = (pred - tgt).abs()
    elif kind == "l2":
        per_pix = (pred - tgt) ** 2
    else:
        raise ValueError(kind)
    return per_pix.flatten(1).mean(1).mean()


def background_cosine_loss(s: torch.Tensor, s_ref: torch.Tensor, m_bg: torch.Tensor,
                           eps: float = 1e-8) -> torch.Tensor:
    """Eq. 5: penalises *spatial* drift of background relevance, magnitude is free."""
    a = (s * m_bg).flatten(1)
    b = (s_ref * m_bg).flatten(1)
    cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)
    return (1.0 - cos).mean()


def background_distance_loss(s: torch.Tensor, s_ref: torch.Tensor, m_bg: torch.Tensor, kind: str) -> torch.Tensor:
    d = (s - s_ref) * m_bg
    per = d.abs() if kind == "l1" else d ** 2
    return per.flatten(1).mean(1).mean()


def background_suppression_loss(s: torch.Tensor, m_bg: torch.Tensor) -> torch.Tensor:
    """||M_bg * S_x||_2^2 (pixel-averaged) - explicit suppression used by C3L-NoReplay."""
    return ((s * m_bg) ** 2).flatten(1).mean(1).mean()


class C3LRegularizer:
    """Computes lambda_concept * L_concept + lambda_stab * L_stab for one batch."""

    def __init__(self, mcfg):
        self.lc = mcfg.lambda_concept
        self.ls = mcfg.lambda_stab
        self.concept_kind = mcfg.concept_loss
        self.stab_kind = mcfg.stab_loss
        self.background = mcfg.background
        self.variant = mcfg.name

    def concept_term(self, s, m_concept, s_ref=None):
        if self.variant == "c3l_fullreplay":
            return concept_overlap_loss(s, m_concept, self.concept_kind, target=m_concept.float() * s_ref)
        return concept_overlap_loss(s, m_concept, self.concept_kind)

    def background_term(self, s, m_concept, s_ref=None):
        m_bg = 1.0 - m_concept.float()
        if self.background == "none":
            return s.new_zeros(())
        if self.background == "suppress" or s_ref is None:
            return background_suppression_loss(s, m_bg)
        if self.variant == "c3l_fullreplay":
            return background_distance_loss(s, s_ref, m_bg, "l2")
        if self.stab_kind == "cosine":
            return background_cosine_loss(s, s_ref, m_bg)
        return background_distance_loss(s, s_ref, m_bg, self.stab_kind)

    def __call__(self, s, m_concept, s_ref=None, apply_concept=True, apply_stab=True):
        s = s.float()  # BCE is not autocast-safe; keep regularisers in fp32
        s_ref = s_ref.float() if s_ref is not None else None
        zero = s.new_zeros(())
        l_c = self.concept_term(s, m_concept, s_ref) if (apply_concept and self.lc > 0) else zero
        l_s = self.background_term(s, m_concept, s_ref) if (apply_stab and self.ls > 0) else zero
        return self.lc * l_c + self.ls * l_s, {"l_concept": float(l_c.detach()), "l_stab": float(l_s.detach())}
