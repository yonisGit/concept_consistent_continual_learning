"""
Method registry: which data each variant needs and where each loss term is applied.

| name            | replay | L_concept on          | background term                         |
|-----------------|--------|-----------------------|-----------------------------------------|
| c3l             | yes    | replay                | cosine stability vs S_ref (Eq. 5)       |
| c3l_pp (C3L++)  | yes    | replay + current      | cosine stability on replay              |
| c3l_noreplay    | no     | current               | suppression ||M_bg * S||^2 on current    |
| c3l_fullreplay  | yes    | replay, target M*S_ref| ||M_bg * (S - S_ref)||^2 on replay        |
| er              | yes    | -                     | -                                       |
| finetune        | no     | -                     | -                                       |
| joint           | -      | -                     | - (i.i.d. joint training)               |

Component ablations of Table "ablations" are obtained with ``method.lambda_concept=0``
(w/o L_concept), ``method.lambda_stab=0`` (w/o L_stab) or ``method.name=er`` (w/o both).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    uses_replay: bool
    replay_masks: bool          # store M_concept in the buffer
    replay_ref: bool            # store S_ref in the buffer
    reg_on_replay: bool         # apply the C3L regulariser on replayed samples
    concept_on_current: bool    # apply L_concept on current-task samples
    stab_on_current: bool       # apply the (suppression) background term on current samples
    joint: bool = False


def get_method_spec(mcfg) -> MethodSpec:
    n = mcfg.name
    if n == "c3l":
        cur = bool(mcfg.concept_on_current)
        return MethodSpec(n, True, True, True, True, cur, False)
    if n == "c3l_pp":
        return MethodSpec(n, True, True, True, True, True, False)
    if n == "c3l_noreplay":
        return MethodSpec(n, False, False, False, False, True, True)
    if n == "c3l_fullreplay":
        return MethodSpec(n, True, True, True, True, False, False)
    if n == "er":
        return MethodSpec(n, True, False, False, False, False, False)
    if n == "finetune":
        return MethodSpec(n, False, False, False, False, False, False)
    if n == "joint":
        return MethodSpec(n, False, False, False, False, False, False, joint=True)
    raise ValueError(f"Unknown method {n}")


def needs_concepts(spec: MethodSpec) -> bool:
    return spec.replay_masks or spec.concept_on_current
