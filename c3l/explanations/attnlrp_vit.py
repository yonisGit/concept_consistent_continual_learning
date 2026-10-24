"""
Differentiable AttnLRP (Achtibat et al., 2024) for timm Vision Transformers.

The ViT forward is re-implemented explicitly so that AttnLRP's rules can be
expressed as modified gradients (the same idea as LXT's "efficient" mode), making
Input x Gradient equal to the AttnLRP relevance while staying differentiable
w.r.t. the weights (``create_graph=True``):

* Linear / patch-embedding / residual additions ..... epsilon-rule (``stabilize``)
* LayerNorm .......................................... identity rule on the normaliser (std detached)
* GELU (and other point-wise activations) ............ identity rule
* softmax ............................................ plain gradient: Input x Gradient of softmax
                                                        reproduces the AttnLRP softmax rule
                                                        R_i = x_i (R_i - s_i sum_j R_j)
* q @ k^T and A @ v .................................. uniform rule (gradient halved)

The same forward (``lrp=False``) also exposes the attention maps, which is used
by Grad-CAM-ViT, T-Attr and IIA.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from c3l.explanations.base import halve_gradient, identity_rule, stabilize


def _layernorm(ln: nn.Module, x: torch.Tensor, lrp: bool, eps: float) -> torch.Tensor:
    if not lrp or not isinstance(ln, nn.LayerNorm):
        return ln(x)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    std = (var + ln.eps).sqrt().detach()  # identity rule: the normaliser is treated as a constant
    y = (x - mean) / std
    if ln.weight is not None:
        y = y * ln.weight + ln.bias
    return stabilize(y, eps)


def _linear(layer: nn.Module, x: torch.Tensor, lrp: bool, eps: float) -> torch.Tensor:
    out = layer(x)
    return stabilize(out, eps) if lrp else out


def _attention(attn: nn.Module, x: torch.Tensor, lrp: bool, eps: float,
               store: Optional[List[torch.Tensor]]) -> torch.Tensor:
    B, N, C = x.shape
    H = attn.num_heads
    hd = getattr(attn, "head_dim", C // H)
    qkv = _linear(attn.qkv, x, lrp, eps).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    q = q * attn.scale
    scores = q @ k.transpose(-2, -1)
    if lrp:
        scores = halve_gradient(scores)
    probs = scores.softmax(dim=-1)
    if store is not None:
        store.append(probs)
    out = probs @ v
    if lrp:
        out = halve_gradient(out)
    out = out.transpose(1, 2).reshape(B, N, H * hd)
    if hasattr(attn, "norm"):
        out = _layernorm(attn.norm, out, lrp, eps)
    return _linear(attn.proj, out, lrp, eps)


def _mlp(mlp: nn.Module, x: torch.Tensor, lrp: bool, eps: float) -> torch.Tensor:
    h = _linear(mlp.fc1, x, lrp, eps)
    a = mlp.act(h)
    if lrp:
        a = identity_rule(a, h)
    if hasattr(mlp, "norm"):
        a = _layernorm(mlp.norm, a, lrp, eps)
    return _linear(mlp.fc2, a, lrp, eps)


def _layerscale(ls: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return ls(x)  # Identity or element-wise gamma (linear -> plain gradient is the LRP rule)


def vit_forward(
    vit: nn.Module,
    x: torch.Tensor,
    lrp: bool = False,
    eps: float = 1e-6,
    attn_store: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Deterministic ViT forward (dropout / drop-path disabled)."""
    e = eps if lrp else 0.0
    h = vit.patch_embed.proj(x)
    if lrp:
        h = stabilize(h, e)
    if h.ndim == 4:
        h = h.flatten(2).transpose(1, 2)
    h = vit.patch_embed.norm(h)
    h = vit._pos_embed(h)  # concat [CLS] (+ registers) and add positional embeddings (bias terms)
    if lrp:
        h = stabilize(h, e)
    h = vit.norm_pre(h)
    for blk in vit.blocks:
        a = _attention(blk.attn, _layernorm(blk.norm1, h, lrp, e), lrp, e, attn_store)
        h = h + _layerscale(blk.ls1, a)
        if lrp:
            h = stabilize(h, e)
        m = _mlp(blk.mlp, _layernorm(blk.norm2, h, lrp, e), lrp, e)
        h = h + _layerscale(blk.ls2, m)
        if lrp:
            h = stabilize(h, e)
    h = _layernorm(vit.norm, h, lrp, e)
    if getattr(vit, "global_pool", "token") == "avg":
        h = h[:, getattr(vit, "num_prefix_tokens", 1):].mean(1)
    else:
        h = h[:, 0]
    h = _layernorm(vit.fc_norm, h, lrp, e)
    return _linear(vit.head, h, lrp, e)


def patch_grid(vit: nn.Module, x: torch.Tensor):
    ps = vit.patch_embed.patch_size
    ps = ps[0] if isinstance(ps, (tuple, list)) else ps
    return x.shape[-2] // ps, x.shape[-1] // ps


def attnlrp_vit(vit: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, create_graph: bool) -> torch.Tensor:
    """Returns the (B,C,H,W) AttnLRP relevance of the target logits ``y``."""
    x = x.detach().requires_grad_(True)
    logits = vit_forward(vit, x, lrp=True, eps=eps)
    score = logits.gather(1, y.view(-1, 1)).sum()
    (grad,) = torch.autograd.grad(score, x, create_graph=create_graph)
    return x * grad
