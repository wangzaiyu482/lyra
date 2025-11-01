# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token subsampling utilities used by the Gaussian decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


@dataclass
class SubsampleResult:
    features: torch.Tensor
    rays_os: torch.Tensor
    rays_ds: torch.Tensor
    mask: Optional[torch.Tensor]


SubsamplerFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Sequence[int],
        str,
        str,
        float,
        bool,
    ],
    SubsampleResult,
]


def subsample_tokens_and_rays(
    features: torch.Tensor,
    rays_os: torch.Tensor,
    rays_ds: torch.Tensor,
    mask_logits: Optional[torch.Tensor],
    factor: Sequence[int],
    mode: str,
    token_mode: str,
    temperature: float,
    training: bool,
) -> SubsampleResult:
    """Subsample tokens/rays either randomly or via learned logits."""

    device = features.device
    factor_tensor = torch.as_tensor(factor, device=device, dtype=torch.long)
    x_shape = torch.as_tensor(features.shape[-3:], device=device, dtype=torch.long)
    target_shape = torch.div(x_shape, factor_tensor, rounding_mode="floor")
    t_out, h_out, w_out = target_shape.tolist()

    if mode == "random":
        result = _random_subsample(features, rays_os, rays_ds, t_out, h_out, w_out)
        mask = None
    elif mode == "learned":
        if mask_logits is None:
            raise ValueError("mask_logits must be provided for learned subsampling")
        result, mask = _learned_subsample(
            features,
            rays_os,
            rays_ds,
            mask_logits,
            t_out,
            h_out,
            w_out,
            token_mode,
            temperature,
            training,
        )
    else:
        raise ValueError(f"Unknown subsampling mode: {mode}")

    if training:
        mask = None

    return SubsampleResult(result[0], result[1], result[2], mask)


def _random_subsample(
    features: torch.Tensor,
    rays_os: torch.Tensor,
    rays_ds: torch.Tensor,
    t_out: int,
    h_out: int,
    w_out: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if t_out <= 0 or h_out <= 0 or w_out <= 0:
        raise ValueError("Requested subsample size must be positive")

    batch, _, t_in, h_in, w_in = features.shape
    indices = _sample_spatio_temporal_indices(
        (batch, t_in, h_in, w_in), (t_out, h_out, w_out), device=features.device
    )

    features = rearrange(features, "b c t h w -> b t h w c")
    rays_os = rearrange(rays_os, "b t c h w -> b t h w c")
    rays_ds = rearrange(rays_ds, "b t c h w -> b t h w c")

    features = _query_with_indices(indices, features)
    rays_os = _query_with_indices(indices, rays_os)
    rays_ds = _query_with_indices(indices, rays_ds)
    return features, rays_os, rays_ds


def _learned_subsample(
    features: torch.Tensor,
    rays_os: torch.Tensor,
    rays_ds: torch.Tensor,
    mask_logits: torch.Tensor,
    t_out: int,
    h_out: int,
    w_out: int,
    token_mode: str,
    temperature: float,
    training: bool,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    rays_os = rearrange(rays_os, "b t c h w -> b c t h w")
    rays_ds = rearrange(rays_ds, "b t c h w -> b c t h w")

    if token_mode == "local":
        features, (rays_os, rays_ds), mask = _process_tensors(
            tokens=features,
            mask_logits=mask_logits,
            other_tensors=[rays_os, rays_ds],
            k_t=t_out,
            k_hw=h_out * w_out,
            temperature=temperature,
            training=training,
        )
    elif token_mode == "global":
        features, (rays_os, rays_ds), mask = _process_tensors(
            tokens=features,
            mask_logits=mask_logits,
            other_tensors=[rays_os, rays_ds],
            total_k=t_out * h_out * w_out,
            temperature=temperature,
            training=training,
        )
    else:
        raise ValueError(f"Unknown token selection mode: {token_mode}")

    features = rearrange(features, "b c n -> b n c")
    rays_os = rearrange(rays_os, "b c n -> b n c")
    rays_ds = rearrange(rays_ds, "b c n -> b n c")
    return (features, rays_os, rays_ds), mask


def _sample_spatio_temporal_indices(
    dimensions: Sequence[int],
    target: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, t, h, w = dimensions
    m_t, m_h, m_w = target

    if not (1 <= m_t <= t and 1 <= m_h <= h and 1 <= m_w <= w):
        raise ValueError("Requested samples exceed tensor dimensions")

    t_indices = torch.multinomial(torch.ones(t, device=device).expand(b, -1), m_t, replacement=False)
    h_indices = torch.multinomial(torch.ones(h, device=device).expand(b, -1), m_h, replacement=False)
    w_indices = torch.multinomial(torch.ones(w, device=device).expand(b, -1), m_w, replacement=False)

    t_grid = t_indices[:, :, None, None]
    h_grid = h_indices[:, None, :, None]
    w_grid = w_indices[:, None, None, :]

    t_grid = t_grid.expand(-1, m_t, m_h, m_w)
    h_grid = h_grid.expand(-1, m_t, m_h, m_w)
    w_grid = w_grid.expand(-1, m_t, m_h, m_w)

    b_idx = torch.arange(b, device=device)[:, None].expand(b, m_t * m_h * m_w)
    t_idx = t_grid.reshape(b, -1)
    h_idx = h_grid.reshape(b, -1)
    w_idx = w_grid.reshape(b, -1)
    return b_idx, t_idx, h_idx, w_idx


def _query_with_indices(
    indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tensor: torch.Tensor,
) -> torch.Tensor:
    b_idx, t_idx, h_idx, w_idx = indices
    _, t, h, w, c = tensor.shape
    flat = rearrange(tensor, "b t h w c -> b (t h w) c")
    flat_idx = (t_idx * h * w) + (h_idx * w) + w_idx
    return torch.gather(flat, dim=1, index=flat_idx.unsqueeze(-1).expand(-1, -1, c))


def _process_tensors(
    tokens: torch.Tensor,
    mask_logits: torch.Tensor,
    other_tensors: List[torch.Tensor],
    total_k: Optional[int] = None,
    k_t: Optional[int] = None,
    k_hw: Optional[int] = None,
    temperature: float = 1.0,
    eps: float = 1e-6,
    training: bool = True,
    soft_inference: bool = True,
):
    b, c, t, h, w = tokens.shape
    mask_logits = mask_logits.squeeze(1)

    if training or soft_inference:
        method = "softmax"
        hard = True
    else:
        method = "topk"
        hard = False

    if total_k is not None:
        mask = _global_selection(mask_logits, total_k, method, temperature, hard, eps)
    elif k_t is not None and k_hw is not None:
        mask = _structured_selection(mask_logits, k_t, k_hw, method, temperature, hard, eps)
    else:
        raise ValueError("Provide either total_k or both k_t and k_hw")

    tokens_out, others_out = _apply_mask_and_select(tokens, other_tensors, mask)
    return tokens_out, others_out, mask


def _sample_gumbel(shape: torch.Size, eps: float = 1e-6, device=None, dtype=None) -> torch.Tensor:
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u.clamp(min=eps, max=1 - eps)))


def _select_topk(
    logits: torch.Tensor,
    k: int,
    method: str,
    temperature: float,
    hard: bool,
    eps: float,
) -> torch.Tensor:
    if method == "topk":
        _, topk_idx = torch.topk(logits, k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, topk_idx, 1.0)
    elif method == "softmax":
        gumbel_noise = _sample_gumbel(logits.shape, eps=eps, device=logits.device, dtype=logits.dtype)
        y = (logits + gumbel_noise) / temperature
        y_soft = F.softmax(y, dim=-1)
        if hard:
            topk_idx = y_soft.topk(k, dim=-1).indices
            hard_mask = torch.zeros_like(y_soft).scatter(-1, topk_idx, 1.0)
            mask = hard_mask - y_soft.detach() + y_soft
        else:
            mask = y_soft
    else:
        raise ValueError(f"Unknown selection method: {method}")
    return mask


def _global_selection(
    mask_logits: torch.Tensor,
    total_k: int,
    method: str,
    temperature: float,
    hard: bool,
    eps: float,
) -> torch.Tensor:
    b, t, h, w = mask_logits.shape
    mask_flat = _select_topk(mask_logits.reshape(b, -1), total_k, method, temperature, hard, eps)
    return mask_flat.reshape(b, t, h, w)


def _structured_selection(
    mask_logits: torch.Tensor,
    k_t: int,
    k_hw: int,
    method: str,
    temperature: float,
    hard: bool,
    eps: float,
) -> torch.Tensor:
    b, t, h, w = mask_logits.shape

    logits_t = mask_logits.mean(dim=[2, 3])
    mask_t = _select_topk(logits_t, k_t, method, temperature, hard, eps)

    mask_spatial = []
    for b_idx in range(b):
        mask_b = []
        for t_idx in range(t):
            logits_hw = mask_logits[b_idx, t_idx].reshape(-1)
            mask_hw = _select_topk(logits_hw.unsqueeze(0), k_hw, method, temperature, hard, eps)
            mask_b.append(mask_hw.reshape(h, w))
        mask_b = torch.stack(mask_b, dim=0)
        mask_spatial.append(mask_b)
    mask_spatial = torch.stack(mask_spatial, dim=0)
    return mask_spatial * mask_t.unsqueeze(-1).unsqueeze(-1)


def _apply_mask_and_select(
    tokens: torch.Tensor,
    other_tensors: Iterable[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    b, c, t, h, w = tokens.shape
    n = t * h * w

    tokens_flat = tokens.reshape(b, c, n)
    mask_flat = mask.reshape(b, n)

    selected_tokens = []
    selected_others: List[List[torch.Tensor]] = [[] for _ in other_tensors]

    for batch_idx in range(b):
        keep_idx = mask_flat[batch_idx].nonzero(as_tuple=False).squeeze(-1)
        selected_tokens.append(tokens_flat[batch_idx, :, keep_idx])

        for i, tensor in enumerate(other_tensors):
            tensor_flat = tensor.reshape(b, -1, n)
            selected_others[i].append(tensor_flat[batch_idx, :, keep_idx])

    tokens_out = torch.stack(selected_tokens, dim=0)
    others_out = [torch.stack(entries, dim=0) for entries in selected_others]
    return tokens_out, others_out
