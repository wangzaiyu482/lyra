# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Gaussian decoder implementation."""
from __future__ import annotations

import math
from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import GaussianDecoderConfig
from .subsampling import SubsampleResult, SubsamplerFn, subsample_tokens_and_rays


class Renderer(Protocol):
    """Protocol describing the renderer interface expected by :class:`GaussianDecoder`."""

    def render(self, gaussians: torch.Tensor, *args, **kwargs):
        ...


class GaussianDecoder(nn.Module):
    """Decode latent features and rays into Gaussian parameters.

    The module mirrors Lyra's ``forward_gaussians`` / ``gaussian_processing`` pipeline but
    is packaged as a clean, self-contained PyTorch module that accepts an arbitrary
    renderer implementation.  It takes care of subsampling, Gaussian activation
    functions and pruning so the host project only needs to provide features and the
    corresponding ray bundles.
    """

    def __init__(
        self,
        config: GaussianDecoderConfig,
        renderer: Optional[Renderer] = None,
        subsampler: SubsamplerFn = subsample_tokens_and_rays,
    ) -> None:
        super().__init__()
        self.config = config
        self.renderer = renderer
        self._subsampler = subsampler

        output_dims = config.output_dims
        if config.gaussians_predict_offset:
            output_dims += 3
        self.output_dims = output_dims

        scale_cap = config.gaussian_scale_cap
        self._scale_shift = 1 - math.log(scale_cap)
        self._scale_cap = scale_cap
        self.opacity_act = lambda x: torch.sigmoid(x - 2.0)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5

    def forward(
        self,
        features: torch.Tensor,
        rays_os: torch.Tensor,
        rays_ds: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert features into Gaussian primitives.

        Args:
            features: Tensor of shape ``(B, C, T, H, W)`` containing decoded tokens.
            rays_os: Tensor of shape ``(B, T, 3, H, W)`` with ray origins.
            rays_ds: Tensor of shape ``(B, T, 3, H, W)`` with ray directions.
            training: Whether the decoder is in training mode.  Controls the behaviour
                of learned subsampling masks.
        Returns:
            A tuple containing the Gaussian tensor ``(B, N, 14)`` and an optional
            subsampling mask (present when ``config.keep_eval_mask`` is set).
        """

        features, mask_logits = self._prepare_features(features)

        if self.config.sub_sample_gaussians and self.config.sub_sample_gaussians_factor is not None:
            subsample: SubsampleResult = self._subsampler(
                features,
                rays_os,
                rays_ds,
                mask_logits,
                self.config.sub_sample_gaussians_factor,
                self.config.sub_sample_gaussians_type,
                self.config.sub_sample_gaussians_type_tokens,
                self.config.sub_sample_gaussians_temperature,
                training,
            )
            features = subsample.features
            rays_os = subsample.rays_os
            rays_ds = subsample.rays_ds
            mask_logits = subsample.mask
        else:
            features = rearrange(features, "b c t h w -> b (t h w) c")
            rays_os = rearrange(rays_os, "b t c h w -> b (t h w) c")
            rays_ds = rearrange(rays_ds, "b t c h w -> b (t h w) c")

        gaussians = self._gaussian_processing(features, rays_os, rays_ds)
        gaussians = self._gaussian_pruning(gaussians)

        if not self.config.keep_eval_mask and not training:
            mask_logits = None
        return gaussians, mask_logits

    def render(self, gaussians: torch.Tensor, *args, **kwargs):
        """Render Gaussians via the provided renderer."""

        if self.renderer is None:
            raise RuntimeError("No renderer was supplied when constructing GaussianDecoder")
        kwargs.setdefault("bg_color", gaussians.new_tensor(self.config.deferred_background_color))
        kwargs.update(self.config.renderer_kwargs)
        return self.renderer.render(gaussians, *args, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers

    def _prepare_features(self, features: torch.Tensor):
        mask_logits = None
        if self.config.sub_sample_gaussians_factor is not None and self.config.sub_sample_gaussians:
            if self.config.sub_sample_gaussians_type == "learned":
                features, mask_logits = features[:, :-1], features[:, [-1]]
        return features, mask_logits

    def _gaussian_processing(
        self,
        features: torch.Tensor,
        rays_os: torch.Tensor,
        rays_ds: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.gaussians_predict_offset:
            pos_offset = features[..., -3:]
            features = features[..., :-3]
        distance, rgb, scaling, rotation, opacity = features.split([1, 3, 3, 4, 1], dim=-1)

        weights = torch.sigmoid(distance + self.config.pre_sigmoid_distance_shift)
        depths = self.config.dnear * (1 - weights) + self.config.dfar * weights
        pos = rays_os + rays_ds * depths

        if self.config.gaussians_predict_offset and self.config.use_gaussians_predict_offset:
            pos_offset = self._apply_offset_activation(pos_offset)
            pos = pos + pos_offset

        opacity = self.opacity_act(opacity)
        scale = self._scale_activation(scaling)
        rotation = self.rot_act(rotation)
        rgbs = self.rgb_act(rgb)
        return torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)

    def _gaussian_pruning(self, gaussians: torch.Tensor) -> torch.Tensor:
        prune_ratio = self.config.gaussians_prune_ratio
        if prune_ratio <= 0:
            return gaussians

        opacity = gaussians[:, :, [3]]
        num_gaussians = gaussians.shape[1]
        keep_ratio = 1 - prune_ratio
        random_ratio = keep_ratio * self.config.gaussians_random_ratio
        keep_ratio = keep_ratio - random_ratio

        num_keep = int(num_gaussians * keep_ratio)
        num_keep_random = int(num_gaussians * random_ratio)

        idx_sort = opacity.argsort(dim=1, descending=True)
        keep_idx = idx_sort[:, :num_keep]

        if num_keep_random > 0:
            rest_idx = idx_sort[:, num_keep:]
            random_perm = torch.randperm(rest_idx.shape[1], device=gaussians.device)
            random_idx = rest_idx[:, random_perm[:num_keep_random]]
            keep_idx = torch.cat([keep_idx, random_idx], dim=1)

        return gaussians.gather(1, keep_idx.expand(-1, -1, gaussians.shape[-1]))

    def _scale_activation(self, x: torch.Tensor) -> torch.Tensor:
        cap = torch.tensor([self._scale_cap], device=x.device, dtype=x.dtype)
        return torch.minimum(torch.exp(x - self._scale_shift), cap)

    def _apply_offset_activation(self, offset: torch.Tensor) -> torch.Tensor:
        act = self.config.gaussians_predict_offset_act
        min_val, max_val = self.config.gaussians_predict_offset_range

        if act == "clamp":
            return offset.clamp(min_val, max_val)
        if act == "tanh":
            return max_val * torch.tanh(offset)
        raise ValueError(f"Unknown offset activation: {act}")
