# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.rendering.gs import GaussianRenderer
from src.rendering.gs_deferred import GaussianRendererDeferred
from src.models.utils.render import subsample_x_and_rays


class GaussianDecoder(nn.Module):
    """Decode convolutional features into Gaussian parameters and render them."""

    def __init__(self, opt, subsample_fn=subsample_x_and_rays):
        super().__init__()
        self.opt = opt
        self._subsample_fn = subsample_fn

        # Channels produced by the decoder before Gaussian post-processing
        output_dims = self.opt.output_dims
        if self.opt.sub_sample_gaussians_type == "learned":
            output_dims += 1
        if self.opt.gaussians_predict_offset:
            output_dims += 3
        self.output_dims = output_dims

        # Activations for Gaussian attributes
        scale_cap = opt.gaussian_scale_cap
        self._scale_shift = 1 - math.log(scale_cap)
        self._scale_cap = scale_cap
        self.opacity_act = lambda x: torch.sigmoid(x - 2.0)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5
        self.dnear = opt.dnear
        self.dfar = opt.dfar

        # Renderer
        if self.opt.deferred_bp:
            self.renderer = GaussianRendererDeferred(opt)
            self._render_kwargs = {"patch_size": self.opt.gs_render_patch_size}
        else:
            self.renderer = GaussianRenderer(opt)
            self._render_kwargs = {}

    def forward(
        self,
        features: torch.Tensor,
        rays_os: torch.Tensor,
        rays_ds: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert convolutional features to Gaussian parameters."""
        features, x_mask = self._prepare_features(features)

        if self.opt.sub_sample_gaussians and self.opt.sub_sample_gaussians_factor is not None:
            features, rays_os, rays_ds, x_mask = self._subsample(features, rays_os, rays_ds, x_mask, training)
        else:
            features = rearrange(features, "b c t h w -> b (t h w) c")
            rays_os = rearrange(rays_os, "b t c h w -> b (t h w) c")
            rays_ds = rearrange(rays_ds, "b t c h w -> b (t h w) c")

        gaussians = self.gaussian_processing(features, rays_os, rays_ds)
        gaussians = self.gaussian_pruning(gaussians)
        return gaussians, x_mask

    def render(self, gaussians: torch.Tensor, cam_view: torch.Tensor, intrinsics: torch.Tensor):
        """Render Gaussians with the configured renderer."""
        if self.opt.deferred_bp:
            bg_color = [1, 1, 1]
        else:
            bg_color = torch.ones(3, dtype=gaussians.dtype, device=gaussians.device)
        return self.renderer.render(
            gaussians,
            cam_view,
            bg_color=bg_color,
            intrinsics=intrinsics,
            **self._render_kwargs,
        )

    def gaussian_processing(self, x: torch.Tensor, rays_os: torch.Tensor, rays_ds: torch.Tensor):
        if self.opt.gaussians_predict_offset:
            pos_offset = x[..., -3:]
            x = x[..., :-3]
        distance, rgb, scaling, rotation, opacity = x.split([1, 3, 3, 4, 1], dim=-1)
        w = torch.sigmoid(distance + self.opt.pre_sigmoid_distance_shift)
        depths = self.dnear * (1 - w) + self.dfar * w
        pos = rays_os + rays_ds * depths

        if self.opt.gaussians_predict_offset and self.opt.use_gaussians_predict_offset:
            if self.opt.gaussians_predict_offset_act == "clamp":
                pos_offset = pos_offset.clamp(
                    self.opt.gaussians_predict_offset_range[0],
                    self.opt.gaussians_predict_offset_range[1],
                )
            elif self.opt.gaussians_predict_offset_act == "tanh":
                pos_offset = self.opt.gaussians_predict_offset_range[1] * torch.tanh(pos_offset)
            pos = pos + pos_offset

        opacity = self.opacity_act(opacity)
        scale = self.scale_act(scaling)
        rotation = self.rot_act(rotation)
        rgbs = self.rgb_act(rgb)
        return torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)

    def gaussian_pruning(self, gaussians: torch.Tensor):
        prune_ratio = self.opt.gaussians_prune_ratio
        if prune_ratio > 0:
            opacity = gaussians[:, :, [3]]
            num_gaussians = gaussians.shape[1]
            keep_ratio = 1 - prune_ratio
            random_ratio = self.opt.gaussians_random_ratio
            random_ratio = keep_ratio * random_ratio
            keep_ratio = keep_ratio - random_ratio
            num_keep = int(num_gaussians * keep_ratio)
            num_keep_random = int(num_gaussians * random_ratio)
            idx_sort = opacity.argsort(dim=1, descending=True)
            keep_idx = idx_sort[:, :num_keep]
            if num_keep_random > 0:
                rest_idx = idx_sort[:, num_keep:]
                random_idx = rest_idx[:, torch.randperm(rest_idx.shape[1])[:num_keep_random]]
                keep_idx = torch.cat([keep_idx, random_idx], dim=1)
            gaussians = gaussians.gather(1, keep_idx.expand(-1, -1, gaussians.shape[-1]))
        return gaussians

    def scale_act(self, x: torch.Tensor):
        cap = torch.tensor([self._scale_cap], device=x.device, dtype=x.dtype)
        return torch.minimum(torch.exp(x - self._scale_shift), cap)

    def _prepare_features(self, features: torch.Tensor):
        x_mask = None
        if self.opt.sub_sample_gaussians_factor is not None:
            if self.opt.sub_sample_gaussians_type == "learned":
                features, x_mask = features[:, :-1], features[:, [-1]]
        return features, x_mask

    def _subsample(
        self,
        features: torch.Tensor,
        rays_os: torch.Tensor,
        rays_ds: torch.Tensor,
        x_mask: Optional[torch.Tensor],
        training: bool,
    ):
        return self._subsample_fn(
            features,
            rays_os,
            rays_ds,
            x_mask,
            self.opt.sub_sample_gaussians_factor,
            self.opt.sub_sample_gaussians_type,
            self.opt.sub_sample_gaussians_type_tokens,
            self.opt.sub_sample_gaussians_temperature,
            training,
        )
