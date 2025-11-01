# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses for the Gaussian decoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass
class GaussianDecoderConfig:
    """Hyper-parameters used by :class:`GaussianDecoder`.

    The defaults mirror the most common settings used inside Lyra but can be tweaked for
    different reconstruction pipelines.  Only ``output_dims`` is strictly required when
    instantiating the decoder – everything else falls back to sensible defaults.
    """

    output_dims: int = 12
    """Number of channels predicted by the feature decoder before post-processing."""

    gaussian_scale_cap: float = 1.0
    """Maximum permitted scale for each Gaussian axis."""

    pre_sigmoid_distance_shift: float = 0.0
    """Offset applied before the sigmoid that converts depth logits into distances."""

    dnear: float = 0.0
    dfar: float = 1.0

    gaussians_predict_offset: bool = False
    use_gaussians_predict_offset: bool = False
    gaussians_predict_offset_act: str = "tanh"
    gaussians_predict_offset_range: Tuple[float, float] = (-0.05, 0.05)

    sub_sample_gaussians: bool = False
    sub_sample_gaussians_factor: Optional[Sequence[int]] = field(default_factory=lambda: (1, 1, 1))
    sub_sample_gaussians_type: str = "random"
    sub_sample_gaussians_type_tokens: str = "global"
    sub_sample_gaussians_temperature: float = 1.0

    gaussians_prune_ratio: float = 0.0
    gaussians_random_ratio: float = 0.0

    deferred_background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    renderer_kwargs: dict = field(default_factory=dict)

    keep_eval_mask: bool = False
    """Retain the learned mask tensor at evaluation time if True."""

    def effective_output_dims(self) -> int:
        dims = self.output_dims
        if self.gaussians_predict_offset:
            dims += 3
        return dims
