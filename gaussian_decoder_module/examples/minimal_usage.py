"""Minimal usage demo for the standalone Gaussian decoder."""

import torch

from gaussian_decoder.config import GaussianDecoderConfig
from gaussian_decoder.decoder import GaussianDecoder


class DummyRenderer:
    """Toy renderer that just returns the Gaussian tensor for inspection."""

    def render(self, gaussians: torch.Tensor, *args, **kwargs):
        return {"gaussians": gaussians, "kwargs": kwargs}


def main():
    config = GaussianDecoderConfig(
        output_dims=12,
        dnear=0.1,
        dfar=4.0,
        gaussian_scale_cap=0.5,
        sub_sample_gaussians=True,
        sub_sample_gaussians_factor=(2, 2, 2),
        sub_sample_gaussians_type="random",
    )
    decoder = GaussianDecoder(config, renderer=DummyRenderer())

    batch, channels, time, height, width = 1, config.effective_output_dims(), 4, 8, 8
    features = torch.randn(batch, channels, time, height, width)
    rays_os = torch.randn(batch, time, 3, height, width)
    rays_ds = torch.randn(batch, time, 3, height, width)

    gaussians, mask = decoder(features, rays_os, rays_ds, training=False)
    rendered = decoder.render(gaussians, camera_pose=torch.eye(4))

    print("gaussians shape:", gaussians.shape)
    print("mask:", None if mask is None else mask.shape)
    print("render keys:", rendered.keys())


if __name__ == "__main__":
    main()
