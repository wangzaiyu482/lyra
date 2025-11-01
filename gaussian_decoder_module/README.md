# Gaussian Decoder Module

This directory contains a self-contained PyTorch implementation of the Gaussian decoder
that powers Lyra's latent reconstruction pipeline.  It is packaged as a small, reusable
module that can be dropped into other projects without depending on the rest of the Lyra
codebase.

The module exposes two main entry points:

- `GaussianDecoderConfig` – a dataclass that stores the hyper-parameters that control
  subsampling, Gaussian post-processing and pruning behaviour.
- `GaussianDecoder` – a `torch.nn.Module` that maps latent features and corresponding
  ray bundles to Gaussian primitives and optionally renders them with a user supplied
  renderer implementation.

See `examples/minimal_usage.py` for a short example that demonstrates the minimal inputs
required to perform a forward pass.
