# Known-working environment

This project did not previously ship a `pyproject.toml`; the new file pins the
MagicBrush training dependencies observed in the working environment. Use Python
3.10 and start with `uv sync --extra dev`. `flash-attn` is optional: it was not
installed in the known-working run, and the model falls back to PyTorch SDPA.

## Verified runtime

| Component | Version |
| --- | --- |
| Python | 3.10.21 |
| PyTorch | 2.3.1+cu121 |
| CUDA runtime | 12.1 |
| NVIDIA driver | 535.230.02 |
| transformers | 4.46.2 |
| diffusers | 0.34.0 |
| accelerate | 1.14.0 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| flash-attn | not installed |
| scikit-learn | not installed |

Known-working hardware was 4 x NVIDIA A800 80GB. On the target H100 cluster,
keep BF16 for the first migration run; do not enable FP8 automatically.

## PyTorch CUDA wheels

`uv sync` resolves the Python dependency set. Install the CUDA-compatible
PyTorch wheel according to the target cluster policy before running training.
The source run used the PyTorch CUDA 12.1 wheel family. Verify `torch.cuda.is_available()`
and `torch.version.cuda` after installation. Do not install flash-attn until the
basic smoke test passes.
