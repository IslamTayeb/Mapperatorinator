# TensorRT Packaging Probe

## Result

The current DCC Mapperatorinator environment should stay untouched for TensorRT work. Package resolution shows two plausible but disruptive install paths, both requiring an isolated env before any runtime experiment.

Evidence paths:

- Standard Torch-TensorRT dry-run log: `/work/imt11/Mapperatorinator/logs/torch_tensorrt_210_unpinned_dryrun.log`
- Standard Torch-TensorRT dry-run report: `/work/imt11/Mapperatorinator/logs/torch_tensorrt_210_unpinned_dryrun_report.json`
- TensorRT-RTX dry-run log: `/work/imt11/Mapperatorinator/logs/torch_tensorrt_rtx_dryrun.log`
- TensorRT-RTX dry-run report: `/work/imt11/Mapperatorinator/logs/torch_tensorrt_rtx_dryrun_report.json`
- Driver/GPU probe job: `49138561`, node `dcc-core-ferc-s-z25-21`, driver `595.71.05`, reported CUDA `13.2`, GPU `NVIDIA GeForce RTX 2080 Ti`, capability `(7, 5)`

## Standard Torch-TensorRT

The current env has PyTorch `2.10.0+cu128`. `torch-tensorrt==2.10.0` resolves against that PyTorch line, but it wants the CUDA 13 TensorRT package family:

| package | resolved version |
| --- | --- |
| `torch_tensorrt` | `2.10.0` |
| `tensorrt` | `10.14.1.48.post1` |
| `tensorrt_cu13` | `10.14.1.48.post1` |
| `tensorrt_cu13_bindings` | `10.14.1.48.post1` |
| `tensorrt_cu13_libs` | `10.14.1.48.post1` |
| `cuda-toolkit` | `13.3.1` |
| `nvidia-cuda-runtime` | `13.3.29` |
| `nvidia-cuda-runtime-cu13` | `0.0.0a0` |
| `dllist` | `2.0.0` |

Risk: the DCC GPU node reported driver CUDA `13.2`, while this resolver path wants CUDA toolkit/runtime `13.3`. Do not assume it imports or runs without a GPU-node validation job.

## Torch-TensorRT-RTX

`torch-tensorrt-rtx` resolves, but it is not a drop-in addition to the current env. The unpinned dry-run would install `torch-tensorrt-rtx==2.12.1`, replace the PyTorch line with `torch==2.12.1`, and pull a 44-package stack including `executorch`, CUDA 13.0 packages, and TensorRT-RTX 1.4:

| package | resolved version |
| --- | --- |
| `torch-tensorrt-rtx` | `2.12.1` |
| `torch` | `2.12.1` |
| `tensorrt_rtx` | `1.4.0.76` |
| `tensorrt_rtx_cu13` | `1.4.0.76` |
| `tensorrt_rtx_cu13_bindings` | `1.4.0.76` |
| `tensorrt_rtx_cu13_libs` | `1.4.0.76` |
| `cuda-toolkit` | `13.0.2` |
| `nvidia-cuda-runtime` | `13.0.96` |
| `triton` | `3.7.1` |
| `executorch` | `1.3.1` |
| `torchao` | `0.17.0` |

The package index also lists `tensorrt-rtx==1.5.0.114`, but the resolved `torch-tensorrt-rtx==2.12.1` dependency selected `<1.5.0`, so a TensorRT-RTX 1.5 test would need an explicit compatibility investigation.

Official docs matter here:

- PyTorch documents Torch-TensorRT-RTX as experimental, with `torch-tensorrt-rtx` installed separately and the Python import still named `torch_tensorrt`: https://docs.pytorch.org/TensorRT/getting_started/tensorrt_rtx.html
- NVIDIA documents PyTorch usage via `torch.compile(..., backend="tensorrt")`: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/torch-trt-rtx.html
- NVIDIA's TensorRT-RTX 1.5 support matrix lists Turing/RTX 2080Ti and Linux x86-64 compute capability `7.5` support, but footnote `[1]` says TensorRT-RTX on Turing does not support FP32 GEMMs in this release: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix-1/1.5.html

## Interpretation

TensorRT is still a plausible 200 tok/s research path, but this probe makes it lower-confidence for same-calculation FP32 on RTX 2080 Ti:

- Standard Torch-TensorRT is aligned with the current PyTorch major/minor version, but its resolved CUDA 13.3 runtime stack may not match the current DCC driver report.
- Torch-TensorRT-RTX is better aligned with RTX/Turing as a product target and exposes CUDA graph/runtime-cache knobs, but it changes the PyTorch line and official TensorRT-RTX docs warn about FP32 GEMM support on Turing.
- Because the retained baseline is same-calculation FP32-style inference, TensorRT-RTX should not be assumed to accelerate the dominant decoder GEMMs unless a GPU-node compile/import/logits test proves it.

## Next Gate

If continuing this path, use a separate persistent env, for example `/hpc/group/romerolab/imt11/envs/mapperatorinator-trt`, with caches under `/work/imt11/Mapperatorinator/cache`. The first Slurm GPU job should do only:

1. Print driver, GPU, `torch`, CUDA, `torch_tensorrt`, and TensorRT package versions.
2. Import `torch_tensorrt`.
3. Compile a tiny fixed-shape CUDA module with `torch.compile(..., backend="tensorrt")`.
4. Only after that succeeds, try the repeated one-token decoder forward.
5. Compare logits before any end-to-end generation run.

Do not promote this to an inference optimization unless the 15s smoke generated-token IDs match, then a full-song run matches and beats the retained SDPA + generation-compile baseline by at least `10%`.
