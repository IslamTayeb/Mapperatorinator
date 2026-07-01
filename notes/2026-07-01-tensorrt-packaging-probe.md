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

## Isolated TensorRT-RTX Env Validation

Created an isolated venv at `/hpc/group/romerolab/imt11/envs/mapperatorinator-trt-rtx` with `--system-site-packages` from the retained Mapperatorinator env. The retained env was not modified.

- Bootstrap log: `/work/imt11/Mapperatorinator/logs/trt-rtx-env-bootstrap-20260701-014015.log`
- Env size after install: `4.6G`
- Pip cache size after install: `/work/imt11/Mapperatorinator/cache/pip-trt`, `2.7G`
- Installed runtime line: `torch==2.12.1+cu130`, `torch-tensorrt-rtx==2.12.1`, `tensorrt_rtx==1.4.0.76`, CUDA package family `13.0`
- Known dependency warning: inherited `torchaudio 2.10.0` from the system-site env requires `torch==2.10.0`; this is acceptable for isolated TensorRT import/compile smoke tests but another reason not to use the venv for normal inference.

Login-node import still fails because no NVIDIA driver is visible there:

```text
torch_tensorrt_import_error RuntimeError('Found no NVIDIA driver on your system...')
```

GPU import smoke passed:

| job | node | result |
| --- | --- | --- |
| `49138760` | `dcc-core-ferc-s-z25-21` | `torch_tensorrt 2.12.1` imported on RTX 2080 Ti, CUDA available, driver `595.71.05`, CUDA `13.2` reported by `nvidia-smi` |

The first toy `torch.compile(..., backend="tensorrt")` smoke was not a true TensorRT lowering success. Stderr reported:

```text
WARNING:torch_tensorrt.dynamo._compiler:3 supported operations detected in subgraph containing 3 computational nodes. Skipping this subgraph, since min_block_size was detected to be 5
```

So the successful outputs in job `49138760` were GraphModule/PyTorch fallback, not evidence of a TensorRT engine.

The stricter lowering smoke also failed to produce a TensorRT engine:

| job | node | cases | observed result |
| --- | --- | --- | --- |
| `49138769` | `dcc-core-ferc-s-z25-21` | larger FP32 MLP with default `min_block_size`; tiny FP32 MLP with `min_block_size=1` | both returned numerically identical outputs, but stderr showed TensorRT conversion failure and GraphModule fallback |

Key stderr from job `49138769`:

```text
Internal Error: MyelinCheckException: cudnn_graph_utils.h:405: CHECK(false) failed. cuDNN graph compilation failed.
ERROR: [Torch-TensorRT] - ICudaEngine::createExecutionContext: Error Code 1: Myelin
WARNING:torch_tensorrt.dynamo.backend.backends:TRT conversion failed on the subgraph. See trace above. Returning GraphModule forward instead.
RuntimeError: ... Unable to create TensorRT execution context
```

Interpretation: TensorRT-RTX import works on the RTX 2080 Ti node, but this package/runtime combination does not currently validate as a same-calculation FP32 acceleration path. It falls back before producing a TensorRT engine even on toy FP32 graphs. This matches the official TensorRT-RTX 1.5 warning that Turing lacks FP32 GEMM support in that release, though the installed RTX runtime resolved to 1.4.

Future TensorRT work should not proceed to Mapperatorinator one-token decoder export until a toy graph proves true TensorRT lowering. Use `pass_through_build_failures=True` or equivalent stderr/type checks so fallback fails loudly instead of being mistaken for an optimization.
