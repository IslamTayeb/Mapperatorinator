# FlashInfer SM75 Decode Feasibility

## Purpose

After active64 kernel attribution showed q_len=1 attention dominates the accepted opt-in path, this pass tested whether FlashInfer could be a narrow attention backend for RTX 2080 Ti without changing the calculation.

This was a backend feasibility probe only. It did not modify Mapperatorinator inference code and does not claim an inference speed result.

## Setup

- Main sweep job: `49212042`
- H=12 fp32 confirmation job: `49212270`
- Status: `COMPLETED`, exit `0:0`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
- Driver/CUDA from `nvidia-smi`: driver `595.71.05`, CUDA `13.2`
- PyTorch runtime: `2.10.0+cu128`, CUDA `12.8`
- Run dir: `/work/imt11/Mapperatorinator/runs/flashinfer-sm75-light-49212042`
- Report: `/work/imt11/Mapperatorinator/runs/flashinfer-sm75-light-49212042/report.json`
- H=12 report: `/work/imt11/Mapperatorinator/runs/flashinfer-fp32-h12-49212270/report.json`
- Isolated package target: `/work/imt11/Mapperatorinator/cache/flashinfer-target-0614-light-py310`
- Installed only into the isolated target, not the main Mapperatorinator env:
  - `flashinfer-python==0.6.14`
  - `apache-tvm-ffi==0.1.12`
  - `nvidia-ml-py==13.610.43`

Earlier package probes found two practical packaging hazards:

- `flashinfer-python==0.6.14` plus `flashinfer-cubin==0.6.13` fails FlashInfer's version check.
- A forced cubin-target reinstall left the isolated target inconsistent while mutating a large cubin tree, so the final decision used the clean light target above.

## Microbench Shape

The active64 no-graph trace showed the real q_len=1 SDPA decoder shape is `[1, 12, 1, 64]` for Q and `[1, 12, kv_len, 64]` for K/V. The first broad sweep used `H=6`, which was an initial shape assumption; because the failure happens at dtype dispatch before kernel timing, job `49212270` repeated the decisive fp32 call with `H=12`.

- heads: `12` in the confirmation job
- head dim: `64`
- q shape for FlashInfer: `[H, D]`
- K/V layout for FlashInfer: `HND`, `[H, kv_len, D]`
- SDPA reference: q `[1, H, 1, D]`, K/V `[1, H, kv_len, D]`
- KV lengths in the broad sweep: `64`, `96`, `128`, `256`, `512`, `640`, `1024`, `2560`
- Confirmation dtype/shape: `torch.float32`, `H=12`, `kv_len=64`

## Results

FlashInfer imported successfully in the isolated target:

```text
flashinfer_version: 0.6.14
gpu: NVIDIA GeForce RTX 2080 Ti
capability: [7, 5]
```

All `torch.float32` `single_decode_with_kv_cache` calls failed before timing. The same error reproduced with the actual model head count in job `49212270`:

```text
KeyError(torch.float32)
...
flashinfer/jit/attention/modules.py
filename_safe_dtype_map[dtype_q]
```

All broad-sweep `torch.float16` calls failed because the light target did not include `ninja` for JIT build:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

The fp16 path is not a same-calculation candidate for the current fp32 inference objective anyway. Installing `ninja` or a matching cubin/JIT-cache stack would only answer a non-equivalent or future mixed-precision question unless FlashInfer also supports the fp32 decode path.

## Integration Risk If Revisited

The minimal exact integration point would be inside VarWhisper attention after Q/K/V projection, RoPE, and `StaticCache.update`, but before `Wo`. That means a default-off branch in the current attention path, not a new decode loop, and only for decoder self-attention with `B=1`, `q_len=1`, static cache, active-prefix decode, eval/no dropout, no attentions, no beams, no CFG, and no server batching.

The hard correctness issue is active-prefix bucket masking. The accepted active64 graph path uses a bucketed prefix length while SDPA receives a sliced 4D static-cache mask that hides future unused slots in the bucket. A FlashInfer replacement must either preserve the same static bucket shape while respecting the true valid cache length/mask, or it will attend to future zero/uninitialized cache slots and fail the one-token logits gate. Using exact per-token valid lengths avoids future slots, but changes K/V shape every token and likely destroys the current manual CUDA graph reuse.

Other risks:

- Static cache is laid out like `[B, H, max_len, D]`; any transpose/contiguous copy into FlashInfer layout must be included in benchmarks.
- VarWhisper already applies RoPE before attention; FlashInfer positional modes must stay disabled unless proven exact.
- Cross-attention, prefill, encoder attention, server batching, and non-active paths should remain out of scope until the self-attention gate passes.

## Decision

Reject FlashInfer as a current same-calculation fp32 backend for Mapperatorinator on RTX 2080 Ti. The isolated SM75 probe shows FlashInfer's single-request decode API does not accept `torch.float32` here, so it cannot replace SDPA in the current fp32 retained/active-prefix paths.

Do not integrate FlashInfer or retry it as a quick optimization. Revisit only if a future FlashInfer release or isolated environment proves fp32 `single_decode_with_kv_cache` works on SM75, then pass these gates before any throughput claim:

1. Isolated fp32 q_len=1 SDPA comparison with model-like shapes and no layout-copy omission.
2. Active-prefix bucket valid-length/mask equivalence story.
3. `utils/verify_one_token_decode.py --sequence-index 9`.
4. `utils/verify_direct_decode_loop.py` token/logit/RNG gate.
5. 15s SALVALAI generated-token equivalence.
6. Full-song SALVALAI generated-token equivalence and non-regression.
