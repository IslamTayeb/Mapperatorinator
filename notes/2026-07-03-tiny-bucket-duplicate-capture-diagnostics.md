# Tiny-Bucket Duplicate-Capture Diagnostics

## Purpose

After active64 became the fastest full-song opt-in bucket, the remaining question was whether smaller buckets were only losing because they repeated more CUDA graph captures. If so, a persistent graph/cache runtime might make tiny buckets attractive.

This was a diagnostic-only Slurm run. It does not replace full-song validation.

## Run

- Job: `49209984`
- Status: `COMPLETED`, exit `0:0`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
- Driver: `595.71.05`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Commit: `24342d0`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-tiny-bucket-diag-49209984-24342d0`
- Config: `profile_salvalai_smoke15`

All runs used:

```bash
inference_generation_compile=true
inference_active_prefix_decode_loop=true
inference_active_prefix_decode_cuda_graph=true
inference_active_prefix_decode_cuda_graph_warmup=0
inference_active_prefix_decode_cuda_graph_min_decode_steps=1
inference_stateful_monotonic_logits_processor=true
profile_active_prefix_decode_diagnostics=true
profile_record_token_ids=true
```

Each bucket used isolated TorchInductor, Triton, and CUDA cache dirs.

## Results

All candidate buckets matched bucket64 main-generation token IDs on the 15s smoke (`1,084 / 1,084`).

| bucket | main tok/s | projected tok/s without duplicate capture | duplicate capture | graph captures | normalized graph shapes | decode forward wall | prepare inputs wall | logits processor wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `16` | `153.033` | `168.054` | `0.633s` | `75` | `37` | `1.498s` | `0.515s` | `0.533s` |
| `32` | `163.862` | `173.221` | `0.357s` | `40` | `19` | `0.926s` | `0.516s` | `0.528s` |
| `48` | `166.950` | `174.719` | `0.289s` | `30` | `13` | `0.758s` | `0.512s` | `0.529s` |
| `64` | `169.290` | `175.853` | `0.239s` | `24` | `10` | `0.650s` | `0.521s` | `0.535s` |
| `96` | `170.973` | `176.136` | `0.186s` | `18` | `7` | `0.542s` | `0.512s` | `0.529s` |
| `128` | `170.026` | `174.726` | `0.172s` | `16` | `6` | `0.512s` | `0.517s` | `0.535s` |
| `192` | `170.404` | `174.187` | `0.138s` | `12` | `4` | `0.444s` | `0.524s` | `0.535s` |

Bucket96 was exact and fastest on smoke, but only by `+1.0%` over bucket64 (`169.290 -> 170.973 tok/s`) and failed strict per-window no-regression on `4 / 10` main windows. This is below the promotion threshold.

## Interpretation

The duplicate-capture hypothesis is not enough. Smaller buckets increase graph churn faster than they reduce steady decode cost, and even the best projected no-duplicate-capture throughput is only about `176 tok/s` on this 15s slice. This is consistent with the full-song active64 duplicate-capture ceiling from job `49207288`, where removing duplicate captures projected only about `166.9 tok/s`.

Persistent/bufferized graph/cache reuse may still be useful for 5-10% runtime cleanup or future batch-serving, but it should not be treated as the main path to `200 tok/s`. A verifier-only persistent runtime is still the right first step if this path is pursued, because graph replay is pointer-bound to cache and encoder-output storage.

## Decision

Do not promote bucket96 from this diagnostic. Keep bucket64 as the current full-song opt-in bucket and bucket192 as the safer timing-stability fallback.

Do not start production persistent graph/cache reuse as a quick optimization. If pursued, implement it first as a verifier-only multi-window runtime that proves token identity, raw-logit/top-k equality, final RNG-state equality, and reduced graph captures before any 15s or full-song speed claim.
