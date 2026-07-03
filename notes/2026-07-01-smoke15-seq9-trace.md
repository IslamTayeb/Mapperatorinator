# 15s Post-Warmup Torch Trace

## Run

- Job: `49133341`
- Node: `dcc-core-ferc-s-z25-21`
- GPU: RTX 2080 Ti
- Commit: `ce92ebb`
- Profile: `/work/imt11/Mapperatorinator/runs/smoke15-trace-seq9-49133341-ce92ebb/beatmap574d4d0acdf84cef8c7efd2758fb74bb.osu.profile.json`
- Trace: `/work/imt11/Mapperatorinator/runs/smoke15-trace-seq9-49133341-ce92ebb/torch_profiles/000_generation_main_generation_seq9.trace.json`

The job used `profile_torch_generation_label_filter=main_generation.seq9` and `profile_torch_generation_limit=1` to trace one post-warmup main-generation window from the retained SDPA + generation-compile baseline.

## Timing

The traced record generated `234` main tokens.

- Normal synchronized model time for the record: `2.562s`, `91.3 tok/s`.
- Outer wall for the traced record: `24.207s`.
- Torch profile range wall: `17.860s`.

This is diagnostic only. The profiler/export overhead is large, so throughput claims should continue to use normal `profile_inference` records.

## Event Mix

Top CUDA self-time events from the JSON summary:

| event | count | self CUDA |
| --- | ---: | ---: |
| `Torch-Compiled Region: 0/2` | 233 | 1.717s |
| `fmha_cutlassF_f32_aligned_64x64_rf_sm75` | 5,628 | 1.476s |
| GEMV kernel variant 7 | 11,184 | 104.156ms |
| GEMV2T kernel | 3,029 | 60.372ms |
| GEMV kernel variant 6 | 2,796 | 42.022ms |
| `Memcpy DtoD` | 10,152 | 13.384ms |
| `aten::sort` | 234 | 6.647ms self CUDA, 7.601ms CUDA total |
| `aten::cat` | 542 | 1.433ms |
| `aten::_softmax` | 468 | 1.759ms |

CPU-side visible overhead was also small relative to the compiled forward:

- `aten::nonzero`: `702` calls, `48.997ms` CPU total.
- `aten::index`: `468` calls, `47.733ms` CPU total.
- `aten::sort`: `234` calls, `22.546ms` CPU total.
- `aten::cat`: `542` calls, `10.697ms` CPU total.

## Interpretation

The post-warmup trace does not support making fused sampling/logits processors the next primary bet. Sorting, softmax, cat, and related Python-visible sampling work are real but far below the `10%` threshold for a 200 tok/s path in this record.

The dominant cost is still the compiled one-token forward, especially f32 memory-efficient attention on SM75 plus many small GEMV/GEMM launches. The next serious project should therefore bias toward exact custom decode/CUDA-graph/TensorRT-style work that can reduce compiled forward launch and attention/model-kernel cost, not just sampling glue.
