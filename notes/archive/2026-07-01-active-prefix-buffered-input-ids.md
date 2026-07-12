# Active-Prefix Buffered Input IDs

## Hypothesis

The active-prefix custom decode loop appends `input_ids` with `torch.cat` every generated token. Preallocating the output-id buffer could reduce per-token allocation/copy overhead while preserving HF generation semantics.

This was tested only as a default-off candidate. It was not intended to change the retained compile-only baseline.

## Evidence

DCC job: `49162138`
Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
Driver: `595.71.05`
Torch: `2.10.0+cu128`
Transformers: `4.57.3`
Commit: dirty test patch on `666db84`
Run dir: `/work/imt11/Mapperatorinator/runs/ap-buffer-smoke15-49162138-666db84`

All runs used `configs/inference/profile_salvalai_smoke15.yaml`, `seed=12345`, `use_server=false`, `attn_implementation=sdpa`, `precision=fp32`, `inference_generation_compile=true`, and isolated TorchInductor/CUDA cache dirs.

| run | main tokens | main model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | --- |
| compile-only | `1,084` | `21.460s` | `50.513` | baseline |
| active512 old path | `1,084` | `29.006s` | `37.372` | PASS vs compile-only |
| active512 buffered input IDs | `1,084` | `29.802s` | `36.373` | PASS vs old path and compile-only |

Comparisons:

- Active512 old vs compile-only: `-26.0%` main tok/s.
- Active512 buffered vs active512 old: `-2.7%` main tok/s.
- Active512 buffered vs compile-only: `-28.0%` main tok/s.

The buffered candidate changed active512 `seq3` from `24.690s` to `25.455s`, and `seq9` from `1.555s` to `1.566s`. Timing generation improved in this run (`4.4 -> 7.0 tok/s`), but the main-generation objective regressed.

## Interpretation

Per-token `torch.cat` on `input_ids` is not the cold active-prefix bottleneck. Removing it did not reduce the first long map-window cost and slightly worsened both the first long window and a warmed later window. The active-prefix issue remains graph/specialization/runtime churn around the model call, not this small output-id append.

## Decision

Reject and revert. Do not retry active-prefix buffered input-id preallocation unless a new trace shows `input_ids` append/copy dominates after graph/runtime stabilization.
