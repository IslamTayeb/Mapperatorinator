# Native Linear+GELU Probe

## Hypothesis

The current decoder MLP still costs several seconds over full-song one-token
decode. A narrow native `fc1 + GELU` kernel could remove the separate activation
kernel and reduce one launch inside each decoder-layer MLP without changing the
rest of the calculation.

This was a diagnostic microprobe only. It was not wired into production
inference and does not claim an end-to-end speedup.

## Probe

Branch: `experiment/native-linear-gelu-probe`

Temporary commit `0a89cf7` added a native
`one_token_linear_gelu_warp_group` kernel to `native_linear.py` and exposed
`native_fc1_gelu_warp{2,4,8}_mlp` variants through
`utils/profile_decode_linear_kernels.py --native-linear-variant`.

The branch then reverted that code after measurement because the result was
below the keep threshold.

## Result

- Job: `49232032`
- Commit: `0a89cf7`
- State: `COMPLETED`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
  `GPU-8de8e67b-289d-a1ae-3745-25f40db81f55`
- Report:
  `/work/imt11/Mapperatorinator/runs/native-gelu-probe-49232032-0a89cf7/native_gelu_probe.json`
- Config: `profile_salvalai_smoke15`, seq9, active-prefix length `640`
- Correctness: logits replay PASS (`max_abs=0.0`), MLP variants allclose
  (`max_abs=1.9073486328125e-06`)

MLP CUDA graph replay timings:

| variant | ms/layer | graph replay speedup |
| --- | ---: | ---: |
| `functional_mlp` | `0.04136271953582764` | `1.000x` |
| `mv_mlp` | `0.04132431983947754` | `1.001x` |
| `native_fc1_gelu_warp2_mlp` | `0.03718496084213257` | `1.112x` |
| `native_fc1_gelu_warp4_mlp` | `0.037229599952697756` | `1.111x` |
| `native_fc1_gelu_warp8_mlp` | `0.037251520156860354` | `1.110x` |

The best isolated variant saves `0.00417775869369507ms` per layer/token. Over
`12` decoder layers and `7,552` full-song one-token decode steps, that projects
to about `0.379s` saved. Against the accepted `28.243s`, `270.475 tok/s`
baseline, the idealized projection is about `274.149 tok/s`, or `+1.36%`.

## Decision

Reject and remove the native `linear+GELU` diagnostic kernel. It is exact enough
for the microprobe and faster in isolation, but the projected full-song gain is
far below the `5%` keep threshold and far below the normal `>=10%` graduation
threshold.

Do not promote narrow `fc1+activation` fusion for the current path unless new
profiling shows the MLP activation launch has become much larger. MLP work is
still plausible only as part of a broader fused MLP/decoder-layer runtime that
saves at least `~1.41s` full-song model time.
