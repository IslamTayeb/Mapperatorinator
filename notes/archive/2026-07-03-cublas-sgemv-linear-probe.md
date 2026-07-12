# cuBLAS SGEMV Linear Probe

## Purpose

Test whether a stable vendor BLAS backend can improve the real one-token
decoder `Linear` calls enough to justify production kernel/backend work.

This was a diagnostic-only probe. It did not touch production inference and
does not claim an inference throughput win.

## Context

Current accepted single-song opt-in baseline:

- DCC job `49230082`
- full-song SALVALAI, `7,639` main tokens
- `28.243s` synchronized main-generation model time
- `270.475 tok/s`
- main/timing generated-token equivalence PASS
- byte-identical generated `.osu`

Prior per-linear native CUDA production work was exact but rejected: full-song
job `49230859` improved only `273.563 -> 281.743 tok/s` (`+2.990%`) and
production wiring was reverted. This probe checked whether cuBLAS SGEMV changes
that conclusion.

## Probe

Branch:

`experiment/cublas-sgemv-linear-probe`

Experiment commit:

`3f64e23` (`profile: add cublas sgemv linear probe`)

DCC job:

`49235621`

Run root:

`/work/imt11/Mapperatorinator/runs/cublas-sgemv-linear-49235621-3f64e23`

Logs:

- `/work/imt11/Mapperatorinator/logs/cublas-sgemv-linear-49235621.out`
- `/work/imt11/Mapperatorinator/logs/cublas-sgemv-linear-49235621.err`

Reports:

- `linear_prefix128.json`
- `linear_prefix640.json`

Environment:

- Node: `dcc-chsi-gpu-ferc-s-i11-1`
- GPU: RTX 2080 Ti, UUID `GPU-5b9229da-8433-e112-04c8-086f173136fb`
- Driver: `595.71.05`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- CUDA runtime reported by PyTorch: `12.8`
- Precision: `fp32`
- Attention: `sdpa`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`, seed `12345`

The temporary diagnostic added `--cublas-linear-variant` to
`utils/profile_decode_linear_kernels.py` and a separate
`native_vendor_linear.py` extension. The extension used `cublasSgemv` for
one-token fp32 `y = W x + bias`, interpreting PyTorch's row-major
`[out_dim, in_dim]` weight as a column-major `[in_dim, out_dim]` matrix and
calling `CUBLAS_OP_T`.

The code was not merged to `main`.

## Results

Both reports passed the existing verifier gates:

| prefix | pass | logits replay | max abs | captured linears |
| --- | --- | --- | ---: | ---: |
| `128` | PASS | PASS | `0.0` | `73` |
| `640` | PASS | PASS | `0.0` | `73` |

CUDA graph replay totals over the captured unique linears:

| prefix | functional `F.linear` | cuBLAS SGEMV | projected full-song saving |
| --- | ---: | ---: | ---: |
| `128` | `0.949982 ms/token` | `0.932900 ms/token` | `0.129s` |
| `640` | `0.949640 ms/token` | `0.932916 ms/token` | `0.126s` |

Representative per-signature graph replay timings:

| signature | calls/token | `F.linear` | cuBLAS SGEMV | best diagnostic variant |
| --- | ---: | ---: | ---: | ---: |
| `768 -> 768`, bias | `36` | `0.006839ms` | `0.006833ms` | `0.004195ms` native warp4 |
| `768 -> 2304`, bias | `12` | `0.015433ms` | `0.015405ms` | `0.013713ms` native warp4 |
| `768 -> 3072`, bias | `12` | `0.020517ms` | `0.020544ms` | `0.018025ms` native warp4 |
| `3072 -> 768`, bias | `12` | `0.020671ms` | `0.019263ms` | `0.018087ms` native warp4 |
| `768 -> 4069`, no bias | `1` | `0.024326ms` | `0.024348ms` | `0.023178ms` native warp2 |

The best diagnostic per-signature native variant projected only `1.29-1.34s`
saved in this run, still marginal and consistent with the already rejected
native-linear production attempt.

## Decision

Reject cuBLAS SGEMV as a per-linear production path.

It is exact, but the projected saving is only about `0.13s` over full-song
SALVALAI, far below the `5%` keep threshold and far below the `~2.57s` strong
`10%` target. Because SGEMV did not show a meaningful raw multiply win, do not
follow it with cuBLASLt bias-epilogue work for individual decoder linears unless
new profiling shows the old measurements are stale.

Future vendor/C++/CUDA work should target broader decoder-layer/runtime islands
or multiple operation classes together. One-linear-at-a-time backend swaps are
not target-sized for the current `270.475 tok/s` stack.
