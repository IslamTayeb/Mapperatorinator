# Native MLP Residual Probe

## Question

Test whether a narrow native CUDA replacement for one decoder MLP residual block
is large enough to justify production integration:

`final_layer_norm -> fc1 -> GELU -> fc2 -> residual add`

The target threshold from the post-270 MLP breakdown was about `1.4s` projected
full-song saving, or roughly the current `5%` keep bar. This was a diagnostic
probe only; it did not run end-to-end inference and did not make a throughput
claim.

## Implementation

Branch: `experiment/native-mlp-residual-probe`

Temporary code added:

- `osuT5/osuT5/inference/native_mlp.py`
- `utils/profile_decode_decoder_layer_island.py --candidate-native-mlp-residual`

The native helper used three CUDA kernels:

- RMSNorm over the one-token hidden vector
- `fc1 + GELU` into an intermediate buffer
- `fc2 + residual add`

Variants used `2`, `4`, and `8` outputs per block. The probe was strict fp32,
batch-1, q_len=1, eval/no-dropout only. The code was removed after measurement.

Setup mistakes before the final run:

- Job `49232948` failed before profiling because the sbatch flags set
  `inference_q1_bmm_cross_attention=true` without
  `inference_active_prefix_decode_loop=true`.
- Job `49232956` proved the baseline MLP numbers but rejected the native setup
  because Transformers exposes exact GELU as `GELUActivation`.
- Job `49232984` compiled the native extension, then rejected setup because
  PyTorch RMSNorm used `eps=None`, meaning dtype epsilon.

## Result

Final valid DCC job: `49233007`

- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
  `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1`
- Commit: `14a077c`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/native-mlp-residual-probe-20260703-143509-14a077c`
- Reports:
  - `native_mlp_seq9_warm50_iter500.json`
  - `native_mlp_seq9_prefix128_warm100_iter2000.json`
  - `summary.json`

Both repeats passed the decoder logits replay gate:

- `pass=true`
- `active_prefix_length=128`
- `logits_replay_max_abs=0.0`
- native MLP allclose passed for all warp variants
- native MLP max abs: `7.62939453125e-06`

Projection over `12` decoder layers and `7,552` full-song decode steps:

| repeat | MLP residual | best native variant | projected saving | projected tok/s |
| --- | ---: | ---: | ---: | ---: |
| warm50 / iter500 | `4.2996s` | `3.4777s` | `0.8219s` | `278.58` |
| prefix128 warm100 / iter2000 | `4.2528s` | `3.4769s` | `0.7759s` | `278.11` |

## Decision

Rejected and removed.

The native MLP residual path is exact enough for an isolated diagnostic, but it
saves only about `0.76-0.82s` projected full-song model time, roughly
`270.475 -> 278 tok/s`. That is below the `5%` keep threshold and well below
the `~1.4s` MLP-only threshold.

This result also confirms the broader pattern: one MLP-only native island is too
small. Future kernel work should target broader decoder-layer/runtime islands or
multiple compute classes together, not a standalone MLP residual replacement.
