# Weighted Manual Decoder-Layer Confirmation

## Purpose

Re-check the prefix640 manual decoder-runtime island signal against the actual
full-song active-prefix bucket distribution before doing any production
runtime/kernel work.

This is a bottleneck proof, not a throughput claim. The question was whether the
exact manual decoder-runtime island clears the current `5%` full-song model-time
bar when weighted across all SALVALAI decode buckets.

## Artifacts

- Branch: `codex/weighted-manual-layer-confirmation`
- Utility commit: `fc66274`
- DCC job: `49262108`
- GPU node: `dcc-core-ferc-s-z25-20`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/weighted-layer-manual-49262108-fc66274`
- Summary:
  `/work/imt11/Mapperatorinator/runs/weighted-layer-manual-49262108-fc66274/weighted_decoder_layer_manual_summary.json`
- Source bucket weights:
  `/work/imt11/Mapperatorinator/runs/decoder-stack-weighted-49233687/weighted_decoder_stack_summary.json`

## Method

Added `utils/summarize_weighted_decoder_layer_island.py`.

The utility combines per-prefix `utils/profile_decode_decoder_layer_island.py`
reports with the weighted full-song replay counts from job `49233687`. This run
used prefixes `128..768`, required `repo_decoder_layer` and
`manual_decoder_runtime_island`, and required candidate cache-write checks.

Each per-prefix report used:

```bash
utils/profile_decode_decoder_layer_island.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --active-prefix-bucket-size 64 \
  --native-q1-rope-cache-self-attention \
  --candidate-decoder-runtime-island \
  --cuda-graph-replay \
  --include-cache-write-fingerprint \
  --verify-cache-write-candidates
```

## Result

The weighted summary passed:

- report pass: `True`
- failures: `[]`
- weighted decode replays: `7,552`
- accepted baseline used for projection: `7,639` tokens, `28.243s`,
  `270.474 tok/s`
- `5%` model-time keep bar: `1.412150s`
- `10%` model-time strong bar: `2.824300s`

Weighted CUDA graph replay:

| variant | weighted seconds | saved vs repo | projected tok/s | decision |
| --- | ---: | ---: | ---: | --- |
| `repo_decoder_layer` | `16.512013s` | `0.000000s` | `270.474` | baseline |
| `manual_decoder_runtime_island` | `15.895097s` | `0.616916s` | `276.514` | below keep bar |

The manual island is exact under the verifier but not target-sized. It saves
only `0.617s`, well below the `1.412s` `5%` bar. Do not production-wire manual
module-boundary recomposition.

The component pressure remains large:

| component variant | weighted seconds | ideal saved vs repo | ideal projected tok/s |
| --- | ---: | ---: | ---: |
| `self_attn_residual_segment` | `7.053539s` | `9.458474s` | `406.664` |
| `cross_attn_residual_segment` | `3.328954s` | `13.183059s` | `507.240` |
| `mlp_residual_segment` | `4.210567s` | `12.301446s` | `479.188` |

These component rows are not production savings. They show where the necessary
model work lives if a future native/runtime candidate can reduce real math or
memory movement.

## Interpretation

The prefix640 manual island signal from job `49258725` did not survive weighted
confirmation. It was a useful warning that a broad decoder-layer boundary can
move, but it is not a reliable production target by itself.

Current bottleneck conclusion:

- The theoretical minimum is still far enough to justify deeper work: weighted
  decoder-layer replay is `16.512s`, about `58.5%` of the accepted `28.243s`
  model time.
- The manual Python/module recomposition path is not the lever: `0.617s` saved
  is below threshold.
- Future work should target real multi-segment native math/memory behavior,
  cache/layout, or a broader DecodeSession-owned decoder-layer island.
- Do not start another narrow kernel or production flag unless a weighted,
  exact verifier projects at least `~1.4s` full-song model-time saving,
  preferably `>=2.8s`.

## Decision

Keep `utils/summarize_weighted_decoder_layer_island.py` as verifier/profiling
infrastructure.

Reject production work for `manual_decoder_runtime_island`. It is exact but too
small.
