# Active-Prefix Cudagraph Trees

## Hypothesis

Active-prefix cold overhead is dominated by TorchInductor/CUDAGraph tree churn rather than raw attention kernels. If so, disabling `torch._inductor.config.triton.cudagraph_trees` might reduce the first-window tax while preserving exact tokens.

This was a diagnostic/runtime experiment, not a model-calculation change.

## Evidence Before The Experiment

Existing diagnostics already showed active-prefix is strong after warmup:

- Cold active512 full song: `96.247 tok/s`, `seq0=25.531s`, post-`seq0` about `131 tok/s`.
- Warm active512 full song run 1: `127.877 tok/s`, `seq0=3.809s`.
- Warm active512 full song run 2: `126.610 tok/s`, `seq0=3.871s`.

`/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-49156749-daa1828/diagnostic_summary.json` reported:

```text
recompiles: 244
cuda_graph_launch_or_record: 8665
recording_function: 3441
perf_hints: 8
```

Representative log lines from `/work/imt11/Mapperatorinator/logs/ap-diagnostics-49156749.err` included:

```text
Recompiling function sdpa_attention_forward ... tensor 'key' size mismatch at index 2. expected 2560, actual 1024
Recompiling function update .../transformers/cache_utils.py:742
layer_idx == 0
...
torch._dynamo hit config.recompile_limit (8)
function: 'update' (.../transformers/cache_utils.py:742)
last reason: layer_idx == 7
```

## Global No-Cudagraph-Trees Probe

DCC jobs:

- `49162564`: compile-only and active512 default profiles completed, but the first no-cudagraph wrapper failed because Hydra could not resolve configs when run from stdin.
- `49162714`: corrected no-cudagraph run using `runpy.run_path("inference.py", run_name="__main__")`.

Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
Torch: `2.10.0+cu128`
Transformers: `4.57.3`
Commit: `ae5968a`
Run dirs:

- `/work/imt11/Mapperatorinator/runs/ap-cudagraph-tree-probe-49162564-ae5968a`
- `/work/imt11/Mapperatorinator/runs/ap-no-cgt-only-49162714-ae5968a`

| run | main tokens | main model time | main tok/s | timing model time | token equivalence |
| --- | ---: | ---: | ---: | ---: | --- |
| compile-only | `1,084` | `21.855s` | `49.600` | `34.243s` | baseline |
| active512 default | `1,084` | `29.820s` | `36.352` | `39.420s` | PASS |
| active512 global no-cudagraph-trees | `1,084` | `16.755s` | `64.696` | `73.607s` | PASS |

Interpretation: disabling cudagraph trees globally made active512 main generation exact and much faster on the 15s smoke (`+30.4%` vs compile-only, `+78.0%` vs active512 default), but timing generation regressed badly (`34.243s -> 73.607s`). This cannot be promoted because it fails the non-regression rule outside main generation.

## MAP-Only Scoped Patch

Hypothesis: keep timing on the retained compile path, but apply active512 plus no-cudagraph-trees only when `context_type=MAP`.

Dirty patch on DCC added default-off flags:

- `inference_active_prefix_decode_contexts=[MAP]`
- `inference_active_prefix_disable_cudagraph_trees=true`

DCC job: `49162877`
Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
Run dir: `/work/imt11/Mapperatorinator/runs/ap-map-nocgt-smoke15-49162877-ae5968a`

The direct loop gate passed first:

- compile enabled
- active-prefix decode length `512`
- cudagraph trees disabled
- generated-token match PASS
- raw-logit `max_abs=0.0`
- RNG-state match PASS
- wall `80.918s`

15s smoke result:

| run | main tokens | main model time | main tok/s | timing model time | token equivalence |
| --- | ---: | ---: | ---: | ---: | --- |
| compile-only | `1,084` | `22.558s` | `48.053` | `34.218s` | baseline |
| active512 MAP-only no-cudagraph-trees | `1,084` | `63.546s` | `17.059` | `37.651s` | PASS |

The candidate was exact but regressed main generation by `-64.5%`. The worst map window was `seq3=54.423s` for `499` tokens, compared with compile-only `seq3=16.166s`.

## Interpretation

Global no-cudagraph-trees is an important diagnostic signal: cudagraph tree capture/state can dominate active-prefix cold behavior. But the attempted per-call scoping made graph state worse, likely because timing compiled/captured one set of graph state and then the MAP call tried to compile a different no-cudagraph active-prefix path in the same process.

The useful next direction is not another naive `cudagraph_trees` flag. It is deeper graph-state isolation:

- bucket-scoped compiled active-prefix decode calls;
- a direct decode runtime that owns the q_len=1 model-call boundary;
- only then broader cache monomorphization if `StaticCache.update(layer_idx)` remains target-sized.

## Decision

Reject and revert the scoped code patch. Do not promote global or MAP-only cudagraph-tree disabling. Keep the diagnostic result as evidence that active-prefix cold overhead is graph-state/runtime churn, not a small Python append/copy issue.
