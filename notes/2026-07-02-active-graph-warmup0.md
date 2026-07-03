# Active Graph Zero-Warmup Promotion

Superseded update: this note promoted `warmup=0` for the active-prefix CUDA graph path and was current at `146.602 tok/s`. Job `49206207` later kept `warmup=0` but reduced `inference_active_prefix_decode_bucket_size` to `64`, reaching `155.578 tok/s` full-song main generation with exact token identity. See `notes/2026-07-02-active-bucket-size-sweep.md`.

## Hypothesis

The active-prefix CUDA graph path no longer benefits from replay warmups before capture. The three warmup forwards were intended to stabilize graph capture, but after the stateful monotonic processor and active-prefix graph path stabilized the repeated one-token decode shape, those extra forwards may be pure measured overhead. Because active-prefix graph remains default-off, a simpler `warmup=0` default is worth keeping if it is exact and improves full-song main, timing, and total stage time.

## Implementation

Set the active-prefix CUDA graph warmup default to zero:

```yaml
inference_active_prefix_decode_cuda_graph_warmup: 0
```

The active graph path remains opt-in. At the time of this warmup promotion, the fastest accepted variant used:

```bash
inference_active_prefix_decode_loop=true
inference_active_prefix_decode_bucket_size=512
inference_active_prefix_decode_cuda_graph=true
inference_stateful_monotonic_logits_processor=true
use_server=false
parallel=false
cfg_scale=1.0
num_beams=1
```

## 15s Smoke

- Job: `49204447`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
- Commit: `f56f2f5`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-warmup-sweep-49204447-f56f2f5`
- Candidate path: active512 graph + stateful monotonic
- Same-calculation metadata: PASS
- Token equivalence: PASS against warmup3 and compile-only, `1,084 / 1,084`

| warmup | main model time | main tok/s | delta vs warmup3 |
| ---: | ---: | ---: | ---: |
| `3` | `7.084s` | `153.030` | baseline |
| `1` | `6.812s` | `159.133` | `+4.0%` |
| `0` | `6.675s` | `162.401` | `+6.1%` |

The zero-tolerance per-window gate failed only on tiny one-token windows, while aggregate main throughput and exact token identity were better. This promoted warmup0 to full-song validation.

## Full Song

- Job: `49204568`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
- Stack: torch `2.10.0+cu128`, Transformers `4.57.3`
- Commit: `f56f2f5`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-warmup0-full-isolated-49204568-f56f2f5`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/active-warmup0-full-isolated-49204568-f56f2f5/full_active_w0.profile.json`
- Same-job warmup3 profile: `/work/imt11/Mapperatorinator/runs/active-warmup0-full-isolated-49204568-f56f2f5/full_active_w3.profile.json`
- Previous control profile: `/work/imt11/Mapperatorinator/runs/stateful-bottleneck-49204317-94fcb31/full-control.profile.json`

Against same-job isolated warmup3:

| metric | warmup3 | warmup0 | delta |
| --- | ---: | ---: | ---: |
| main tokens | `7,639` | `7,639` | unchanged |
| main model time | `55.732s` | `52.107s` | `-3.625s`, `-6.5%` |
| main throughput | `137.066 tok/s` | `146.602 tok/s` | `+7.0%` |
| timing throughput | `58.616 tok/s` | `75.130 tok/s` | `+28.2%` |
| total timing+map stage | `74.685s` | `68.283s` | `-6.403s`, `-8.6%` |
| main token equivalence | baseline | PASS, `7,639 / 7,639` | exact |
| timing token equivalence | baseline | PASS, `821 / 821` | exact |

Against the previous untraced active512 graph + stateful control:

| metric | previous control | warmup0 | delta |
| --- | ---: | ---: | ---: |
| main tokens | `7,639` | `7,639` | unchanged |
| main model time | `55.230s` | `52.107s` | `-3.123s`, `-5.7%` |
| main throughput | `138.311 tok/s` | `146.602 tok/s` | `+6.0%` |
| timing throughput | `58.753 tok/s` | `75.130 tok/s` | `+27.9%` |
| total timing+map stage | `74.376s` | `68.283s` | `-6.094s`, `-8.2%` |
| main token equivalence | baseline | PASS, `7,639 / 7,639` | exact |
| timing token equivalence | baseline | PASS, `821 / 821` | exact |

## Regression Check

Aggregate main generation, timing generation, total profiled stage time, generated-token counts, and token equivalence all improved or stayed identical. Strict zero-tolerance per-window gates still reported scoped micro-regressions:

- Same-job warmup3 vs warmup0: one late one-token map window regressed by `0.128ms` model time.
- Previous control vs warmup0: two late one-token map windows regressed by `0.126ms` and `0.487ms` model time.
- Previous control vs warmup0 timing-context: one late one-token timing window regressed only in outer wall by `0.150ms`; model time was slightly faster.

These are not meaningful operational regressions relative to `3.123s` to `3.625s` main-generation model-time savings and `6.094s` to `6.403s` total timing+map stage savings.

## Decision

Keep `inference_active_prefix_decode_cuda_graph_warmup=0` as the default for the default-off active-prefix CUDA graph path. This is a simple exact 5-10% full-song main-generation improvement over the previous fastest opt-in active graph path and improves timing-context throughput as well.

The retained cold single-song baseline remains compile-only SDPA with active-prefix disabled: `92.465 tok/s`. At the time of this note, the fastest exact opt-in path was active512 graph + stateful monotonic + warmup0: `146.602 tok/s`, `52.107s` main model time, `7,639` generated main tokens, token-equivalence PASS.

## Next

From `146.602 tok/s`, reaching `200 tok/s` still requires reducing full-song main model time from `52.107s` to about `38.195s`, another `13.912s` or `26.7%`. The next useful profiling should split the warmup0/stateful active graph path after the warmup removal, because the old logits-processor and graph-capture cost shares are now stale.
