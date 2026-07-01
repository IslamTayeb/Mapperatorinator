# Active-Prefix Decode Loop Full-Song Acceptance

## Summary

Bucketed active-prefix decode loop with bucket size `256` is now an accepted exact-calculation inference speedup for the validated full-song SALVALAI path.

This replaces SDPA plus generation compile as the retained profiling baseline, but only for the scoped simple path that was tested.

## Accepted Baseline

- Commit: `821fb41`
- Job: `49150185`
- Slurm status: `COMPLETED`
- Elapsed: `00:04:15`
- Partition: `gpu-common`
- Node/GPU: `dcc-core-ferc-s-z25-20`, `NVIDIA GeForce RTX 2080 Ti`
- Driver: `595.71.05`
- PyTorch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-prefix256-full-49150185-821fb41`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/active-prefix256-full-49150185-821fb41/compile-active256/beatmapfc1c76f54dbc48a4bbf2097ba5227fc4.osu.profile.json`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/active-prefix256-full-49150185-821fb41/compile-baseline/beatmapd8db783700284b039c0e9a172c47e992.osu.profile.json`

## Hypothesis

Full static-cache self-attention forces most one-token decode steps to attend over the full static target length. Prior SDPA microprofiles showed `q_len=1` attention cost scales strongly with KV length. Decode-only active-prefix self-attention passed one-token logits gates, while active-prefix prefill failed. The candidate therefore leaves prefill unchanged, then uses bucketed active-prefix lengths only during one-token decode.

Bucketing keeps graph/cache shapes reusable while cutting the effective self-attention length for most decode steps.

## Full-Song Result

| run | main tokens | main model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | --- |
| compile-only baseline | 7,639 | 82.142s | 92.998 | baseline |
| active-prefix bucket 256 | 7,639 | 62.653s | 121.926 | PASS, 7,639 / 7,639 |

Delta:

- Throughput: `+28.928 tok/s`, `+31.1%`
- Model time: `-19.489s`, `-23.7%`
- Generated main tokens: unchanged
- Records: unchanged, `87`
- Same-calculation metadata contract: PASS

## Non-Regression Check

The user explicitly wants accepted changes to prove performance has not degraded at all. This run passed that broader check:

| metric | compile-only baseline | active-prefix bucket 256 | result |
| --- | ---: | ---: | --- |
| main generation stage | 82.912s | 63.405s | improved |
| timing generation stage | 24.379s | 20.815s | improved |
| total profiled stage time | 111.531s | 88.389s | improved |
| timing generation model throughput | 34.1 tok/s | 40.0 tok/s | improved |
| generated main tokens | 7,639 | 7,639 | unchanged |

## Smoke Evidence

15s smoke results were not sufficient by aggregate throughput alone:

- Job `49149223`, active bucket `128`: token equivalence PASS but aggregate compile smoke regressed badly because of compile/specialization spikes.
- Job `49149328`, active bucket `512`: token equivalence PASS, aggregate `52.188 tok/s`, post-warmup `~140 tok/s`.
- Job `49149328`, active bucket `256`: token equivalence PASS, aggregate `71.452 tok/s`, post-warmup `~145 tok/s`.

Lesson: for compiler/runtime paths, the 15s smoke slice is an exactness and directional filter, not a final acceptance metric. If aggregate smoke is hurt by one-time specialization but post-warmup windows are clearly faster, a full-song acceptance run can be justified. Full-song equivalence and non-regression remain mandatory.

## Why It Worked

- The earlier direct-step and attention profiles showed full static self-attention length was target-sized.
- Active-prefix during prefill is non-equivalent, but decode-only active-prefix preserved logits in the tested gates.
- Bucket `256` hits a useful tradeoff: it avoids most full-length static-cache attention while keeping reusable compiled shapes.
- Full-song length amortizes the first-window compile/specialization cost that made short smoke aggregates look weak.

## Scope And Risks

Validated scope:

- `batch_size=1`
- `use_server=false`
- `parallel=false`
- `cfg_scale=1.0`
- `num_beams=1`
- static cache
- SDPA
- `inference_generation_compile=true`
- active-prefix decode only
- bucket size `256`

Do not broaden this path to server batching, CFG, beams, parallel sampling, or prefill without fresh one-token logits gates, 15s generated-token equivalence, full-song generated-token equivalence, and full non-regression checks.

## Decision

Keep and push. `configs/inference/profile_salvalai.yaml` should opt into the accepted path with:

```yaml
inference_generation_compile: true
inference_active_prefix_decode_loop: true
inference_active_prefix_decode_bucket_size: 256
```

The new retained baseline is `121.926 tok/s` full-song main generation. Reaching `200 tok/s` now requires reducing main-generation model time from `62.653s` to about `38.195s`, a further `39.0%` reduction.

## Next

- Profile post-warmup active-prefix windows to see whether the remaining cost is now MLP/projection, cross-attention, self-attention, or loop/sampling overhead.
- Compare bucket sizes only with full-song non-regression if smoke shows a plausible signal.
- Prototype a graph-disciplined direct loop only if it can preserve this active-prefix win and reduce the remaining per-window overhead.
- Keep the no-degradation rule: a candidate must not regress timing generation, total profiled stage time, token counts, or exactness while improving main generation.
