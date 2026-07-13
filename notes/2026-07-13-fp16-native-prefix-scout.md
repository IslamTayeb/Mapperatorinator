# FP16 Native-Prefix Single-Song Speed Scout

## Decision

Stop at the real-tensor component gate. Do not add an inference selector or
runtime wiring, and do not run the fixed-work, smoke, full-song candidate, or
quality ladders.

The sentinel measurement covers `4,147 / 7,597` current accepted decode graph
replays (`54.59%`). Unmeasured buckets receive zero candidate regression in the
projection, yet every candidate is already slower than the accepted engine and
misses the `25.419s` target.

## Current accepted baseline

Job `49714108` refreshed the current-main SALVALAI inventory on one RTX 2080 Ti:

- main model time: `28.243949s`;
- generated main tokens: `7,684` (`+0.59%` versus the requested `7,639` work
  reference);
- main model throughput: `272.058 tok/s`;
- decode graph replays: `7,597`;
- active-prefix counts: `128:30`, `192:64`, `256:64`, `320:71`, `384:161`,
  `448:448`, `512:1166`, `576:1647`, `640:2470`, `704:1142`, `768:242`,
  `832:92`.

The historical 11-bucket distribution was stale: current main has a real `832`
bucket. The target remains the predeclared `25.419s` main time.

## Sentinel component result

Measurement job: `49715303`, commit
`8f27b7b354a82e5c2cf3826c9ccc2600fa5a7e51`, RTX 2080 Ti, Torch
`2.10.0+cu128`, CUDA `12.8`. Candidate graphs used production-style capture
warmup `0`; reciprocal timing used `100` warmup and `1,000` measured replays.

Full one-token model CUDA-graph replay, milliseconds per call:

| Variant | Prefix 128 | Prefix 576 | Prefix 640 |
| --- | ---: | ---: | ---: |
| Accepted FP32 | `1.7919` | `2.1447` | `2.1941` |
| Shared FP32 framework | `2.4699` | `2.5530` | `2.5755` |
| Shared FP32 native self+cross | `2.6839` | `3.0477` | `3.1096` |
| Shared FP16 framework | `2.8238` | `2.9440` | `2.9587` |
| Shared FP16 native self+cross | `2.7347` | `3.0997` | `3.1460` |

Conservative current-prefix projections, charging positive graph setup delta
and assigning zero delta to all unmeasured buckets:

| Candidate | Projected main | Projected TPS | Decision |
| --- | ---: | ---: | --- |
| Shared FP32 framework | `29.922s` | `256.8` | stop |
| Shared FP32 native self+cross | `32.028s` | `239.9` | stop |
| Shared FP16 framework | `31.699s` | `242.4` | stop |
| Shared FP16 native self+cross | `32.213s` | `238.5` | stop |

At the matched one-layer boundary, FP32 native is `20.15%` slower than FP32
framework and FP16 native is `6.40%` slower than FP16 framework. Neither clears
the required `5%` native-retention advantage.

## Correctness and drift

All variants pass the relevant component gates:

- finite outputs and caches;
- valid cache shapes and active-slot writes;
- untouched before/future self-cache slots;
- unchanged cross cache;
- stable cache storage ownership;
- exact candidate self-repeat output/cache and unchanged RNG state;
- exact graph replay repeat and stable measured memory.

Baseline divergence is classified as `documented-drift`, not a failure. FP16
maximum observed differences across the sentinels reached `0.6717` in a cache
key slot and `0.6633` in logits. The scout therefore answers the speed question
without making a quality claim.

## Setup and artifacts

The shared FP32/FP16 q1 extension cold build passed separately in job
`49714943` (`89s`). Extension build time is excluded from graph setup; the
measurement reports it separately. The reusable implementation keeps framework
linear operations for both dtypes and uses the dtype-generic native kernel only
for fused self RoPE/cache/q1 attention and cross q1 attention.

Artifacts:

- `/work/imt11/Mapperatorinator/runs/fp16-prefix-sentinel-measure2-49715303/component.json`
- `/work/imt11/Mapperatorinator/runs/fp16-prefix-sentinel-measure2-49715303/projection.json`
- `/work/imt11/Mapperatorinator/runs/fp16-prefix-sentinel-measure2-49715303/baseline-capture/`
- `/work/imt11/Mapperatorinator/runs/fp16-prefix-sentinel-measure2-49715303/nvidia-smi.csv`

Revisit only if a materially different half-precision linear/attention
implementation first demonstrates at least `5%` weighted full-model component
headroom on the target hardware, or if the target GPU/framework changes. Do not
reopen this framework-FP16/native-q1 shape from isolated half-kernel results.
