# K-Step Decode Graph Ceiling Probe

## Purpose

Measure whether grouping multiple known-token one-step decoder forwards into a
single CUDA graph could materially reduce per-token graph replay overhead. This
was a diagnostic ceiling only:

- it replayed fixed generated tokens from HF `generate()`;
- it did not solve EOS early-exit behavior;
- it did not solve RNG rollback/overdraw;
- it did not run production inference;
- it is not a throughput claim.

The production exactness blocker remains the forced-EOS audit: a fixed `K`
multi-token graph that samples past the eager stop point can match the output
prefix while diverging in final CUDA RNG state.

## Implementation

Experiment branch:
`experiment/kstep-forward-graph-ceiling`

Commit:
`2c4fd6b40191fedc07f203d6bb585b78b126ab3f`

Temporary utility:
`utils/profile_decode_k_step_graph_ceiling.py`

The utility prepares a seq9 active-prefix decode state, captures known tokens
from HF `generate()`, builds exact one-token prepared inputs for each fixed
step, and compares:

- one-step full forward graph replay;
- fixed `K` sequential full forwards inside one graph replay.

It records CUDA-event and wall time per token. It validates allclose/max-abs
against the direct one-token logits for the fixed tokens.

This branch was not merged to `main`.

## DCC Results

Common setup:

- Config: `profile_salvalai_smoke15`
- Sequence index: `9`
- Precision: `fp32`
- Attention: `sdpa`
- Active-prefix decode length forced to `640`
- Native q1 RoPE/cache self-attention enabled
- Baseline for projection: full-song SALVALAI job `49230082`,
  `7,639` main tokens, `28.243s` model time, `270.475 tok/s`

### K=4

- Job: `49235335`
- Node/GPU: `dcc-chsi-gpu-ferc-s-i11-1`, RTX 2080 Ti
  `GPU-5b9229da-8433-e112-04c8-086f173136fb`
- Report:
  `/work/imt11/Mapperatorinator/runs/kstep-graph-ceiling-49235335/k4_L640.json`

Result:

- Pass: `true`
- One-step graph replay: `2.383444 ms/token`
- K-step graph replay: `2.222698 ms/token`
- Projected saving versus one-step graph replay: `1.214s`
- Graph replay max abs: `0.0`

### K=8

- Job: `49235341`
- Node/GPU: `dcc-chsi-gpu-ferc-s-i11-1`, RTX 2080 Ti
  `GPU-5b9229da-8433-e112-04c8-086f173136fb`
- Report:
  `/work/imt11/Mapperatorinator/runs/kstep-graph-ceiling-49235341/k8_L640.json`

Result:

- Pass: `true`
- One-step graph replay: `2.450269 ms/token`
- K-step graph replay: `2.245907 ms/token`
- Projected saving versus one-step graph replay: `1.543s`
- Graph replay max abs: `0.0`

### K=16

- Job: `49235341`
- Node/GPU: `dcc-chsi-gpu-ferc-s-i11-1`, RTX 2080 Ti
  `GPU-5b9229da-8433-e112-04c8-086f173136fb`
- Report:
  `/work/imt11/Mapperatorinator/runs/kstep-graph-ceiling-49235341/k16_L640.json`

Result:

- Pass: `true`
- One-step graph replay: `2.287157 ms/token`
- K-step graph replay: `2.219168 ms/token`
- Projected saving versus one-step graph replay: `0.513s`
- Graph replay max abs: `0.0`

## Interpretation

The fixed-token K-step ceiling is real but modest. The best observed point was
K=8 at roughly `1.54s` projected full-song saving, barely around the `5%`
threshold before paying any production complexity.

That is not enough to justify a production exact multi-token runtime because a
real implementation would still need to solve:

- exact EOS early exit inside the graph block;
- exact final CUDA RNG state when stopping occurs before K;
- active-prefix bucket transitions inside or across blocks;
- logits-processor and sampling behavior without changing token IDs;
- full-song timing-context and output hash non-regression.

K=16 did not improve the bound; the observed projected saving dropped to about
`0.51s`. This suggests the ceiling is mostly graph/control amortization, not a
large decoder-compute improvement.

## Decision

Do not merge the experiment branch and do not pursue production multi-token
CUDA graphing as the next main path.

Keep multi-token graph work as a later `DecodeSession` runtime idea only if a
future design first proves exact EOS/RNG behavior and has a projected saving
comfortably above `~1.6s`, preferably closer to a `10%` full-song win.

The next serious optimization path remains broad decoder-layer/backend work:
C++/CUDA/CUTLASS/cuBLASLt or an equivalent stable runtime island that reduces
real graph-replayed decoder compute, not only graph launch/control overhead.
