# DecodeSession Tail Graph Runtime Rejection

## Purpose

Test whether the verifier-only tail graph ceiling translates into a real
single-song production speedup. The verifier result was target-sized:
`49246243` projected about `6.134s` full-song main-generation ceiling for a
four-step fixed-start tail graph, so a guarded production one-step tail graph was
worth exactly one smoke attempt.

## Candidate

- Branch: `experiment/tail-graph-runtime`
- Candidate commit: `7891398`
- Revert commit: `aac5b7f`
- DCC job: `49250043`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
  `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Driver/CUDA: NVIDIA driver `595.71.05`, CUDA runtime reported by torch `12.8`
- Torch/Transformers: `torch 2.10.0+cu128`, `transformers 4.57.3`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/tail-graph-runtime-49250043-7891398`
- Control profile:
  `/work/imt11/Mapperatorinator/runs/tail-graph-runtime-49250043-7891398/control/beatmap8192c5bddd124222a77ca445e79cbd0b.osu.profile.json`
- Candidate profile:
  `/work/imt11/Mapperatorinator/runs/tail-graph-runtime-49250043-7891398/candidate/beatmap026b73f6066d4b1e8517987511c9d6b0.osu.profile.json`
- Compare:
  `/work/imt11/Mapperatorinator/runs/tail-graph-runtime-49250043-7891398/compare_strict.json`

The implementation added default-off
`inference_decode_session_tail_cuda_graph=true`, routed through
`inference.py`, `server.py:model_generate()`, and the active-prefix loop. It was
restricted to the accepted batch-1 DecodeSession graph stack and disabled for
timing contexts. It ran one eager tail step to initialize the stateful monotonic
processor, then used a CUDA graph for the map-generation sampling tail.

## Smoke Result

The smoke result was exact but not useful:

| Metric | Control | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Main tokens | `1,084` | `1,084` | same |
| Main model time | `3.740091s` | `3.726128s` | `-0.013963s` |
| Main tok/s | `289.832` | `290.919` | `+0.375%` |
| Main outer wall | `6.004414s` | `4.509362s` | `-24.9%` |
| Total profiled stage wall | `13.731840s` | `11.137653s` | `-18.9%` |
| Token equivalence | PASS | PASS | `1,084 / 1,084` |
| Output artifact hash | PASS | PASS | `ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba` |
| Strict per-window no-regression | PASS | FAIL | 7 failed map windows |

The candidate did execute the tail graph on map windows: by `seq9`, the profile
record showed one tail graph capture and `1,074` cumulative tail graph replays.
The steady high-token windows were flat to slightly worse:

- `seq3`: `320.606 -> 320.168 tok/s`
- `seq4`: `276.193 -> 274.707 tok/s`
- `seq5`: `258.545 -> 255.761 tok/s`
- `seq8`: `272.672 -> 266.681 tok/s`
- `seq9`: `331.309 -> 329.368 tok/s`

The aggregate wall improvement is not a promoted speed claim. The candidate ran
second in the same Slurm allocation and benefited from warmed native extension
and cache state. The synchronized model-time throughput, which is the relevant
single-song TPS metric, improved only `0.4%`.

## Decision

Rejected and reverted.

The production tail graph landed in the wrong optimization band: the
verifier-only graph ceiling looked target-sized, but the production path mostly
removed setup/wall effects and did not reduce steady synchronized model time.
Keeping about `300` lines of runtime complexity for `+0.4%` model-time TPS and
per-window regressions violates the current bottleneck discipline and the
`<5%` default removal rule.

## Lessons

- Verifier ceilings must still be checked against the current bottleneck before
  production integration. A fixed-logit tail graph can look large because it
  captures host/control launch effects around a synthetic fixed shape; the
  production model-time bottleneck remains dominated by decoder compute and the
  broader runtime boundary.
- Do not retry the one-step DecodeSession tail graph runtime as implemented in
  `7891398` unless a fresh diagnostic explains why the steady-window model-time
  regression disappears and projects a real `>=5%` full-song saving.
- Future work should return to measured target-sized areas: broad
  DecodeSession/decoder-layer runtime or native backend work that attacks real
  decoder stack compute plus surrounding per-token control. Narrow tail
  graphing should only reappear as part of that broader runtime island.
