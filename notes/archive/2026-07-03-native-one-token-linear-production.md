# Native One-Token Linear Production Attempt

## Summary

The post-fused native-linear probe was promising enough to try once, but the production path did not clear the keep threshold.

Decision: reject and revert the production wiring. Keep the diagnostic native-linear probe infrastructure.

## Baseline

Current accepted exact opt-in single-song baseline remains DCC job `49230082`:

- `7,639` full-song SALVALAI main tokens
- `28.243s` synchronized main model time
- `270.475 tok/s`
- main/timing fixed-seed token equivalence PASS
- byte-identical generated `.osu`

## Diagnostic Signal

DCC native-linear diagnostic jobs:

| job | commit | result |
| --- | --- | --- |
| `49230317` | `20f3b74` | native block variants projected `0.940s` saved, `279.781 tok/s`; below threshold |
| `49230412` | `16dbda0` | added smaller blocks, projected `1.251s` saved, `283.014 tok/s`; still below threshold |
| `49230428` | `ee7d5ee` | added grouped warp variants, projected `1.419s` saved, `284.779 tok/s`; barely above the `5%` threshold |

The best variant was the same `native_linear_warp4` for all captured one-token decoder linear signatures, so a bounded default-off production attempt was reasonable.

## Candidate

Production candidate commits:

- `51a2c2e`: added default-off `inference_native_one_token_linear`
- `9937caa`: added typed config/root metadata/validation

The candidate wrapped only fp32 CUDA `[1, 1, D]` VarWhisper decoder linears:

- decoder self-attn `Wqkv`
- decoder self/cross-attn `Wo`
- decoder cross-attn `Wq`
- decoder MLP `fc1` and `fc2`
- decoder output projection

It left encoder, prefill, cross-attention `Wkv`, NWhisper `NormLinear`, non-fp32, server, parallel, CFG, and beam paths on the normal implementation.

## Correctness Gates

DCC job `49230641`, commit `9937caa`, RTX 2080 Ti:

- Run dir: `/work/imt11/Mapperatorinator/runs/native-linear-prod-gates-49230641-9937caa`
- One-token gate: PASS, `max_abs=2.6702880859375e-05`
- Direct-loop gate: PASS for `64` sampled steps with token/logit/RNG checks
- 15s smoke main token equivalence: PASS
- 15s smoke timing token equivalence: PASS

15s smoke performance:

| label | control | candidate | delta |
| --- | ---: | ---: | ---: |
| main tok/s | `286.375` | `300.414` | `+4.902%` |
| main model time | `3.785s` | `3.608s` | `-0.177s` |
| main outer wall | `4.592s` | `5.885s` | regression |
| total stage wall | `11.615s` | `12.723s` | regression |

The smoke result was exact but not promotable: model-time gain was just under `5%`, and stage wall regressed due first-use/setup cost.

## Full-Song Result

DCC job `49230859`, commit `9937caa`, RTX 2080 Ti:

- Run dir: `/work/imt11/Mapperatorinator/runs/native-linear-full-49230859-9937caa`
- Control profile: `/work/imt11/Mapperatorinator/runs/native-linear-full-49230859-9937caa/control/beatmap2aa442d8625b445dad4a92ae1d3321c1.osu.profile.json`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/native-linear-full-49230859-9937caa/candidate/beatmapfaba9bc4ae5344bb808913502452c948.osu.profile.json`
- Strict compare: `/work/imt11/Mapperatorinator/runs/native-linear-full-49230859-9937caa/compare_strict_full.json`
- `.osu` byte compare: PASS, `cmp_exit=0`, both `31,709` bytes
- Main token equivalence: PASS, `7,639 / 7,639`
- Timing token equivalence: PASS, `821 / 821`

| label | control | candidate | delta |
| --- | ---: | ---: | ---: |
| main tok/s | `273.563` | `281.743` | `+2.990%` |
| main model time | `27.924s` | `27.113s` | `-0.811s` |
| main outer wall | `89.352s` | `88.906s` | `-0.499%` |
| timing tok/s | `95.582` | `101.324` | `+6.007%` |
| total stage wall | `103.084s` | `101.938s` | `-1.111%` |

Strict full-song comparison still failed per-window no-regression:

- main: `5` failed windows
- timing: `79` failed windows

The timing aggregate improved, but the production flag is disabled for timing contexts, so that timing result should be treated as order/noise rather than a real timing optimization.

## Why It Failed To Graduate

The end-to-end model-time gain was only `+2.99%`, well below the `5%` strategic keep threshold and far below the normal `>=10%` graduation threshold.

The diagnostic projection overestimated production value because isolated per-linear CUDA-event wins do not add cleanly under the real DecodeSession/CUDA-graph generation path. The production wrapper also adds model-call-site complexity and another native extension surface area.

## Decision

Reverted production wiring:

- `09cda4f`: revert validation/metadata commit
- `55a80cb`: revert default-off production path commit

Keep:

- `osuT5/osuT5/inference/native_linear.py`
- `utils/profile_decode_linear_kernels.py --native-linear-variant`

Do not retry a per-linear native wrapper unless new profiling shows a full-song projected win comfortably above `5%` and explains why the `49230859` production result is stale.

Next linear work, if any, should target a broader exact decoder island: native/CUTLASS/cuBLASLt fused MLP, projection plus adjacent layout, or a larger decoder-layer runtime boundary. A single replacement kernel per `nn.Linear` is not enough.
