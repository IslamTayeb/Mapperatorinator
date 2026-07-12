# Post-Fused Active-Prefix Diagnostics

## Purpose

Explain the gap between accepted production synchronized model time and isolated decoder-forward graph replay after the fused RoPE/cache self-attention win.

Accepted exact opt-in baseline from DCC job `49230082`:

- main tokens: `7,639`
- synchronized main model time: `28.243s`
- throughput: `270.475 tok/s`
- main/timing token equivalence: PASS
- generated `.osu`: byte-identical to same-job fused-off control

The post-fused weighted replay job `49230173` measured full-forward CUDA graph replay at about `17.000s`, leaving roughly `11.2s` of production synchronized model time outside isolated forward replay. This pass enables active-prefix decode diagnostics on the current fused stack to see where the surrounding control cost appears.

## Run

- DCC job: `49230249`
- Commit: `e63e022`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Driver: `595.71.05`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Run dir: `/work/imt11/Mapperatorinator/runs/fused-active-diag-full-49230249-e63e022`
- Profile: `/work/imt11/Mapperatorinator/runs/fused-active-diag-full-49230249-e63e022/output/beatmap73fba522607441e38caafbda290ae560.osu.profile.json`
- Active summary: `/work/imt11/Mapperatorinator/runs/fused-active-diag-full-49230249-e63e022/active_main_diag.json`

Preflight job `49230248` produced no evidence; it failed immediately because local shell expansion mangled the Slurm script variables.

Flags matched the current fastest exact opt-in stack plus:

```text
profile_active_prefix_decode_diagnostics=true
```

## Equivalence

Comparison target: accepted profile `/work/imt11/Mapperatorinator/runs/fused-rope-cache-full-49230082-d7b8684/candidate.profile.json`.

| label | token result | generated tokens | model-time delta | note |
| --- | --- | ---: | ---: | --- |
| main generation | PASS | `7,639 / 7,639` | `+0.031s`, `-0.1% tok/s` | diagnostic-only, not a speed claim |
| timing context | PASS | `821 / 821` | `-0.024s`, `+0.3% tok/s` | diagnostic-only |

Strict no-regression failed because diagnostics inflated outer wall, especially `main_generation.seq0` wall (`62.077s`) while synchronized model time stayed normal. Treat this run only as attribution. It is not a promoted throughput result.

## Main-Generation Diagnostics

Aggregate with diagnostics:

| metric | value |
| --- | ---: |
| records | `87` |
| tokens | `7,639` |
| model time | `28.274s` |
| throughput | `270.176 tok/s` |
| stage wall | `89.115s` |

Largest CPU-side diagnostic spans:

| span | wall |
| --- | ---: |
| token append + stopping | `13.036s` |
| stopping criteria | `12.591s` |
| logits processors | `4.489s` |
| prepare inputs | `2.682s` |
| decode forward span | `1.608s` |
| steady decode forward span | `1.578s` |
| cache position setup | `1.214s` |
| sampling | `1.142s` |
| prefill forward span | `1.124s` |
| multinomial | `1.030s` |
| update kwargs | `0.628s` |

Logits processor split:

| processor | wall | calls |
| --- | ---: | ---: |
| `MonotonicTimeShiftLogitsProcessor` | `3.007s` | `7,639` |
| `TopPLogitsWarper` | `1.230s` | `7,639` |
| `TemperatureLogitsWarper` | `0.156s` | `7,639` |

CUDA graph capture is not target-sized anymore:

| metric | value |
| --- | ---: |
| graph captures | `198` |
| normalized graph shapes | `11` |
| decode replays | `7,552` |
| capture seconds | `0.143s` |
| duplicate capture seconds | `0.117s` |
| duplicate capture share of model time | `0.412%` |
| estimated tok/s without duplicate capture | `271.294` |

## Interpretation

The current production gap is not duplicate graph capture and not Python dictionary cleanup. The diagnostics show large CPU-side spans around stopping, logits processors, sampling, and per-token control. Those spans are not exclusive GPU-kernel timings; they can include synchronization caused by host-side loop decisions over CUDA tensors.

The key exactness constraint is EOS stopping. Full-song generated lengths vary widely:

- main windows: `1` to `586` generated tokens
- timing windows: `1` to `51` generated tokens

So the loop cannot simply run a fixed number of tokens on GPU and trim later without changing generated-token behavior. A valid loop/runtime optimization must preserve EOS stopping, sampling RNG, logits processor behavior, and generated-token counts.

This explains why narrow kernels alone are unlikely to reach `500 tok/s`: the weighted forward replay ceiling and the control/synchronization gap both matter.

## Decision

No optimization graduated. Keep job `49230082` as the accepted exact opt-in single-song baseline.

Next diagnostic: rerun direct-loop tail diagnostics on the current fused stack. The old tail diagnostic was just below the old `5%` threshold; after the fused win, the threshold is lower, so tail work deserves one fresh measurement before being ruled out again.
