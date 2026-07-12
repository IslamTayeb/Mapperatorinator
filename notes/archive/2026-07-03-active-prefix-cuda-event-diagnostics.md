# Active-Prefix CUDA Event Diagnostics

## Purpose

Add CUDA-event timing to active-prefix decode diagnostics so the CPU wall spans from earlier runs can be separated from queued GPU work. The goal was attribution only, not a throughput claim.

Current accepted exact opt-in single-song baseline remains the fused RoPE/cache self-attention stack from job `49230082`: `7,639` main tokens, `28.243s` synchronized model time, `270.475 tok/s`, main/timing token equivalence PASS, byte-identical `.osu`.

## Run

- DCC job: `49231413`
- Commit: `6f126d3680ffbc606d3049d495937afcb13e2fc6`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti `GPU-8de8e67b-289d-a1ae-3745-25f40db81f55`
- Driver/CUDA: `595.71.05` / `13.2`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-diag-cuda-events-49231413-6f126d3`
- Control profile: `/work/imt11/Mapperatorinator/runs/active-diag-cuda-events-49231413-6f126d3/control/beatmap60596c76ff9b4c21a13d93e98704fef1.osu.profile.json`
- Diagnostic profile: `/work/imt11/Mapperatorinator/runs/active-diag-cuda-events-49231413-6f126d3/diagnostic/beatmapa7062cbcf44f4aa8a7c13f5fb51b6b6f.osu.profile.json`

Flags matched the current fused opt-in stack plus:

```text
profile_active_prefix_decode_diagnostics=true
```

## Equivalence

| label | token result | note |
| --- | --- | --- |
| main generation | PASS | same-calculation metadata PASS, generated tokens match |
| timing context | PASS | same-calculation metadata PASS, generated tokens match |
| `.osu` output | PASS | control and diagnostic outputs were byte-identical |

The diagnostic run changed timing and synchronized model time because it creates many CUDA events and synchronizes while finalizing diagnostics. Do not use this job for throughput claims.

## Main-Generation Event Split

Diagnostic 15s middle-song slice:

| metric | value |
| --- | ---: |
| records | `10` |
| records with diagnostics | `10` |
| generated tokens | `1,084` |
| decode steps | `1,074` |
| diagnostic model time | `4.243s` |
| diagnostic throughput | `255.461 tok/s` |

Largest CUDA-event ranges:

| range | CUDA event time | weighted per decode step | calls |
| --- | ---: | ---: | ---: |
| `decode_forward.cuda_graph` | `2464.025ms` | `2294.250us` | `1,074` |
| `prepare_inputs` | `542.224ms` | `504.864us` | `1,084` |
| `stopping_criteria` | `298.459ms` | `277.895us` | `1,084` |
| `prefill_forward` | `237.011ms` | `220.681us` | `10` |
| `logits_processor` | `107.854ms` | `100.423us` | `1,084` |
| `finished_check` | `59.789ms` | `55.670us` | `1,084` |
| `sampling.multinomial` | `46.728ms` | `43.509us` | `1,084` |
| `eos_mask` | `10.588ms` | `9.859us` | `1,084` |
| `update_model_kwargs` | `9.029ms` | `8.407us` | `1,084` |
| `sampling.softmax` | `7.029ms` | `6.545us` | `1,084` |

Largest CPU-side wall spans from the same records:

| span | wall |
| --- | ---: |
| token append + stopping | `1.065s` |
| stopping criteria | `0.887s` |
| logits processors | `0.735s` |
| prepare inputs | `0.580s` |
| decode forward span | `0.465s` |
| sampling | `0.335s` |

Logits processor wall split:

| processor | wall | calls |
| --- | ---: | ---: |
| `MonotonicTimeShiftLogitsProcessor` | `0.394s` | `1,084` |
| `TopPLogitsWarper` | `0.237s` | `1,084` |
| `TemperatureLogitsWarper` | `0.029s` | `1,084` |

## Timing-Context Event Split

Timing context is not the main 500 tok/s target, but it exposed a cold timing-model compile/setup tax:

| range | CUDA event time | weighted per decode step | calls |
| --- | ---: | ---: | ---: |
| `setup.compile_lookup` | `1473.069ms` | `9565.383us` | `10` |
| `decode_forward.cuda_graph` | `429.863ms` | `2791.319us` | `154` |
| `prefill_forward` | `200.774ms` | `1303.729us` | `10` |
| `prepare_inputs` | `124.955ms` | `811.396us` | `164` |
| `stopping_criteria` | `74.117ms` | `481.278us` | `164` |
| `logits_processor` | `69.774ms` | `453.076us` | `164` |

## Interpretation

The diagnostic confirms that the earlier CPU wall ranges were misleading for the model-forward region: `decode_forward_wall_cpu_s` was only `0.465s` on the 15s main slice because graph replay is asynchronous, while CUDA events attribute `2.464s` to `decode_forward.cuda_graph`.

For main generation, the largest remaining measured bucket is still the captured one-token decoder forward. Tail/control CUDA-event ranges are real and visible, but the individually obvious ones are much smaller than decode graph replay. This supports the current direction:

- do not chase duplicate graph capture or standalone sampling fusion as the next primary path;
- keep tail work as part of a broader `DecodeSession`/native-runtime island, where exact EOS/stopping and RNG behavior can still be preserved;
- if pursuing `500 tok/s`, target broad one-token decoder compute and loop/runtime synchronization together.

The event ranges are diagnostic-only and non-exclusive. They can include queued work between range entry and exit on the current stream, and the diagnostic run pays extra event-recording/final-sync overhead.

## Decision

No speed optimization graduated. Keep job `49230082` as the accepted exact opt-in single-song baseline.

Keep the CUDA-event diagnostic instrumentation and update `utils/summarize_active_prefix_diagnostics.py` so future runs can report event totals without ad hoc parsers.

Next useful implementation project remains broad and verifier-first: a measured decoder-layer/runtime island that can reduce `decode_forward.cuda_graph` and the per-token control/synchronization cost together. A narrow logits/sampling kernel is not target-sized from this evidence.
