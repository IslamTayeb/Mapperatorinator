# Active-Prefix Decode Diagnostics

## Summary

Added a default-off active-prefix diagnostic flag:

```text
profile_active_prefix_decode_diagnostics=true
```

This is profiling infrastructure only. It does not change the retained cold single-song baseline and does not claim an inference speedup.

## What It Records

When active-prefix decode is enabled, generation profile records can now include `active_prefix_decode_diagnostics` with CPU-side wall counters for:

- cache-position setup and max-cache-shape lookup;
- generation compile-call lookup;
- `prepare_inputs_for_generation`;
- normal prefill forward;
- first and steady active-prefix decode forward;
- `_update_model_kwargs_for_generation`;
- logits extraction and logits processors;
- sampling;
- token append and stopping checks;
- `decode_steps`, `bucket_lengths_seen`, and `bucket_transition_count`.

The decode loop does not add tensor `.cpu()`, `.item()`, `.tolist()`, RNG inspection, or CUDA synchronization. NVTX/`record_function` spans use stable names such as `mapperatorinator.active_prefix.decode_forward`; bucket lengths are recorded in JSON counters instead of dynamic per-token range names.

## Use

Use it for torch-profiler or Nsight diagnosis only:

```bash
profile_active_prefix_decode_diagnostics=true \
profile_generation_detail_ranges=true
```

For JSON counters without broader VarWhisper ranges, use only `profile_active_prefix_decode_diagnostics=true`.

## Validation Plan

- Compile and local dependency-light tests.
- DCC import/compile checks.
- Direct-loop token/logit/RNG gate with active-prefix diagnostics enabled.
- 15s smoke token-equivalence check before treating any diagnostic output as trustworthy.

## DCC Validation

DCC job `49164750` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, commit `c7ab3b8`:

- Run dir: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-smoke-49164750-c7ab3b8`
- Logs: `/work/imt11/Mapperatorinator/logs/ap-diag-smoke-49164750.out` and `.err`
- No-diagnostics profile: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-smoke-49164750-c7ab3b8/active512-nodiag/beatmapfdc929262db74906b8ddac18927ad2fc.osu.profile.json`
- Diagnostics profile: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-smoke-49164750-c7ab3b8/active512-diag/beatmap1d886eb6cffd4cc7b967e38944a3d716.osu.profile.json`
- Compare JSONs: `compare-main-token.json` and `compare-timing-token.json`

The Slurm job state is `FAILED` because the final ad-hoc Python reporting snippet had a shell quoting bug (`NameError: name 'tokens_per_second' is not defined`) after the profiles and compare files were already written. Treat the profiling artifacts as valid but the wrapper as failed.

Main-generation comparison:

| run | main tokens | main model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | --- |
| active512 no diagnostics | `1,084` | `31.303s` | `34.629` | baseline |
| active512 diagnostics | `1,084` | `31.309s` | `34.623` | PASS |

Timing-context comparison:

| run | timing tokens | timing model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | --- |
| active512 no diagnostics | `164` | `35.640s` | `4.602` | baseline |
| active512 diagnostics | `164` | `34.649s` | `4.733` | PASS |

Zero-tolerance per-window no-regression failed on small noise: main windows moved by roughly sub-percent to `1.3%`, while aggregate main throughput changed by only `-0.02%`. That is acceptable for a default-off diagnostics switch, but it is not a speed claim.

Aggregated diagnostics across the 20 generation records:

| counter | value |
| --- | ---: |
| `decode_steps` | `1,228` |
| `decode_forward_wall_cpu_s` | `54.791s` |
| `first_decode_forward_wall_cpu_s` | `30.560s` |
| `steady_decode_forward_wall_cpu_s` | `24.231s` |
| `logits_processor_wall_cpu_s` | `6.040s` |
| `compile_lookup_wall_cpu_s` | `1.900s` |
| `prepare_inputs_wall_cpu_s` | `0.847s` |
| `sampling_wall_cpu_s` | `0.301s` |

Map-only split:

| counter | value |
| --- | ---: |
| `decode_steps` | `1,074` |
| `decode_forward_wall_cpu_s` | `23.953s` |
| `first_decode_forward_wall_cpu_s` | `11.657s` |
| `steady_decode_forward_wall_cpu_s` | `12.296s` |
| `logits_processor_wall_cpu_s` | `5.250s` |
| `sampling_wall_cpu_s` | `0.243s` |

Interpretation: active-prefix cold weakness is still mostly first decode forward graph/runtime/specialization cost. The first long map window (`seq3`) paid `11.538s` in first decode forward and ran at `18.764 tok/s`, while later windows reached about `116-138 tok/s`. Fused sampling/logits processors are not the next primary target yet; graph stabilization, explicit priming with honest setup accounting, or a bufferized/direct decode ABI remains higher leverage.
