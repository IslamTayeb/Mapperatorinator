# Simple Stopping Scout Rejection

## Hypothesis

The active-prefix CUDA graph path still spends surprising wall time around token append and stopping checks after the model forward is captured. Job `49204765` showed the warmup0 + stateful active graph path at `146.963 tok/s` with default omitted warmup, but diagnostics attributed large CPU-side wall buckets to:

| diagnostic bucket | full-song wall time |
| --- | ---: |
| `token_append_stop_wall_cpu_s` | `35.276s` |
| `stopping_criteria_wall_cpu_s` | `34.826s` |
| `logits_processor_wall_cpu_s` | `4.452s` |
| `prepare_inputs_wall_cpu_s` | `3.874s` |
| `decode_forward_wall_cpu_s` | `3.444s` |
| `sampling_wall_cpu_s` | `1.167s` |

The seq9 torch-profiler trace in the same job also showed `aten::_local_scalar_dense` and `aten::isin` activity near stopping checks. A narrow batch-1 stopping fast path looked worth testing because the active graph path is already limited to simple generation: `use_server=false`, `parallel=false`, `cfg_scale=1`, `num_beams=1`, and static cache.

## Implementation Tested

A local dirty patch specialized active-prefix batch-1 stopping for the common `MaxLengthCriteria` and `EosTokenCriteria` case and skipped the generic `unfinished_sequences`/EOS masking path when the fast path was valid. It also extended `utils/verify_direct_decode_loop.py` so the direct-loop correctness gate used the same candidate stopping path.

This patch was never committed.

## Evidence

- Job: `49204960`
- Run dir: `/work/imt11/Mapperatorinator/runs/simple-stop-smoke-49204960-bb11d9f-simple-stop-dirty`
- Baseline smoke profile: `/work/imt11/Mapperatorinator/runs/active-warmup-sweep-49204447-f56f2f5/active_warmup0.profile.json`
- Candidate smoke profile: `/work/imt11/Mapperatorinator/runs/simple-stop-smoke-49204960-bb11d9f-simple-stop-dirty/candidate.profile.json`
- Candidate diagnostic profile: `/work/imt11/Mapperatorinator/runs/simple-stop-smoke-49204960-bb11d9f-simple-stop-dirty/candidate_diag.profile.json`
- Direct gate: `/work/imt11/Mapperatorinator/runs/simple-stop-smoke-49204960-bb11d9f-simple-stop-dirty/direct_gate_seq9_64.json`

The direct-loop graph gate passed over 64 sampled steps: generated tokens matched, raw logits matched within tolerance, final RNG state matched, and reference/candidate both stopped on `max_new_tokens`.

| run | main tokens | main model time | main tok/s | token equivalence |
| --- | ---: | ---: | ---: | --- |
| active512 graph + stateful + warmup0 baseline | `1,084` | `6.675s` | `162.401` | baseline |
| simple stopping candidate | `1,084` | `6.433s` | `168.512` | PASS, `1,084 / 1,084` |

Timing context also stayed exact and improved only slightly:

| run | timing tok/s | token equivalence |
| --- | ---: | --- |
| active512 graph + stateful + warmup0 baseline | `47.556` | baseline |
| simple stopping candidate | `48.209` | PASS |

Strict comparison failed on total stage wall because the candidate ran in a colder model/cache context than the older baseline (`13.960s -> 22.254s`) and one early one-token map window regressed by `20.9ms`. The aggregate main model-time result is still the relevant signal for this smoke: `+3.76%`.

The candidate diagnostic run showed the wall bucket was not actually removed:

| diagnostic bucket | candidate 15s wall time |
| --- | ---: |
| `token_append_stop_wall_cpu_s` | `4.296s` |
| `stopping_criteria_wall_cpu_s` | `4.265s` |
| `logits_processor_wall_cpu_s` | `0.547s` |
| `decode_forward_wall_cpu_s` | `0.448s` |
| `prepare_inputs_wall_cpu_s` | `0.391s` |
| `sampling_wall_cpu_s` | `0.166s` |

The most likely interpretation is that the large stopping bucket includes required per-token CPU/GPU synchronization and loop-control waiting, not just avoidable Hugging Face stopping-criteria Python overhead. The specialization moved a little overhead, but did not change the dominant synchronization structure.

## Decision

Reject and revert. The candidate was exact and had a small positive smoke signal (`+3.76%` main), but it is below the keep threshold for a custom stopping path, does not explain or remove the dominant wall bucket, and does not justify a full-song run.

Do not retry simple Python stopping-criteria specialization unless a new runtime design avoids the per-token synchronization/control dependency while preserving exact EOS/max-length semantics and fixed-seed token identity.
