# Active64 Post-Bucket Diagnostics

## Purpose

After bucket64 became the fastest exact opt-in active-prefix graph setting, the previous bucket512 diagnostics were stale. This pass measured the new cost mix and checked whether even smaller buckets were worth promoting.

## Full-Song Active64 Diagnostics

- Job: `49207288`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
- Commit: `cf4f87e`
- Run dir: `/work/imt11/Mapperatorinator/runs/active64-nextdiag-49207288-cf4f87e`
- Diagnostic profile: `/work/imt11/Mapperatorinator/runs/active64-nextdiag-49207288-cf4f87e/active64_diag/profile.json`
- Torch trace profile: `/work/imt11/Mapperatorinator/runs/active64-nextdiag-49207288-cf4f87e/active64_seq66_trace/profile.json`
- Trace: `/work/imt11/Mapperatorinator/runs/active64-nextdiag-49207288-cf4f87e/active64_seq66_trace/torch_profiles/000_generation_main_generation_seq66.trace.json`

The untraced diagnostic run stayed close to the accepted active64 full-song result:

| run | main tokens | main model time | main tok/s | timing tok/s |
| --- | ---: | ---: | ---: | ---: |
| accepted active64, job `49206207` | `7,639` | `49.101s` | `155.578` | `76.524` |
| active64 diagnostics, job `49207288` | `7,639` | `48.914s` | `156.171` | `75.903` |

Main-generation diagnostic wall totals:

| counter | wall time |
| --- | ---: |
| `token_append_stop_wall_cpu_s` | `30.269s` |
| `stopping_criteria_wall_cpu_s` | `29.823s` |
| `decode_forward_wall_cpu_s` | `4.990s` |
| `logits_processor_wall_cpu_s` | `4.426s` |
| `prepare_inputs_wall_cpu_s` | `3.848s` |
| `steady_decode_forward_wall_cpu_s` | `3.534s` |
| `first_decode_forward_wall_cpu_s` | `1.456s` |
| `prefill_forward_wall_cpu_s` | `1.214s` |
| `sampling_wall_cpu_s` | `1.156s` |
| `multinomial_wall_cpu_s` | `1.043s` |
| `update_kwargs_wall_cpu_s` | `0.639s` |
| `compile_lookup_wall_cpu_s` | `0.010s` |

Logits processor detail:

| processor | wall time |
| --- | ---: |
| `MonotonicTimeShiftLogitsProcessor` | `2.930s` |
| `TopPLogitsWarper` | `1.242s` |
| `TemperatureLogitsWarper` | `0.156s` |

CUDA graph diagnostics:

| metric | value |
| --- | ---: |
| records with CUDA graph | `83` |
| graphs captured | `198` |
| total capture time | `3.328s` |
| decode replays | `7,552` |
| bucket transitions | `115` |

Normalizing graph captures by active-prefix length and static input tensor shapes reduces the `198` captures to only `11` graph shapes:

| metric | value |
| --- | ---: |
| normalized graph shapes | `11` |
| first-capture floor | `0.175s` |
| duplicate-capture ceiling | `3.153s` |
| duplicate capture share of main model time | `6.446%` |
| estimated tok/s without duplicate capture | `166.931` |

Largest duplicate-capture buckets:

| prefix | duplicate captures | duplicate capture time |
| ---: | ---: | ---: |
| `640` | `63` | `1.075s` |
| `576` | `50` | `0.885s` |
| `704` | `35` | `0.592s` |
| `512` | `18` | `0.315s` |

The graph cache is local to each `active_prefix_decode_generate()` call in `osuT5/osuT5/inference/decode_loop.py`, so graph captures are repeated across generation windows. Bucket64 wins despite that capture tax because it reduces padded active-prefix attention work. A persistent graph/cache runtime could theoretically reclaim up to about `3.15s` full-song capture time, but it is not a small patch and is not enough by itself to reach `200 tok/s`: the captured graph currently closes over per-window cache and encoder-output objects, so reuse across windows would require stable cache/encoder buffers and exact copying/priming discipline.

## Torch Seq66 Trace

The seq66 torch-profiler trace is diagnostic only. It inflates traced runtime and should not be used for throughput claims.

Top attribution from the profile JSON:

| event | count | time |
| --- | ---: | ---: |
| `mapperatorinator.active_prefix.decode_forward.cuda_graph` | `125` | `694.950ms` self CUDA |
| `fmha_cutlassF_f32_aligned_64x64_rf_sm75...` | `2,964` | `379.462ms` self CUDA |
| `cudaStreamSynchronize` | `874` | `241.334ms` CPU |
| `cudaGraphLaunch` | `122` | `96.197ms` CPU |
| GEMV kernel family 1 | `5,856` | `56.551ms` self CUDA |
| `mapperatorinator.active_prefix.stopping_criteria` | `123` | `55.431ms` self CUDA / `320.922ms` CPU total |
| GEMV kernel family 2 | `1,586` | `32.639ms` self CUDA |
| `volta_sgemm_128x64_tn` | `121` | `26.191ms` self CUDA |
| GEMV kernel family 3 | `1,464` | `21.777ms` self CUDA |
| `aten::empty` | `4,591` | `18.451ms` CPU |
| device-to-device memcpy | `11,241` | `16.165ms` self CUDA |

Interpretation:

- There is no obvious small-kernel cleanup left that plausibly saves `>=10%` alone.
- Real CUDA time is still dominated by attention plus many one-token GEMV/GEMM/layernorm/elementwise launches inside the captured decoder forward.
- The large `stopping_criteria_wall_cpu_s` counter is still mostly synchronization/control around the per-token loop. The earlier simple stopping specialization did not remove it.
- Logits processors are a possible small-to-medium target, but exact top-p/sampling semantics and RNG behavior make this risky. A useful fused path would need a verifier gate before any speed claim.
- The next high-ceiling exact path is a bufferized direct runtime with persistent graph/cache/encoder buffers, or deeper kernel/backend work for q_len=1 attention and one-token projection/MLP work.

## Tiny-Bucket Sweep

- Job: `49208036`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-tiny-bucket-sweep-49208036-cf4f87e`
- Config: `profile_salvalai_smoke15`, fixed seed, SDPA, `profile_record_token_ids=true`

All candidates matched bucket64 token IDs for main and timing on the 15s smoke.

| bucket | main tok/s | main model time | timing tok/s | total stage wall | main delta vs first bucket64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `64` first | `170.194` | `6.369s` | `40.038` | `14.633s` | baseline |
| `16` | `153.602` | `7.057s` | `47.149` | `14.247s` | `-9.7%` |
| `32` | `164.636` | `6.584s` | `45.223` | `13.916s` | `-3.3%` |
| `48` | `166.669` | `6.504s` | `48.634` | `13.671s` | `-2.1%` |
| `80` | `169.588` | `6.392s` | `45.671` | `14.065s` | `-0.4%` |
| `96` | `171.338` | `6.327s` | `49.472` | `13.725s` | `+0.7%` |
| `64` repeat | `169.552` | `6.393s` | `46.074` | `14.059s` | `-0.4%` |

## Decision

Do not promote buckets below or near bucket64 from this smoke. Bucket96 was exact and slightly faster on smoke main generation, but only by `+0.7%`, well inside the noise/complexity rejection band and far below the full-song promotion threshold. Bucket16/32/48 were slower on main generation. Keep bucket64 as the current fastest full-song opt-in bucket.

Next work should not be another bucket sweep. The remaining plausible exact gains are:

- persistent/bufferized graph-cache runtime across generation windows;
- logits processor fusion only if a verifier proves exact tokens, logits, and RNG state;
- deeper kernel/backend work for q_len=1 attention and one-token linear/GEMV-heavy decoder work.
