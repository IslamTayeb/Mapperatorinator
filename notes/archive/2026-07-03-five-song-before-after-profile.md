# Five-Song Before/After DCC TPS Profile

## Summary

Ran the requested before/after benchmark on five realistic songs: Lambada, PEGASUS, Ela ke Leitada, SALVALAI, and Nube Negra. "Before" means original-repo-equivalent flags disabled on the current instrumented branch, not literal upstream `origin/main`, because upstream lacks the current profiling suite and token-accounting gates.

Official TPS numbers below come only from untraced `profile_inference` synchronized model time. Nsight Systems and `nvidia-smi` telemetry were collected for GPU diagnosis, but traced wall time is not used for TPS claims.

Result: the current optimized opt-in path is exact on all five songs and reaches `195.545 tok/s` cold separate-process aggregate main generation, with one cold first song in the together suite at `201.749 tok/s`.

## Run Setup

- DCC repo: `/hpc/group/romerolab/imt11/projects/Mapperatorinator`
- DCC env: `/hpc/group/romerolab/imt11/envs/mapperatorinator`
- Commit: `8a2de7281f13a71d92003200d409dc7a07629cc7`
- Run root: `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72`
- Log root: `/work/imt11/Mapperatorinator/logs/five-song-profile-20260702-232516-8a2de72`
- Node: `dcc-core-ferc-s-z25-20`
- GPUs: one RTX 2080 Ti per Slurm job, assigned by Slurm.

Slurm jobs submitted together:

| case | job | status | elapsed | GPU UUID |
| --- | ---: | --- | ---: | --- |
| `before_together` | `49218365` | COMPLETED `0:0` | `00:51:12` | `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1` |
| `after_together` | `49218366` | COMPLETED `0:0` | `00:16:13` | `GPU-21baa305-92c5-a528-5ae8-f2001d7aac38` |
| `before_separate` | `49218367` | COMPLETED `0:0` | `00:19:40` | `GPU-de50b870-b708-690d-74a4-6e855163a133` |
| `after_separate` | `49218368` | COMPLETED `0:0` | `00:08:51` | `GPU-ba708720-cba9-538c-6e0f-ecaea3486d09` |

Common config:

- `gamemode=0`
- `difficulty=6.3`
- `circle_size=4`
- `overall_difficulty=8.5`
- `approach_rate=9.5`
- `hp_drain_rate=5`
- `slider_multiplier=1.6`
- `slider_tick_rate=1`
- `hitsounded=true`
- `seed=12345`
- `precision=fp32`
- `attn_implementation=sdpa`
- `use_server=false`
- `parallel=false`
- `cfg_scale=1.0`
- `num_beams=1`
- `profile_inference=true`
- `profile_sync_cuda=true`
- `profile_record_token_ids=true`

Before flags:

- `inference_generation_compile=false`
- `inference_active_prefix_decode_loop=false`
- `inference_active_prefix_decode_cuda_graph=false`
- `inference_stateful_monotonic_logits_processor=false`
- `inference_q1_bmm_cross_attention=false`

After flags:

- `inference_generation_compile=true`
- `inference_active_prefix_decode_loop=true`
- `inference_active_prefix_decode_bucket_size=64`
- `inference_active_prefix_decode_cuda_graph=true`
- `inference_active_prefix_decode_cuda_graph_warmup=0`
- `inference_active_prefix_decode_cuda_graph_min_decode_steps=1`
- `inference_stateful_monotonic_logits_processor=true`
- `inference_q1_bmm_cross_attention=true`

## Audio Provenance

Prepared once under `/work/imt11/Mapperatorinator/data/five-song-profile/` using isolated `yt-dlp 2026.6.9` at `/work/imt11/Mapperatorinator/tools/yt-dlp-20260703`; the main env was not mutated.

| song | URL | duration | codec | sha256 |
| --- | --- | ---: | --- | --- |
| Lambada | `https://www.youtube.com/watch?v=iyLdoQGBchQ` | `206.890s` | mp3 | `1e692d9ffbc2a3051219ac19e96f2c32b428fa0e4fc36278022866e19e6e6183` |
| PEGASUS | `https://www.youtube.com/watch?v=x_NfautnhQo` | `174.649s` | mp3 | `a8de11b2c507f596f63bd3ff452b8d888b37c38df83f06b2ea968cb222378204` |
| Ela ke Leitada | `https://www.youtube.com/watch?v=AqLATnmNyHY` | `149.734s` | mp3 | `59b961e46f1242b3be280de376103035d20bf89e9c3796e54df5ace93f35d646` |
| SALVALAI | `https://www.youtube.com/watch?v=_VmnBu3IZ9s` | `156.992s` | mp3 | `f27c3472fe1c5abc022ddcf445cfe91a7a955e05474d62930fffa5e6d7148f0b` |
| Nube Negra | `https://www.youtube.com/watch?v=MBJcmVQC2WI` | `115.667s` | mp3 | `e22d5c37f4ed602edc09c441ce11c85a27cd7427a89053ff0cce81d12b816983` |

## Aggregate TPS

| class | tokens | before model time | before main tok/s | after model time | after main tok/s | speedup | timing tok/s | total stage wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `separate_cold` | `39,497` | `609.504s` | `64.802` | `201.984s` | `195.545` | `+201.8%` | `38.879 -> 82.817` | `743.129s -> 277.842s` |
| `together_first_run` | `9,501` | `147.043s` | `64.614` | `47.093s` | `201.749` | `+212.2%` | `36.129 -> 78.842` | suite manifest |
| `together_first_pass` | `39,497` | `609.853s` | `64.765` | `202.062s` | `195.470` | `+201.8%` | `39.571 -> 98.548` | suite manifest |
| `together_all` | `78,994` | `1304.805s` | `60.541` | `405.532s` | `194.791` | `+221.8%` | `38.458 -> 100.649` | suite manifest |
| `together_warmed` | `39,497` | `694.952s` | `56.834` | `203.471s` | `194.116` | `+241.5%` | `37.405 -> 102.843` | suite manifest |

Interpretation:

- The optimized path is not just warm-only. Cold separate-process aggregate is `195.545 tok/s`, and the first run in the long-lived process is `201.749 tok/s`.
- The together suite is serial multi-song / long-lived-process evidence, not true concurrent batching.
- The before/no-optimization repeat got slower in the warmed repeat (`64.765 -> 56.834 tok/s`), while the after path stayed stable around `194-195 tok/s`.

## Separate Cold Per-Song TPS

| song | tokens | before main tok/s | after main tok/s | speedup | main model time | timing tok/s | total stage wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lambada | `9,501` | `64.602` | `201.630` | `+212.1%` | `147.071s -> 47.121s` | `34.545 -> 81.423` | `179.352s -> 63.760s` |
| PEGASUS | `8,521` | `64.331` | `195.664` | `+204.2%` | `132.457s -> 43.549s` | `39.841 -> 86.529` | `161.387s -> 59.723s` |
| Ela ke Leitada | `7,858` | `64.481` | `191.604` | `+197.1%` | `121.866s -> 41.012s` | `41.487 -> 84.907` | `149.111s -> 56.858s` |
| SALVALAI | `8,134` | `65.183` | `194.792` | `+198.8%` | `124.788s -> 41.757s` | `39.412 -> 82.870` | `151.472s -> 56.990s` |
| Nube Negra | `5,483` | `65.804` | `192.081` | `+191.9%` | `83.323s -> 28.545s` | `40.527 -> 76.361` | `101.807s -> 40.511s` |

## Together Warmed Per-Song TPS

| song | tokens | before warmed main tok/s | after warmed main tok/s | speedup | main model time | timing tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lambada | `9,501` | `51.399` | `198.405` | `+286.0%` | `184.848s -> 47.887s` | `34.971 -> 96.677` |
| PEGASUS | `8,521` | `53.003` | `194.566` | `+267.1%` | `160.764s -> 43.795s` | `35.746 -> 104.396` |
| Ela ke Leitada | `7,858` | `66.337` | `190.324` | `+186.9%` | `118.456s -> 41.288s` | `39.163 -> 105.815` |
| SALVALAI | `8,134` | `54.282` | `193.684` | `+256.8%` | `149.848s -> 41.996s` | `38.269 -> 104.145` |
| Nube Negra | `5,483` | `67.662` | `192.351` | `+184.3%` | `81.035s -> 28.505s` | `40.819 -> 104.329` |

## Equivalence And Gates

Strict main-generation profile comparisons passed for all five separate cold pairs:

- Lambada: PASS
- PEGASUS: PASS
- Ela ke Leitada: PASS
- SALVALAI: PASS
- Nube Negra: PASS

Strict suite comparisons passed:

- `--suite-scope all_runs --strict`: PASS
- `--suite-scope warmed_runs --strict`: PASS
- `--suite-scope all_runs --strict --gate-cold-run0`: PASS

Timing-context separate cold comparisons improved aggregate timing throughput for every song, but strict per-window timing checks failed on `timing/sequential/seq0` for each song:

| song | timing aggregate | failed timing window | model-time regression |
| --- | ---: | --- | ---: |
| Lambada | `34.545 -> 81.423 tok/s` | `timing/sequential/seq0` | `1.304346s -> 2.224136s` |
| PEGASUS | `39.841 -> 86.529 tok/s` | `timing/sequential/seq0` | `1.294285s -> 2.261970s` |
| Ela ke Leitada | `41.487 -> 84.907 tok/s` | `timing/sequential/seq0` | `1.411791s -> 2.477952s` |
| SALVALAI | `39.412 -> 82.870 tok/s` | `timing/sequential/seq0` | `1.369217s -> 2.375507s` |
| Nube Negra | `40.527 -> 76.361 tok/s` | `timing/sequential/seq0` | `1.109477s -> 2.340041s` |

This is a scoped timing first-record regression, not a main-generation TPS regression. Total timing+map stage wall still improved substantially for every separate song.

## GPU Profiling Artifacts

Each of the four jobs wrote 1s `nvidia-smi` telemetry and a diagnostic Nsight Systems report with `profile_nvtx_generation_ranges=true`.

| case | samples | avg GPU util | avg active GPU util | max GPU util | max mem | avg power | max temp | telemetry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `before_together` | `2,950` | `41.1%` | `48.9%` | `67%` | `2294 MiB` | `101.9 W` | `63 C` | `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/before_together/gpu_telemetry.csv` |
| `after_together` | `923` | `60.7%` | `72.1%` | `84%` | `2348 MiB` | `156.6 W` | `71 C` | `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/after_together/gpu_telemetry.csv` |
| `before_separate` | `1,129` | `43.6%` | `53.0%` | `66%` | `2266 MiB` | `123.1 W` | `84 C` | `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/before_separate/gpu_telemetry.csv` |
| `after_separate` | `477` | `46.8%` | `68.5%` | `84%` | `2312 MiB` | `122.7 W` | `60 C` | `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/after_separate/gpu_telemetry.csv` |

Nsight reports:

- `before_together`: `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/before_together/nsight/before_together.nsys-rep`
- `after_together`: `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/after_together/nsight/after_together.nsys-rep`
- `before_separate`: `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/before_separate/nsight/before_separate.nsys-rep`
- `after_separate`: `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/after_separate/nsight/after_separate.nsys-rep`

The telemetry direction matches the runtime result: the after path drives the RTX 2080 Ti harder while finishing much sooner. Detailed kernel summary export was not used for the TPS claim; keep the `.nsys-rep` files for timeline/kernel inspection if the next question is why these five-song runs stabilize near `195 tok/s` instead of materially above `200 tok/s`.

## Decision

Keep the current after stack as the fastest exact default-off opt-in path for realistic single-song and serial multi-song use:

- `inference_generation_compile=true`
- `inference_active_prefix_decode_loop=true`
- `inference_active_prefix_decode_bucket_size=64`
- `inference_active_prefix_decode_cuda_graph=true`
- `inference_active_prefix_decode_cuda_graph_warmup=0`
- `inference_active_prefix_decode_cuda_graph_min_decode_steps=1`
- `inference_stateful_monotonic_logits_processor=true`
- `inference_q1_bmm_cross_attention=true`

Do not relabel this as true concurrent batching. This benchmark validates cold separate-process and long-lived serial multi-song behavior. True batch inference still needs separate mode-specific profiling and gates.
