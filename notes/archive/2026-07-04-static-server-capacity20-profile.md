# Static Server Capacity-20 Profile

## Purpose

Track B static IPC server throughput characterization. This is throughput-only
batch/server evidence, not exact single-song TPS and not same-calculation
evidence, because static server runs still use shared global server RNG.

## Job

- DCC job: `49269905`
- Node/GPU: `dcc-core-gpu-ferc-s-h36-5`, RTX 2080 Ti
  `GPU-5db878e3-a34e-853b-e5be-357ca7d5862a`
- Commit: `b4039b0`
- Branch: `codex/batching-server-throughput-track`
- Run root:
  `/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0`
- Song list:
  `/work/imt11/Mapperatorinator/data/five-song-profile/songs.yaml`
- Config: `profile_salvalai_smoke15`, five songs, `repeats=4`,
  `max_workers=20`, fp32, SDPA, `server_batch_timeout=0.2`,
  `inference_generation_compile=false`, `use_server=true`, `parallel=false`

Preceding job `49269896` failed before profiling because runtime-keyed socket
paths were too long for AF_UNIX. That was fixed by hash-shortening overlong
socket names; no throughput result came from `49269896`.

## Results

| Mode | Main tokens | Scheduler wall | Scheduler main tok/s | Request p95 | Request max | Main queue total | Main unique batch sizes | Strict static compare |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static max_batch=10 | 31,420 | 198.706s | 158.123 | 191.490s | 198.354s | 1410.768s | `10:18, 9:2, 1:2` | baseline |
| static max_batch=20 | 29,858 | 199.888s | 149.374 | 199.500s | 199.569s | 899.027s | `20:2, 19:4, 17:1, 16:1, 13:1, 9:1, 7:1, 4:2, 2:2, 1:10` | FAIL |
| serial multi-song denominator | 23,820 | 346.984s model | 68.649 model tok/s | n/a | n/a | n/a | n/a | serial exact repeat PASS after per-song baselines |

`max_batch_size=20` versus `10`:

- scheduler main TPS: `158.123 -> 149.374` (`-5.5%`);
- scheduler wall: `198.706s -> 199.888s` (`+0.6%`, worse);
- main generated tokens: `31,420 -> 29,858` (`-1,562`);
- attributed request model tok/s: `21.811 -> 16.121`;
- strict comparator exit: `1`;
- both manifests self-validated successfully.

Combined telemetry for the job:

- GPU utilization: avg `60.19%`, max `100%`;
- memory used: avg `3265.96 MiB`, max `9334 MiB`;
- power draw: avg `155.62 W`, max `292.28 W`.

Artifacts:

- max10 manifest:
  `/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0/maxbatch10/static-server-batch-maxbatch10-49269905/static_server_batch_manifest.json`
- max20 manifest:
  `/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0/maxbatch20/static-server-batch-maxbatch20-49269905/static_server_batch_manifest.json`
- strict compare:
  `/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0/compare-rerun.json`
- serial manifest:
  `/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0/serial20/serial_multi_song-serial20-49269905/suite_manifest.json`
- telemetry:
  `/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0/nvidia-smi.csv`

## Interpretation

Do not promote `max_batch_size=20`. It reduced queue wait but made the model
side slower enough that scheduler throughput regressed. It also pushed memory
close to the 2080 Ti limit and still produced many tail/singleton main batches.

For static IPC throughput, `max_batch_size=10` is the better tested capacity
point so far. On this 20-request smoke it reached `158.123` scheduler-wall main
tok/s versus the serial denominator's `68.649` model tok/s, but this remains
throughput-only evidence under shared server RNG.

Next batching work should not keep sweeping larger static batch sizes. The
remaining interesting server work is scheduler/runtime structure: reduce tail
batches, add per-request RNG/equivalence protocol, or build a real continuous
batching prototype with explicit cache/logits/RNG state ledgers.
