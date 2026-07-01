# Full-Song Warm-Repeat Active-Prefix Validation

## Question

Does bucketed active-prefix decode remain useful when measured as a full-song same-process warmed workload, even though it did not graduate as the retained cold single-song baseline?

## Run

- Job: `49154643`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`
- Stack: Python `3.10.12`, torch `2.10.0+cu128`, Transformers `4.57.3`
- Commit: `b394a9d20c8601c458c628bb32249d88e04db3ba`
- Config: `profile_salvalai`, `audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3`, SDPA, `use_server=false`, `profile_record_token_ids=true`, seed `12345`
- Run dir: `/work/imt11/Mapperatorinator/runs/warm-repeat-full-49154643-b394a9d`
- Logs: `/work/imt11/Mapperatorinator/logs/warm-suite-full-49154643.out`, `/work/imt11/Mapperatorinator/logs/warm-suite-full-49154643.err`
- Slurm status: `COMPLETED`, exit `0:0`, elapsed `00:10:48`

## Results

| suite | run | main tokens | model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | ---: | --- |
| compile-only | 0 | 7,639 | 90.579s | 84.335 | baseline |
| compile-only | 1 | 7,639 | 81.738s | 93.457 | PASS vs compile run 0 |
| compile-only | 2 | 7,639 | 83.953s | 90.991 | PASS vs compile run 0 |
| active512 | 0 | 7,639 | 79.824s | 95.698 | PASS vs compile run 0 |
| active512 | 1 | 7,639 | 58.845s | 129.815 | PASS vs compile run 1 |
| active512 | 2 | 7,639 | 59.971s | 127.379 | PASS vs compile run 2 |

Warmed aggregate:

| suite | warmed tokens | warmed model time | warmed tok/s |
| --- | ---: | ---: | ---: |
| compile-only | 15,278 | 165.692s | 92.207 |
| active512 | 15,278 | 118.816s | 128.585 |

Active512 warmed full-song main generation was `+39.5%` over warmed compile-only with exact paired generated-token equivalence.

## Decision

Keep active-prefix as a default-off strategic candidate for warm-repeat, future multi-song, and graph/runtime work. Do not promote it as the retained cold single-song baseline.

The full-song warmed result clears the explorer threshold for another runtime-discipline experiment: active512 warmed full-song is more than `25%` faster than warmed compile-only and exact. The remaining blocker is cold/order sensitivity, not token correctness.

## Next Action

The next active-prefix experiment should target first-window and graph/specialization churn, not sampling policy or another attention backend toggle. Keep prefill unchanged, keep active-prefix decode-only, bucket at `512` first, and require the normal one-token gate, 15s smoke equivalence, and full-song no-regression checks before any cold speed claim.

## Order-Flipped Follow-Up

Job `49155778` repeated the full-song warm-repeat test with active512 first and isolated TorchInductor/CUDA caches per suite.

- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`
- Stack: Python `3.10.12`, torch `2.10.0+cu128`, Transformers `4.57.3`
- Commit: `19e74a9ceb297f7c859071279f17680d5b04af8b`
- Run dir: `/work/imt11/Mapperatorinator/runs/warm-repeat-active-first-49155778-19e74a9`
- Logs: `/work/imt11/Mapperatorinator/logs/warm-active-first-49155778.out`, `/work/imt11/Mapperatorinator/logs/warm-active-first-49155778.err`
- Slurm status: `COMPLETED`, exit `0:0`, elapsed `00:10:56`

| suite | run | main tokens | model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | ---: | --- |
| active512 | 0 | 7,639 | 80.206s | 95.242 | PASS vs compile run 0 |
| active512 | 1 | 7,639 | 59.031s | 129.406 | PASS vs compile run 1 |
| active512 | 2 | 7,639 | 60.304s | 126.676 | PASS vs compile run 2 |
| compile-only | 0 | 7,639 | 91.136s | 83.820 | baseline |
| compile-only | 1 | 7,639 | 81.836s | 93.345 | PASS vs compile run 0 |
| compile-only | 2 | 7,639 | 84.262s | 90.658 | PASS vs compile run 0 |

Warmed aggregate:

| suite | warmed tokens | warmed model time | warmed tok/s |
| --- | ---: | ---: | ---: |
| active512 | 15,278 | 119.335s | 128.026 |
| compile-only | 15,278 | 166.098s | 91.982 |

Active512 warmed full-song main generation was `+39.2%` over warmed compile-only with exact paired generated-token equivalence. Timing generation also improved (`98.324` vs `73.690 tok/s` warmed), and generated-stage wall time was lower on every paired run.

Decision: active-prefix remains default-off for cold single-song profiling, but the warm-repeat signal is now robust enough to prioritize it for future batch/multi-song and long-lived-process runtime work.
