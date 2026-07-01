# Warm Repeat Suite Harness

## Summary

Added `utils/profile_inference_suite.py` to measure same-process inference behavior without changing default inference.

The first use case is `warm_repeat`: load the model once, run the same config multiple times, reset RNG before each run, write one profile JSON per run, and write a `suite_manifest.json` that separates first-run cold/specialization cost from warmed subsequent runs.

The second use case is `serial_multi_song`: load the model once, run an explicit song list in one process, reset RNG before each song/repeat, and compare token identity per song rather than across different songs.

## Why

Active-prefix validation showed strong order and warm-state sensitivity. That means a candidate can be unattractive as a cold single-song baseline but still matter for future long-lived batch or multi-song serving. The old single-profile schema did not make that distinction explicit enough.

## Guardrails

- Requires `profile_inference=true`.
- Requires `profile_record_token_ids=true`.
- Requires `use_server=false` until server reseeding is explicit.
- Resets RNG before each run with `accelerate.set_seed`.
- Labels suite metadata with `suite_id`, `run_kind`, `suite_run_index`, `run_index`, `repeat_index`, `song_index`, `song_id`, `suite_song_count`, `suite_repeat_count`, `rng_reset_policy`, and `warmup_excluded`.
- Reports token equivalence against run 0 for `warm_repeat` and against each song's first repeat for `serial_multi_song`.
- Requires `--song-list` for `serial_multi_song`; the default gate expects at least 5 songs, with `--allow-short-suite` reserved for harness smoke tests.

## Non-Claim

Warmed suite throughput is not a cold single-song speedup. It can guide future runtime, batching, and serving work, but retained cold single-song changes still need the normal full-song SALVALAI token-equivalence and no-regression gates.

## First Intended DCC Runs

Run paired three-repeat suites on RTX 2080 Ti:

```bash
python utils/profile_inference_suite.py \
  --config-name profile_salvalai \
  --repeats 3 \
  --run-kind warm_repeat \
  --output-root "$RUN_DIR/compile" \
  inference_active_prefix_decode_loop=false

python utils/profile_inference_suite.py \
  --config-name profile_salvalai \
  --repeats 3 \
  --run-kind warm_repeat \
  --output-root "$RUN_DIR/active512" \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512
```

Compare run 0 separately from runs 1..2. If active-prefix only wins warmed runs, document it as `warm_repeat` evidence and keep it default-off for cold single-song profiling.

## Smoke15 Result

DCC job `49154124` validated the harness on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`, commit `d20f26a`.

- Run dir: `/work/imt11/Mapperatorinator/runs/warm-repeat-smoke15-49154124-d20f26a`
- Logs: `/work/imt11/Mapperatorinator/logs/warm-suite-smoke15-49154124.out` and `.err`
- Status: `COMPLETED`, exit code `0:0`, elapsed `00:04:01`
- Config: `profile_salvalai_smoke15`, `audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3`, `use_server=false`, `attn_implementation=sdpa`, `profile_record_token_ids=true`, `seed=12345`

Main-generation results:

| suite | run | tokens | model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | ---: | --- |
| compile-only | 0 cold | 1,084 | 22.245s | 48.730 | baseline |
| compile-only | 1 warmed | 1,084 | 10.729s | 101.033 | PASS vs run 0 |
| compile-only | 2 warmed | 1,084 | 10.728s | 101.046 | PASS vs run 0 |
| active512 | 0 cold | 1,084 | 30.022s | 36.107 | baseline |
| active512 | 1 warmed | 1,084 | 7.333s | 147.822 | PASS vs run 0 |
| active512 | 2 warmed | 1,084 | 7.286s | 148.777 | PASS vs run 0 |

Cross-suite token equivalence also passed for compile-only vs active512 on all three matching runs. `utils/summarize_inference_profile.py --compare` reported:

| run | compile tok/s | active512 tok/s | delta | token equivalence |
| ---: | ---: | ---: | ---: | --- |
| 0 cold | 48.730 | 36.107 | -25.9% | PASS, 1,084 / 1,084 |
| 1 warmed | 101.033 | 147.822 | +46.3% | PASS, 1,084 / 1,084 |
| 2 warmed | 101.046 | 148.777 | +47.2% | PASS, 1,084 / 1,084 |

Interpretation: active-prefix bucket512 remains a bad cold smoke default, but it is a strong exact warmed-repeat signal. This supports keeping the path default-off while prioritizing first-window/specialization and graph/runtime-stability work. It also justifies a full-song `warm_repeat` run if future batch or long-lived serving throughput becomes the active question.

## Full-Song Result

DCC job `49154643` validated the same warm-repeat question on the full SALVALAI profile on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`, commit `b394a9d`.

- Run dir: `/work/imt11/Mapperatorinator/runs/warm-repeat-full-49154643-b394a9d`
- Logs: `/work/imt11/Mapperatorinator/logs/warm-suite-full-49154643.out` and `.err`
- Status: `COMPLETED`, exit code `0:0`, elapsed `00:10:48`
- Compile-only warmed aggregate: `15,278` tokens, `165.692s`, `92.207 tok/s`
- Active512 warmed aggregate: `15,278` tokens, `118.816s`, `128.585 tok/s`
- Paired cross-suite token equivalence: PASS for runs 0, 1, and 2 (`7,639 / 7,639` main tokens each)

Decision: this clears the threshold for more default-off active-prefix runtime-discipline work for warm-repeat/future batch-serving. It does not replace the retained cold single-song baseline because the active suite ran after a compile suite in the same Slurm job context, and previous cold-first active-prefix validation remained order-sensitive.
