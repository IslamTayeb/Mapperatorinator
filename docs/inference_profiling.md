# Inference Profiling

Profiling is opt-in. Normal inference does not write profile artifacts unless `profile_inference=true`.

## SALVALAI profile run

Run this on a GPU host, not on a local MacBook:

```bash
python inference.py --config-name profile_salvalai \
  audio_path="/path/to/SALVALAI [Music Video] (_VmnBu3IZ9s).mp3" \
  output_path="/path/to/profile-output"
```

The config targets osu standard, 6 stars, 2015 style, and stream-focused descriptors:

- `skillset/streams`
- `streams/flow aim`
- `streams/spaced streams`
- `streams/bursts`

Each generated beatmap gets a sibling profile JSON by default:

```text
beatmap<uuid>.osu.profile.json
```

Summarize the result:

```bash
python utils/summarize_inference_profile.py /path/to/beatmap.osu.profile.json
```

## DCC GPU Run Notes

Use persistent project/env storage under `/hpc/group/...` and write audio, caches, logs, runs, and generated artifacts under `/work/...`. The conda env `bin` directory must be on `PATH`, otherwise `pydub` may not find `ffmpeg`/`ffprobe` even when they are installed in the env.

For DCC Slurm jobs, keep the Hugging Face cache path consistent between warmup and measured runs:

```bash
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
WORK=/work/imt11/Mapperatorinator
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

python inference.py --config-name profile_salvalai \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/profile-full-${SLURM_JOB_ID}" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  use_server=false
```

`configs/inference/profile_salvalai.yaml` pins `seed=12345` and `profile_record_token_ids=true` so full-song accepted runs can be compared for fixed-token equivalence just like smoke runs.

Validated on DCC `gpu-common` with `--gres=gpu:2080:1`; the allocated node exposed an `NVIDIA GeForce RTX 2080 Ti`.

Observed full SALVALAI run on 2026-06-30:

- Slurm elapsed: 3m20s for a 156.992s MP3.
- Sequence count: 87.
- Map generation: 137.324s stage wall, 8,816 generated tokens, 64.6 tok/s.
- Timing generation: 21.094s stage wall, 821 generated tokens, 39.5 tok/s.
- Model load: 10.152s main model, 8.489s timing model from cache.
- Audio load and postprocessing were sub-second to low-subsecond and not the bottleneck.

The dominant bottleneck is autoregressive map generation. The first map window was the slowest record in this run at 7.655s for 520 generated tokens; most other slow windows were around 2.0-2.4s.

## Same-Calculation Optimization Goal

The current RTX 2080/2080 Ti retained cold single-song baseline is SDPA plus `inference_generation_compile=true`, with active-prefix disabled: full-song job `49113713` generated `7,639` SALVALAI main tokens in `82.615s` synchronized model time, `92.465 tok/s`, with fixed-seed token equivalence PASS against the compile-disabled baseline. Use these targets:

- Starter success target: `100 tok/s` on a cold full-song run.
- Strong first milestone: `120 tok/s` on a cold full-song run.
- Stretch target: `150+ tok/s`.
- Active long-range research target: `200 tok/s`.

The first 100 tok/s loop ended below target by the documented stop condition because profiling showed no remaining plausible current-architecture quick win of `>=10%`. A renewed runtime pass found exact active-prefix evidence, but follow-up validation showed it is order/warm-state sensitive and not the retained cold single-song baseline. Reaching `200 tok/s` from the retained `92.465 tok/s` baseline requires cutting full-song synchronized main-generation model time from `82.615s` to about `38.195s`, a `53.8%` reduction.

Only count a speedup as equivalent when fixed-seed generated token IDs match the baseline for the same audio/config slice. Do not claim wins from changed precision, sampling policy, output policy, model quality, windowing/overlap, or generated-token behavior unless the run is explicitly labeled non-equivalent.

Always verify that performance has not degraded before accepting a change. For full-song candidates, compare main generation, timing generation, total profiled stage time, generated-token counts, token equivalence, and per-window main-generation timings against the current retained baseline. If any meaningful same-config metric regresses, document it as a scoped regression and do not promote it without explicit approval.

Keep SDPA as the current baseline unless profiler evidence strongly contradicts it. Torch profiler wall time is diagnostic only; compare normal `profile_inference` model elapsed time and token throughput.

Keep future batch and multiple-song results in separate reporting classes:

- `cold_single_song`: current acceptance gate. Fresh process/job, one song, fixed seed, full token equivalence, main/timing/total/per-window non-regression.
- `warm_repeat`: same loaded model process, same song repeated. Report the first run separately from runs 2..N; compare warmed results only to warmed results.
- `serial_multi_song` or `batched_multi_request`: operational throughput. Report per-song token equivalence, first-song cold cost, warmed subsequent-song cost, aggregate wall/model throughput, batch-size distribution, RNG reset policy, and any batching/windowing behavior changes.

Do not use warmed-cache, multi-song, or server-batch wins as cold single-song speedups. They can still matter for future batch inference and the planned autoregressive encoder-decoder component, but label them by result class.

Use the same-process suite harness for warm-repeat and future serial multi-song scouting:

```bash
python utils/profile_inference_suite.py \
  --config-name profile_salvalai \
  --repeats 3 \
  --run-kind warm_repeat \
  --output-root "$RUN_DIR" \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512
```

For 5+ song operational scouting, use `serial_multi_song` with an explicit song list:

```bash
python utils/profile_inference_suite.py \
  --config-name profile_salvalai \
  --repeats 2 \
  --run-kind serial_multi_song \
  --song-list "$RUN_DIR/songs.yaml" \
  --output-root "$RUN_DIR" \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512
```

`serial_multi_song` requires at least 5 songs by default because 5+ songs is the expected operational workload. Use `--allow-short-suite` only for harness smoke tests that are not performance evidence.

The song list can be YAML/JSON with a top-level `songs` list, or a plain text file with one audio path per line. YAML/JSON entries can be strings or mappings. Mapping entries support per-song `song_id`/`id`, `audio_path`, `beatmap_path`, `seed`, `start_time`, `end_time`, and `output_subdir`:

```yaml
songs:
  - song_id: salvalai
    audio_path: /work/imt11/Mapperatorinator/data/salvalai.mp3
  - song_id: second-song
    audio_path: /work/imt11/Mapperatorinator/data/second.mp3
    beatmap_path: /work/imt11/Mapperatorinator/data/second.osu
    seed: 12345
    start_time: null
    end_time: null
```

The harness loads the model once, resets RNG before every `generate()` call, writes one profile JSON per run, and writes `suite_manifest.json` with first-run, warmed-run, aggregate throughput, per-song throughput, profile paths, token hashes, and token-equivalence status. `warm_repeat` compares token IDs against run 0. `serial_multi_song` compares token IDs against each song's first repeat, so every song has its own equivalence baseline. Its warmed and multi-song results are useful operational evidence, but they are not cold single-song acceptance evidence.

First smoke result, DCC job `49154124` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, commit `d20f26a`:

| suite | cold run | warmed runs | cross-suite equivalence | interpretation |
| --- | ---: | ---: | --- | --- |
| compile-only | `48.730 tok/s` | `101.040 tok/s` aggregate | baseline | retained path warms substantially |
| active512 | `36.107 tok/s` | `148.298 tok/s` aggregate | PASS vs compile-only for runs 0, 1, and 2 | exact warmed-repeat win, cold regression |

Run dir: `/work/imt11/Mapperatorinator/runs/warm-repeat-smoke15-49154124-d20f26a`. This is `warm_repeat` evidence only. It strengthens the case for fixing active-prefix first-window/specialization cost and for future batch/serving work, but it does not replace the cold single-song baseline.

Full-song warm-repeat validation, DCC job `49154643` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`, commit `b394a9d`:

| suite | run 0 | warmed runs | cross-suite equivalence | interpretation |
| --- | ---: | ---: | --- | --- |
| compile-only | `84.335 tok/s` | `92.207 tok/s` aggregate | baseline | retained path warms to roughly the accepted cold baseline |
| active512 | `95.698 tok/s` | `128.585 tok/s` aggregate | PASS vs compile-only for runs 0, 1, and 2 | exact warmed full-song win, still warm/batch evidence only |

Run dir: `/work/imt11/Mapperatorinator/runs/warm-repeat-full-49154643-b394a9d`. Active512 improved warmed full-song main generation by `+39.5%` over warmed compile-only (`128.585` vs `92.207 tok/s`) with `7,639 / 7,639` generated main tokens matching on each paired run. The same job also showed active512 run 0 at `95.698 tok/s`, but this is not a cold single-song baseline replacement because the active suite ran after a compile suite in the same process/job context and earlier cold-first active-prefix validation remained order-sensitive. Keep active-prefix default-off and use this only as `warm_repeat`/future batch-serving evidence.

Order-flipped full-song warm-repeat validation, DCC job `49155778` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`, commit `19e74a9`:

| suite | run 0 | warmed runs | paired equivalence | interpretation |
| --- | ---: | ---: | --- | --- |
| active512 | `95.242 tok/s` | `128.026 tok/s` aggregate | PASS vs compile-only for runs 0, 1, and 2 | exact warmed full-song win, measured active-first |
| compile-only | `83.820 tok/s` | `91.982 tok/s` aggregate | baseline | retained path, isolated cache after active512 |

Run dir: `/work/imt11/Mapperatorinator/runs/warm-repeat-active-first-49155778-19e74a9`. This repeated the full-song test with active512 first and separate TorchInductor/CUDA caches for each suite. Active512 improved warmed main generation by `+39.2%` (`128.026` vs `91.982 tok/s`), improved warmed timing generation (`98.324` vs `73.690 tok/s`), and reduced paired main/timing stage wall time on every run. Generated main-token IDs matched compile-only for all three paired runs (`7,639 / 7,639`). This strengthens active-prefix as a warm-repeat, long-lived process, and future batch-serving candidate, but it still does not replace the cold single-song baseline.

Targeted cold-overhead attribution, DCC jobs `49156700`, `49156749`, and `49157290` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, commit `daa1828`:

| run | main tok/s | seq0 | seq1+ | token equivalence | interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| cold compile-only | `84.608` | `16.413s`, `35.703 tok/s` | `95.473 tok/s` | baseline | retained path in this matrix was slower than clean baseline |
| cold active512 | `96.247` | `25.531s`, `22.953 tok/s` | `131.003 tok/s` | PASS vs cold compile | cold tax is front-loaded |
| warm active512 run 1 | `127.877` | `3.809s`, `153.829 tok/s` | `126.109 tok/s` | PASS vs run 0 and cold compile | warmed first window is fixed |
| warm active512 run 2 | `126.610` | `3.871s`, `151.393 tok/s` | `124.911 tok/s` | PASS vs run 0 and cold compile | warmed signal repeats |

Run dirs:

- Full-song matrix: `/work/imt11/Mapperatorinator/runs/active-prefix-cold-matrix-49156700-daa1828`
- Torch logs, Nsight, and torch-profiler diagnostics: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-49156749-daa1828`
- Nsight post-processing: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-49156749-daa1828/nsys_active512_smoke15`

Conclusion: active-prefix is weak or unstable cold because the first long generation window pays extra graph/capture/specialization cost. It is not weak in steady state: the same cold active512 run is already `131.003 tok/s` after `seq0`, and warmed repeats reduce `seq0` from about `25-26s` to about `3.8s`. Torch logs support this attribution: diagnostic job `49156749` recorded `244` recompilation lines, `3,441` graph-recording lines, `8,665` CUDA graph/cudagraph log matches, and a `cache_utils.py:update` `recompile_limit` warning keyed on `layer_idx`. The focused post-warmup torch-profiler trace is diagnostic only; it still showed graph/runtime overhead alongside kernels (`20,504` TorchDynamo cache lookups, `16,310` `cudaGraphLaunch` calls, `131` `CUDAGraphNode.record` events, and `5,628` FMHA kernels with `1.063s` self CUDA). Nsight Systems exported `ap512.nsys-rep` and `ap512.sqlite`, but the smoke15 trace generated only one token for early map windows and had heavy diagnostic overhead, so use it for qualitative NVTX availability rather than throughput or full-song seq0 timing.

Next implementation target: graph/runtime stabilization for active-prefix bucket512. Good candidates are explicit graph/cache priming with setup cost honestly included, reducing cache-update specialization by decoder layer, or a bufferized/direct decode loop that avoids repeated HF compiled graph variants. Do not hide prewarm cost outside cold single-song timing; cold claims must include setup/timing/main/total non-regression. For warm-repeat or future batch serving, report first-song cold cost and warmed amortized throughput separately.

## Smoke-To-Full Profiling Loop

Start with the middle 15s SALVALAI smoke config for fastest iteration:

```bash
python inference.py --config-name profile_salvalai_smoke15 \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/profile-smoke15-${SLURM_JOB_ID}" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  use_server=false
```

`configs/inference/profile_salvalai_smoke15.yaml` sets `start_time=71000`, `end_time=86000`, `seed=12345`, and `profile_record_token_ids=true`, inheriting the retained compile-only cold single-song path from `profile_salvalai`. Use this for first-pass scouting; promote to the 30s smoke or full-song configs only when the 15s result is token-equivalent and plausibly meaningful. For compile/runtime changes with one-time specialization costs, inspect post-warmup windows before rejecting a token-equivalent candidate, but full-song non-regression is still required before acceptance.

Reference 15s smoke profiles from commit `ce92ebb` on RTX 2080 Ti node `dcc-core-ferc-s-z25-21`:

| run | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| compile-disabled | `49132862` | `/work/imt11/Mapperatorinator/runs/smoke15-nocompile-49132862-ce92ebb/beatmap3ccc5112a05940db8fdd747994c4bef2.osu.profile.json` | 1,084 | 22.068s | 49.1 |
| retained compile baseline | `49132861` | `/work/imt11/Mapperatorinator/runs/smoke15-compile-49132861-ce92ebb/beatmapfb8360ef206b41f4a6cc1fe7a6a735ed.osu.profile.json` | 1,084 | 13.335s | 81.3 |

`utils/summarize_inference_profile.py --compare` reported token equivalence PASS for all `1,084` generated main-token IDs. This short slice is warmup-sensitive: the retained compile baseline is `+65.5%` over compile-disabled overall, while post-warmup map windows are mostly around `94-105 tok/s`.

Use the middle 30s SALVALAI smoke config when a longer smoke slice is needed before full-song promotion:

```bash
python inference.py --config-name profile_salvalai_smoke \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/profile-smoke-${SLURM_JOB_ID}" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  use_server=false
```

`configs/inference/profile_salvalai_smoke.yaml` sets `start_time=63500`, `end_time=93500`, `seed=12345`, and `profile_record_token_ids=true`. This keeps scouting runs short while allowing token-equivalence checks.

Compare baseline and candidate smoke profiles:

```bash
python utils/summarize_inference_profile.py \
  --compare /path/to/baseline.profile.json /path/to/candidate.profile.json
```

The comparer now prints a same-calculation metadata contract before token comparison. A candidate is not promotable if it differs from the baseline in model, audio, seed, precision, sampling policy, windowing, context/output policy, server/parallel mode, or token-recording setup unless the run is explicitly labeled non-equivalent.

Before any custom decoder-step, CUDA graph, backend, or kernel path is treated as a candidate, run the one-token logits gate on the same 15s smoke config:

```bash
python utils/verify_one_token_decode.py \
  --config-name profile_salvalai_smoke15 \
  --report-path "$WORK/runs/one-token-decode-${SLURM_JOB_ID}.json" \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/profile-smoke15-${SLURM_JOB_ID}" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1
```

This gate compares a direct static-cache `q_len=1` decode step against captured raw logits from HF cached
`generate(max_new_tokens=2)` on the real SALVALAI prompt-building path. The capture processor clones logits before
Mapperatorinator logits processors can mutate them in-place, and HF's first generated token becomes the probe token for
the direct candidate step. The old no-cache full-prefix comparison is kept only as a diagnostic. Passing this gate is
necessary for decoder-runtime work, but not sufficient for a speed claim; still require fixed-seed 15s generated-token
equivalence and full-song equivalence.

The reusable one-token ABI lives in `osuT5.osuT5.inference.direct_decode`. DCC job `49139917` validated the ABI on
`main_generation.seq9` after extraction:

- Compile disabled: `/work/imt11/Mapperatorinator/runs/one-token-abi-gate-49139917-8cb1160/one_token_decode_seq9_compile_false.json`, PASS, `max_abs=0.0`, candidate prepared shape `[1, 1]`.
- Compile enabled: `/work/imt11/Mapperatorinator/runs/one-token-abi-gate-49139917-8cb1160/one_token_decode_seq9_compile_true.json`, PASS, `max_abs=2.2888e-05`, top-k match, candidate prepared shape `[1, 1]`.

Use this helper as the starting point for manual CUDA graph or narrow `torch.compile(mode="reduce-overhead")`
experiments. Do not maintain a second direct-step path unless this ABI is proven wrong or insufficient.

For post-warmup diagnosis, use detailed internal ranges only in torch-profiler or Nsight runs:

```bash
python inference.py --config-name profile_salvalai_smoke15 \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/profile-smoke15-trace-${SLURM_JOB_ID}" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  use_server=false \
  profile_torch_generation=true \
  profile_torch_generation_limit=1 \
  profile_torch_generation_label_filter=main_generation \
  profile_generation_detail_ranges=true
```

`profile_generation_detail_ranges=true` adds NVTX and `record_function` ranges inside VarWhisper decoder self-attention, cross-attention, MLP `fc1`/activation/`fc2`, final norm, and output projection. These traces are diagnostic only; do not use traced wall time for throughput claims.

For Nsight Systems timeline work, add `profile_nvtx_generation_ranges=true` to emit top-level `generation.<label>.seqN` NVTX ranges without enabling `torch.profiler`. This makes cold `main_generation.seq0` and warmed windows such as `main_generation.seq9` visible in the timeline while keeping torch-profiler overhead out of the run. The resulting Nsight wall times are still diagnostic; use normal `profile_inference` model timings for throughput claims.

For SDPA dispatch audits, force one PyTorch SDPA backend at a time:

```bash
python inference.py --config-name profile_salvalai_smoke15 \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/profile-smoke15-sdpa-flash-${SLURM_JOB_ID}" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  use_server=false \
  profile_sdpa_backend=flash
```

Valid backend values are `flash`, `efficient`, `math`, and `cudnn` when the installed PyTorch exposes them. Record PyTorch warnings, actual profiler kernel names, token equivalence, and untraced speed. A forced backend micro-result does not replace the retained SDPA baseline without full-song token-equivalent evidence.

RTX 2080 Ti SDPA backend audit result:

- Smoke job `49139404` found forced `profile_sdpa_backend=math` was token-equivalent and much faster on the 15s aggregate (`78.862 tok/s` vs `50.850 tok/s`), while forced `efficient` was noise and forced `flash` failed on SM75 with no available kernel.
- The full-song paired validation job `49139420` rejected forced `math` for main generation: `/work/imt11/Mapperatorinator/runs/full-sdpa-math-49139420-9d92c34/beatmap8241a2b3f0be495592a5cf2b9bb9d6a2.osu.profile.json` produced `7,639` main tokens in `85.177s`, `89.684 tok/s`, token equivalence PASS, which was slower than the retained compile-only `92.465 tok/s` baseline.
- Conclusion: keep default SDPA plus `inference_generation_compile=true` as the baseline. Do not promote forced `math` from smoke-only evidence; use it only as a diagnostic if a future trace shows a backend-specific reason.

TorchDynamo cache-limit scout result:

- Compile-health job `49139485` showed CUDA graph recordings across smoke map windows and recompiles from
  `decoder_input_ids` stride changes plus timing-model to map-model `proj_out.weight` size changes.
- Raising `torch._dynamo.config.recompile_limit` and `torch._dynamo.config.cache_size_limit` to `32` in job `49139564`
  was token-equivalent but not a main-generation win: `50.850 -> 50.867 tok/s` on the 15s smoke. Do not promote this
  as an optimization; keep it as evidence that the missing gain is not solved by a larger Dynamo cache.

Direct-step CUDA graph and attention ceiling results:

- Direct-step graph POC job `49139948` used `utils/profile_direct_decode_step.py` on `main_generation.seq9`. It was
  logits-exact and measured fixed one-token eager forward at `12.18-12.36ms` vs manual CUDA graph replay at about
  `8.09ms`, `1.5x` faster. This is a diagnostic ceiling only: `8.09ms` is about `124 tok/s` before real sampling,
  token-copy, EOS, and loop control work, so graphing the current forward alone is not a `200 tok/s` path.
- SM75 fp32 SDPA microprofile job `49139966` wrote
  `/work/imt11/Mapperatorinator/runs/attn-sm75-fp32-49139966-596f221/attention_kernel_profile.json`.
  q_len=1 `decode_self_kv512` repo-like SDPA was `0.0686ms`; `decode_self_kv2048` was `0.2884ms`; and
  `cross_attn_kv2048` was `0.2892ms`.
- Focused mask-length job `49139981` wrote `/work/imt11/Mapperatorinator/runs/sdpa-mask-2560-49139981.json`.
  q_len=1 `kv2560` was `0.3576ms` maskless and `0.3739ms` with a static position-84 mask. The mask overhead is small;
  the full static self-attention length is the large cost.
- Interpretation: the next plausible `200 tok/s` lever is exact active-prefix self-attention/cache layout that avoids
  full `max_target_positions` work without the rejected per-token slicing path. Manual CUDA graphing remains useful for
  a likely `120 tok/s`-class runtime path but needs an attention/cache-layout win to approach `200 tok/s`.

Promote a change to a full-song SALVALAI run only when smoke results are stable, token IDs match, and the speedup is plausibly meaningful. For compile-like changes with one-time first-window costs, inspect post-warmup per-window throughput before rejecting a weak total smoke result. Keep changes that improve RTX 2080 full-song main-generation throughput by about 10% or more. Keep 5-10% wins only when they are simple and well-contained or strategically de-risk the custom runtime path. Remove 1-3% complexity by default.

For the current 200 tok/s phase, stop the long-running optimization loop only when either the full-song RTX 2080/2080 Ti run reaches at least `200 tok/s` with identical fixed-seed tokens, or profiling across multiple exact-calculation optimization families shows no remaining plausible major exact-calculation path.

Current status after active-prefix validation:

- Retained cold single-song baseline: SDPA plus `inference_generation_compile=true`, active-prefix disabled. Full-song job `49113713` generated `7,639` main tokens in `82.615s`, `92.465 tok/s`, token equivalence PASS against the compile-disabled baseline. Same-commit paired job `49150185` measured compile-only at `92.998 tok/s`.
- Current retained flags: `attn_implementation=sdpa`, `inference_generation_compile=true`, `inference_active_prefix_decode_loop=false`, `seed=12345`, `profile_record_token_ids=true`.
- Active-prefix is an opt-in strategic candidate, not the retained cold baseline. Bucket `256` in job `49150185` looked like a large exact win (`92.998 -> 121.926 tok/s`), but later full-song runs showed order/warm-state sensitivity. Job `49151748` measured bucket `256` cold at `88.699 tok/s`, bucket `128` at `81.396 tok/s`, and bucket `512` warm at `108.313 tok/s`, all token-equivalent against bucket `256`.
- Bucket `512` cold-first validation job `49152465` generated `7,639` main tokens in `78.108s`, `97.800 tok/s`, token equivalence PASS against a same-job compile-only run. That is only about `+5.8%` against the retained `92.465 tok/s` compile-only baseline and it still regressed the first main-generation window and timing/total stage against clean retained runs, so it stays default-off.
- Reaching `200 tok/s` from the retained `92.465 tok/s` baseline requires reducing main-generation model time to `38.195s`, so about `44.420s` or `53.8%` of retained model time remains to remove.

For the 200 tok/s phase, rank the next experiments this way:

1. Preflight exactness and measurement infrastructure: metadata contract checks, one-token raw-logits gate, detailed decoder-step ranges, and SDPA backend dispatch audits.
2. Exact custom decode-loop prototype with CUDA graph discipline, gated to local batch-one static-cache generation first.
3. Attention/cache backend spike only if detailed ranges show `q_len=1` self-attention or cross-attention is target-sized after PyTorch graph work.
4. Linear/GEMV launch-reduction work only if detailed ranges show MLP/projection launches dominate after attention/cache work.
5. Fused sampling/logits-processor work only if sampling grows to at least `10%` of synchronized main-generation time.
6. Torch-TensorRT / TensorRT-RTX feasibility only after a toy FP32 graph proves real engine execution with fallback disabled.

Custom runtime work must prove token identity in stages before any speed claim graduates: compile-disabled 15s smoke equivalence, compile-enabled 15s smoke equivalence, then full-song equivalence.

Torch-TensorRT/TensorRT environment probe:

- Job `49134026` on `dcc-core-ferc-s-z25-21`, RTX 2080 Ti capability `(7, 5)`, commit `a2cf83a`, log `/work/imt11/Mapperatorinator/logs/trt-probe-49134026.out`.
- DCC env `/hpc/group/romerolab/imt11/envs/mapperatorinator` has `torch 2.10.0+cu128` with CUDA `12.8`, but `torch_tensorrt`, `tensorrt`, `onnx`, `onnxruntime`, and `polygraphy` are not installed on the GPU node.
- Treat TensorRT as a separate environment-install feasibility project, not the immediate runnable optimization path. Do not spend runtime-spike time on TensorRT until the stack exists in an isolated env and can compile the repeated one-token decoder forward with logits/token equivalence checks.

Post-warmup torch-profiler diagnostic for the 15s retained baseline:

- Job `49133341`, profile `/work/imt11/Mapperatorinator/runs/smoke15-trace-seq9-49133341-ce92ebb/beatmap574d4d0acdf84cef8c7efd2758fb74bb.osu.profile.json`.
- Traced label: `generation.main_generation.seq9`, `234` generated tokens.
- Normal synchronized model time for the traced record was `2.562s` (`91.3 tok/s`); torch-profiler/export inflated outer wall to `24.207s`, so use the trace only for event mix.
- Top CUDA self-time events were the compiled forward region and f32 memory-efficient attention on SM75: `Torch-Compiled Region: 0/2` `1.717s`, `fmha_cutlassF_f32_aligned_64x64_rf_sm75` `1.476s`.
- Sampling/logits overhead was visible but not target-sized in this trace: `aten::sort` `234` calls, `7.601ms` CUDA total; `aten::_softmax` `468` calls, `1.759ms` self CUDA; `aten::cat` `542` calls, `1.433ms` self CUDA. Do not prioritize a fused sampler unless a future trace shows it has grown to at least `10%` of synchronized main-generation time.

## Codex Goal Prompt

```text
Optimize Mapperatorinator inference on RTX 2080/2080 Ti for same-calculation speedups only. Current baseline is roughly 65-78 tok/s; first success target is 100 tok/s main-generation throughput, strong milestone is 120 tok/s, and 150+ tok/s is stretch. Do not claim speedups from changed precision, sampling policy, output policy, model quality, windowing/overlap, or generated-token behavior unless explicitly labeled non-equivalent.

Use a profiling-first loop. Start with a middle-30-second song smoke slice, prove fixed-seed generated tokens match baseline, and only promote promising changes to full-song SALVALAI runs. Use full-song runs for accepted results. Keep SDPA as the baseline unless profiler evidence strongly contradicts it. Separate true model time from torch.profiler overhead.

Scout improvement ideas with subagents and web research as useful, but accept only measured wins. Prioritize big wins in generation-loop structure, cache behavior, mask construction, logits processors, repeated small kernels, memory movement, and avoidable per-token setup. Keep changes that improve RTX 2080 full-song main-generation throughput by >=10%; keep 5-10% only if very simple; remove 1-3% complexity. Commit and push clean checkpoints for accepted wins, document why each win worked or failed in docs/inference_profiling.md, update AGENTS.md with durable conventions, write notes under notes/, and stop when the 100 tok/s goal is reached or profiling shows no remaining plausible >=10% exact-calculation improvement.
```

## 200 tok/s Goal Prompt

```text
Optimize Mapperatorinator inference toward 200 tok/s main-generation throughput on RTX 2080/2080 Ti, same-calculation only. Current retained baseline is SDPA + inference_generation_compile=true with active-prefix disabled: 7,639 full-song SALVALAI main tokens, 82.615s synchronized model time, 92.465 tok/s, fixed-seed token equivalence PASS against the compile-disabled full-song baseline.

Use a PyTorch-first, bounded opt-in runtime/kernel plan. Start with measurement and exactness infrastructure: preflight equivalence checks, one-token decoder logits equivalence against the current raw-logits path, 15s middle-song SALVALAI smoke token equivalence, and untraced profile_inference model-time throughput. Treat torch profiler and Nsight traces as diagnostic only.

Prioritize changes that could reduce real one-token decoder forward cost: SDPA backend dispatch audit, static one-token decode ABI, CUDA graph discipline, q_len=1 self-attention and cross-attention cache/layout work, and only then isolated FlashInfer or narrow native CUDA/CUTLASS kernels for measured dominant hotspots. Keep SDPA + generation compile as the baseline unless full-song token-equivalent evidence justifies replacement.

Do not claim wins from changed precision, sampling policy, output policy, model quality, windowing/overlap, generated-token behavior, output length, or non-equivalent RNG behavior unless explicitly labeled non-equivalent. Do not restart rejected quick tweaks unless new profiling evidence explains why old negative or non-equivalent results no longer apply.

Use configs/inference/profile_salvalai_smoke15.yaml for first-pass scouting with seed=12345, use_server=false, attn_implementation=sdpa, inference_generation_compile=true, inference_active_prefix_decode_loop=false, and profile_record_token_ids=true. Promote only after one-token logits gate PASS and 15s fixed-seed generated-token equivalence PASS. Full-song SALVALAI token equivalence, untraced throughput, and no performance degradation in timing generation or total profiled stage time are required for accepted cold single-song results.

Keep full-song RTX 2080/2080 Ti wins >=10%; keep 5-10% only if simple or strategically unlocks the runtime path; revert 1-3% complexity by default. Commit and push clean checkpoints for accepted wins, document accepted and rejected experiments in docs/inference_profiling.md and notes/, update AGENTS.md with durable conventions, and stop only when 200 tok/s is reached or profiling shows no remaining plausible exact-calculation path to a major gain.

Improvements that help future batch inference or multiple-song throughput matter too, but report them separately from cold single-song speedups: first-song cold cost, warmed subsequent-song cost, aggregate multi-song throughput, token equivalence per song, RNG reset policy, and any batching/windowing behavior changes.
```

## Attention Kernel Profiling

Use `utils/profile_attention_kernels.py` to isolate SDPA vs FlashAttention 2 without loading model weights or audio. The script uses fixed random tensors with VarWhisper small v3-like shapes (`6` heads, `64` head dimension) and reports both raw attention-call timing and a repo-like layout path that includes output reshaping and FlashAttention packing overhead.

Run on a GPU Slurm allocation:

```bash
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
WORK=/work/imt11/Mapperatorinator
REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator
LOCAL_TMP="/tmp/imt11-attn-kernels-${SLURM_JOB_ID}"

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/targets/x86_64-linux/lib:$ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$LOCAL_TMP"
export TEMP="$LOCAL_TMP"
export TMP="$LOCAL_TMP"
export TRITON_CACHE_DIR="$LOCAL_TMP/triton"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOCAL_TMP" "$WORK/runs" "$WORK/logs"
cd "$REPO"

python utils/profile_attention_kernels.py \
  --output-dir "$WORK/runs/attn-kernels-${SLURM_JOB_ID}" \
  --dtype fp16 \
  --warmup 50 \
  --iters 200 \
  --repeats 3 \
  --profile-iters 20
```

Outputs:

- `attention_kernel_profile.json`: metadata, per-case timings, tensor shapes, CUDA memory peaks, and top profiler events.
- `attention_kernel_profile.txt`: readable timing table, FA2/SDPA ratios, and profiler tables.
- `*.trace.json`: Chrome trace files for representative decode, cross-attention, prefill, and batch cases.

The default cases cover single-token cached decode, cross-attention against encoder-like lengths up to `2048`, prefill-like self-attention up to `2048`, and a few batch-size-8 what-if cases. Keep this as a microprofile: it does not replace full inference profiling because it does not include model GEMMs, cache updates, tokenizer/server overhead, or sampling.

Observed A5000 kernel microprofile on 2026-06-30:

- Job: `49097689` on `dcc-rental-gpu-07`, `NVIDIA RTX A5000`.
- Artifacts: `/work/imt11/Mapperatorinator/runs/attn-kernels-49097689/attention_kernel_profile.json`, matching `.txt`, and 20 Chrome traces.
- Runtime: `torch==2.10.0+cu128`, `flash-attn==2.8.3.post1`, CUDA runtime `12.8`.

Representative timings:

| case | mode | SDPA | FA2 | FA2/SDPA |
| --- | --- | ---: | ---: | ---: |
| `decode_self_kv512` | attention only | 0.0407ms | 0.1229ms | 3.02x |
| `decode_self_kv512` | repo-like layout | 0.0465ms | 0.1695ms | 3.65x |
| `cross_attn_kv2048` | attention only | 0.0391ms | 0.1237ms | 3.16x |
| `cross_attn_kv2048` | repo-like layout | 0.0453ms | 0.1668ms | 3.69x |
| `prefill_self_len512` | attention only | 0.0389ms | 0.1270ms | 3.26x |
| `prefill_self_len2048` | attention only | 0.1088ms | 0.1131ms | 1.04x |
| `prefill_self_len2048` | repo-like layout | 0.1167ms | 0.2098ms | 1.80x |
| `batch8_cross_attn_kv2048` | attention only | 0.0500ms | 0.1245ms | 2.49x |
| `batch8_cross_attn_kv2048` | repo-like layout | 0.0501ms | 0.1973ms | 3.94x |

Kernel traces showed SDPA dispatching to `aten::_scaled_dot_product_flash_attention` and `pytorch_flash::flash_fwd*` kernels, so SDPA is already using fused flash-style kernels on A5000. FA2 used `flash_attn::_flash_attn_forward` and `flash::flash_fwd*` kernels. For long prefill, raw attention kernel time was close; for single-token decode and cross-attention, FA2 lost mostly to wrapper/launch gaps and repo-like packing overhead. In repo-like FA2 traces, `aten::cat`/`CatArrayBatchedCopy` was a visible extra cost, especially for batch-size what-if cases.

## Full Generation Torch Traces

For full-model kernel traces around actual `model.generate` calls, enable the opt-in torch profiler:

```bash
python inference.py --config-name profile_salvalai \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/profile-torch-${SLURM_JOB_ID}" \
  device=cuda \
  precision=fp16 \
  attn_implementation=sdpa \
  use_server=false \
  profile_torch_generation=true \
  profile_torch_output_dir="$WORK/runs/profile-torch-${SLURM_JOB_ID}/torch_profiles" \
  profile_torch_generation_limit=3 \
  profile_torch_generation_label_filter=main_generation
```

This writes Chrome traces for the first selected `model.generate` calls and adds a `torch_profiles` section to the normal `.profile.json`. Use `profile_torch_generation_label_filter=main_generation` to skip timing-context windows and trace map generation directly, or leave it unset to trace the first generation calls of any label. Keep `profile_torch_generation_limit` small for full songs; each trace includes CPU/CUDA activities, shapes, memory events, NVTX/record-function ranges, and the top profiler events by self CUDA time.

Validated on DCC job `49098455` with a short SALVALAI slice, `precision=fp16`, `attn_implementation=sdpa`, and `profile_torch_generation_label_filter=main_generation`. It produced:

- Profile JSON: `/work/imt11/Mapperatorinator/runs/profile-main-trace-49098455/beatmap9c39eb98ff514d8da459a90cd1238ca3.osu.profile.json`
- Chrome trace: `/work/imt11/Mapperatorinator/runs/profile-main-trace-49098455/torch_profiles/000_generation_main_generation_seq0.trace.json`

The trace file was about `1.33 GB`; the traced first map window took `211s` under `torch.profiler`, while the remaining untraced map windows returned to normal speed. Use this mode for selected windows only. For smoke tests, either trace timing-context windows or lower `train.data.tgt_seq_len`; for bottleneck work, prefer one representative `main_generation` window and inspect the `torch_profiles[0].events` summary before opening the full Chrome trace.

## FlashAttention 2 On DCC

FlashAttention 2 is the relevant package for A5000/5000 Ada testing through the normal Transformers `flash_attention_2` path. FlashAttention 3 is Hopper-focused and should be treated as a separate H100/H800-class ablation, not as a replacement for A5000 runs.

The DCC env at `/hpc/group/romerolab/imt11/envs/mapperatorinator` has been validated with `flash-attn==2.8.3.post1` on an `NVIDIA RTX A5000`:

- `torch==2.10.0+cu128`
- CUDA runtime: 12.8
- GPU capability: `(8, 6)`
- `transformers.utils.is_flash_attn_2_available()` returned `True` on the A5000 allocation.
- A tiny `flash_attn_func` fp16 call returned finite output.

Build source wheels on node-local temp storage. A `/work`-backed build reached 73/73 CUDA compile steps but failed during wheel packaging with an `egg-info` directory cleanup error. The successful build used:

```bash
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
export PATH="$ENV/bin:$PATH"
export CUDA_HOME="$ENV"
export CUDA_PATH="$ENV"
export TMPDIR="/tmp/imt11-flash-attn-${SLURM_JOB_ID}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=2
export LD_LIBRARY_PATH="$ENV/targets/x86_64-linux/lib:$ENV/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$ENV/targets/x86_64-linux/include:$ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/include:${CPATH:-}"

python -m pip install ninja packaging "wheel<0.46" "setuptools<81"
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation --no-cache-dir -v
```

Do not use a login-node `is_flash_attn_2_available()` result as the final check; it can return `False` because CUDA is unavailable even when `import flash_attn` works. Validate inside a GPU Slurm allocation.

For FlashAttention profiling runs, keep Triton temp and cache directories on node-local storage too:

```bash
export TMPDIR="/tmp/imt11-profile-fa2-${SLURM_JOB_ID}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton"
```

A `/work`-backed `TMPDIR` hit a Triton temporary-directory cleanup failure before map generation.

Observed A5000 SALVALAI ablation on 2026-06-30 with `precision=fp16`, `seed=12345`, full-song input, and `use_server=false`:

| attention | main generation | map tokens | map tok/s | timing generation | timing tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sdpa` | 110.956s | 6,992 | 63.4 | 29.376s | 28.3 |
| `flash_attention_2` | 187.007s | 8,554 | 45.9 | 31.656s | 26.2 |

FlashAttention 2 did not improve this single-song profile. It generated more map tokens despite the same seed, so raw wall time is partly output-length-dependent, but normalized map throughput was still lower than SDPA.

## Accepted Exact-Calculation Optimizations

### Transformers generation compile

Accepted in commit `3e9033c` after smoke and full-song profiling. The change adds `inference_generation_compile`, which leaves the global default disabled but allows profiling/long inference runs to opt into the Transformers generation compile path by setting `model.generation_config.disable_compile = False`.

RTX 2080 Ti full-song comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `3e9033c` | `49113712` | `/work/imt11/Mapperatorinator/runs/full-base-49113712-3e9033c/beatmap9024531ea69844218e3c15e53ad2972c.osu.profile.json` | 7,639 | 121.410s | 62.9 |
| candidate | `3e9033c` | `49113713` | `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json` | 7,639 | 82.615s | 92.5 |

`utils/summarize_inference_profile.py --compare` reported token equivalence PASS for all `7,639` generated main-generation token IDs. Throughput improved by `+47.0%`; synchronized model time dropped by `38.795s` (`-32.0%`). Smoke profiling was only `+1.3%` overall because the first compiled main-generation window paid a one-time compile cost, but post-warmup smoke windows ran around `94-96 tok/s` compared with `68-69 tok/s` baseline. This win should carry to future Mapperatorinator-like autoregressive encoder-decoder cores because it improves the repeated single-token decode loop rather than beatmap-specific output logic.

Keep `inference_generation_compile=true` for full-song profiling baselines. For very short one-off inference, the global default remains `false` so callers can choose whether the compile warmup cost is worthwhile.

### Bucketed active-prefix decode loop

Implemented in commit `821fb41` as an opt-in batch-1 decode loop, but not retained as the cold single-song baseline after follow-up validation. The global default and `profile_salvalai` retained config keep active-prefix disabled.

Why it is still useful: the loop leaves prefill unchanged, then applies active-prefix self-attention only during one-token decode. One-token gates showed decode-only active-prefix can preserve logits, while active-prefix prefill is non-equivalent. Bucketed active-prefix lengths preserve causal-mask equivalence while avoiding the full static-cache self-attention length for most decode steps.

The first full-song validation looked very strong but was later shown to be order/warm-state sensitive:

| job | run | main tokens | main model time | tok/s | token equivalence | status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `49150185` | compile-only baseline | 7,639 | 82.142s | 92.998 | baseline | clean compile-only comparison |
| `49150185` | active-prefix bucket 256 | 7,639 | 62.653s | 121.926 | PASS, 7,639 / 7,639 | over-promoted; later validation did not reproduce as cold baseline |
| `49151748` | active-prefix bucket 256, first in bucket sweep | 7,639 | 86.122s | 88.699 | PASS vs bucket 128 | rejected for cold baseline |
| `49151748` | active-prefix bucket 512, third in bucket sweep | 7,639 | 70.527s | 108.313 | PASS vs bucket 256 | warm/order-sensitive candidate |
| `49152465` | active-prefix bucket 512, cold-first validation | 7,639 | 78.108s | 97.800 | PASS vs same-job compile-only | strategic opt-in candidate, not retained |

Job `49152465` compared bucket `512` against a same-job compile-only run and passed token equivalence. However, the same-job compile-only baseline was anomalously slow (`84.984 tok/s`) compared with the retained clean compile-only baseline (`92.465-92.998 tok/s`). Against the retained clean baseline, bucket `512` is only about a `5-6%` main-generation gain, and it still regressed first-window behavior and timing/total stage time versus clean retained runs. This does not meet the normal cold single-song graduation bar.

Keep this path scoped to the validated simple generation mode: batch size `1`, `use_server=false`, `parallel=false`, `cfg_scale=1.0`, `num_beams=1`, static cache, SDPA, and no active-prefix during prefill. Any broader scope, bucket-size change, warmed-repeat claim, or batch/multi-song claim requires new one-token logits gates, 15s token-equivalent smoke, full-song token equivalence, and result-class-specific non-regression checks.

Post-active-prefix attribution job `49150687` traced the bucket-256 path with detailed ranges and `TORCH_LOGS=recompiles,cudagraphs`. It is diagnostic only, because torch profiler/logging inflated the traced `seq9` wall time to `107.836s`. The useful evidence is that sampling/logits processors remained small, while graph/runtime churn was large: `1,058` graph-recording log lines, `20,504` TorchDynamo cache lookups, `16,310` CUDA graph launches, and a visible `sdpa_attention_forward` recompile from full static key length `2560` to active length `1024`. See `notes/2026-07-01-post-active-prefix-attribution.md`.

## Rejected Exact-Calculation Experiments

### Stateful monotonic time-shift masking

Attempted in commit `9d7e5b7` and reverted after smoke profiling. The change replaced the per-token full-prefix scan in `MonotonicTimeShiftLogitsProcessor` with a stateful batch-size-1 path that tracks the last time-shift token after the last SOS token.

RTX 2080 Ti smoke comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `01c18d6` | `49109301` | `/work/imt11/Mapperatorinator/runs/smoke-base-49109301-01c18d6/beatmap6f980906005d441fb87edde94f269b83.osu.profile.json` | 2,894 | 41.707s | 69.4 |
| candidate | `9d7e5b7` | `49109743` | `/work/imt11/Mapperatorinator/runs/smoke-cand-49109743-9d7e5b7/beatmapd81b370ad0ac422cb1b5a01b3d3a093d.osu.profile.json` | 2,894 | 43.487s | 66.5 |

`utils/summarize_inference_profile.py --compare` reported token equivalence PASS for all `2,894` generated main-generation token IDs, but throughput was `-4.1%` worse. The likely reason is that removing `torch.isin`/full-prefix work also added per-token state, mask, slice, `masked_fill`, and `torch.where` work; this did not pay back in normal generation. Do not reintroduce this shape of stateful logits processor without profiler evidence that the replacement removes more work than it adds.

### Last-position generation logits

Attempted in commits `1011c1d` and `786a5fd` and reverted after smoke profiling. The change passed `logits_to_keep=1` during generation for VarWhisper so the LM head projected only the last decoder position instead of all positions.

RTX 2080 Ti smoke comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `01c18d6` | `49109301` | `/work/imt11/Mapperatorinator/runs/smoke-base-49109301-01c18d6/beatmap6f980906005d441fb87edde94f269b83.osu.profile.json` | 2,894 | 41.707s | 69.4 |
| candidate | `786a5fd` | `49110976` | `/work/imt11/Mapperatorinator/runs/smoke-logits2-49110976-786a5fd/beatmapac03a230f58f4de5a23bf66b2074c844.osu.profile.json` | 2,894 | 53.091s | 54.5 |

`utils/summarize_inference_profile.py --compare` reported token equivalence PASS for all `2,894` generated main-generation token IDs, but throughput was `-21.4%` worse. The likely reason is that the smaller LM-head projection shape loses the efficient larger GEMM path or causes less favorable per-token kernel behavior, despite doing less nominal arithmetic. Do not reintroduce final-logit-only projection for VarWhisper generation without profiler evidence that the GEMM/kernel path has changed.

### Static-cache SDPA prefix trim

Attempted in commit `aac2c4b` and reverted after smoke profiling. The change reduced the static-cache SDPA decode attention length from `max_target_positions` to the current decoder mask length by building a shorter 4D mask and slicing self-attention K/V tensors to that valid prefix.

RTX 2080 Ti smoke comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `01c18d6` | `49109301` | `/work/imt11/Mapperatorinator/runs/smoke-base-49109301-01c18d6/beatmap6f980906005d441fb87edde94f269b83.osu.profile.json` | 2,894 | 41.707s | 69.4 |
| candidate | `aac2c4b` | `49112400` | `/work/imt11/Mapperatorinator/runs/smoke-prefix-49112400-aac2c4b/beatmapdd5d12b22215462a973be40cd9143954.osu.profile.json` | 2,894 | 43.398s | 66.7 |

`utils/summarize_inference_profile.py --compare` reported token equivalence PASS for all `2,894` generated main-generation token IDs, but throughput was `-3.9%` worse. The likely reason is that the shorter attention shape saved masked attention work but added slicing and less favorable SDPA/kernel dispatch behavior. Do not reintroduce this exact static-cache K/V prefix trim unless a new trace shows max-cache attention is still dominant and the replacement avoids per-token slicing overhead.

### Persistent static causal-mask buffer

Attempted in commit `2d807c1` and reverted after 15s smoke profiling. The change added a default-off `inference_persistent_static_mask` path that reused a full-size 4D static-cache causal mask buffer for the guarded SDPA, batch-size-1, one-token decode path. Unlike the rejected prefix trim, it kept the full static-cache attention shape and tried to stabilize the mask data pointer for compiled decode.

RTX 2080 Ti middle-15s SALVALAI comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `2d807c1` | `49137140` | `/work/imt11/Mapperatorinator/runs/smoke15-mask-base-2d807c1/beatmap4438946a6a864548b8d968d4a3a24682.osu.profile.json` | 1,084 | 21.093s | 51.4 |
| candidate | `2d807c1` | `49137141` | `/work/imt11/Mapperatorinator/runs/smoke15-mask-cand-2d807c1/beatmap63be4f635c7147cd9b127038054db1d2.osu.profile.json` | 1,084 | 25.590s | 42.4 |

Token equivalence PASS: all `1,084` generated main-generation token IDs matched, and profile records showed `persistent_static_mask_enabled=true` for the candidate. Throughput was `-17.6%` worse overall and post-warmup seq9 fell from `104.6 tok/s` to `91.5 tok/s`, so the regression was not just first-window compile noise. The likely reason is that stable buffer reuse still added per-token `fill_`, boolean mask construction, and `masked_fill_` work, while the existing mask path is already efficient enough relative to the compiled forward. Do not reintroduce persistent mask mutation unless a future trace shows mask construction itself is target-sized and the replacement avoids extra per-token kernels.

### Dynamic/default cache generation

Attempted in commit `52b8871` and reverted after smoke profiling. The change added an `inference_static_cache=false` path that skipped the preallocated `StaticCache` and let generation use the default/dynamic cache behavior while keeping `inference_generation_compile=true`.

RTX 2080 Ti smoke comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `3e9033c` | `49113275` | `/work/imt11/Mapperatorinator/runs/smoke-compile-49113275-3e9033c/beatmapa00e6b0a1f394a05a1bc95ab28a1dac4.osu.profile.json` | 2,894 | 41.190s | 70.3 |
| candidate | `52b8871` | `49114897` | `/work/imt11/Mapperatorinator/runs/smoke-dyncache-49114897-52b8871/beatmap36600487b37a414b9f239d9a8f6e9586.osu.profile.json` | 1,633 | 23.522s | 69.4 |

`utils/summarize_inference_profile.py --compare` reported token equivalence FAIL: baseline generated `2,894` main-generation token IDs, candidate generated `1,633`, and the first mismatch was at token `0`. This is not an equivalent speed result. Do not use dynamic/default-cache generation for accepted same-calculation claims unless a future implementation proves fixed-seed token equivalence first.

### Static generation compile config

Attempted in commit `d6ce772` and reverted after smoke profiling. The change kept `inference_generation_compile=true` but forced `model.generation_config.compile_config = CompileConfig(dynamic=False, mode="reduce-overhead")`.

RTX 2080 Ti smoke comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `3e9033c` | `49113275` | `/work/imt11/Mapperatorinator/runs/smoke-compile-49113275-3e9033c/beatmapa00e6b0a1f394a05a1bc95ab28a1dac4.osu.profile.json` | 2,894 | 41.190s | 70.3 |
| candidate | `d6ce772` | `49116934` | `/work/imt11/Mapperatorinator/runs/smoke-compilecfg-49116934-d6ce772/beatmap9c4f9667e17a4de8a00b5e3a86400669.osu.profile.json` | 2,894 | 53.836s | 53.8 |

`utils/summarize_inference_profile.py --compare` reported token equivalence PASS for all `2,894` generated main-generation token IDs, but throughput was `-23.5%` worse. Slurm stderr showed TorchDynamo hit its recompile limit in `modeling_mapperatorinator.py:139`; the last reason was a `decoder_input_ids` stride mismatch, expected `24` but actual `25`. This likely defeated the useful stable decode-loop compile behavior. Do not force `CompileConfig(dynamic=False)` for generation unless a future Transformers/PyTorch version fixes the recompile pattern and a fresh smoke run proves a win.

### Dynamic generation compile config

Attempted in commit `96aecec` and reverted after smoke profiling. The change kept `inference_generation_compile=true` but forced `model.generation_config.compile_config = CompileConfig(dynamic=True, mode="reduce-overhead")`.

RTX 2080 Ti smoke comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `3e9033c` | `49113275` | `/work/imt11/Mapperatorinator/runs/smoke-compile-49113275-3e9033c/beatmapa00e6b0a1f394a05a1bc95ab28a1dac4.osu.profile.json` | 2,894 | 41.190s | 70.3 |
| candidate | `96aecec` | `49118947` | `/work/imt11/Mapperatorinator/runs/smoke-dyncfg-49118947-96aecec/beatmapff0b384d836c47d0b502e73f79a16e26.osu.profile.json` | 2,894 | 45.772s | 63.2 |

`utils/summarize_inference_profile.py --compare` reported token equivalence PASS for all `2,894` generated main-generation token IDs, but throughput was `-10.0%` worse. Slurm stderr showed repeated cudagraph partition messages and a large first-window setup cost. Do not force `CompileConfig(dynamic=True)` for this workload unless a future PyTorch/Transformers version changes the compile behavior and a fresh smoke run proves a win.

### Max-autotune generation compile config

Attempted in commit `0cccf36` and reverted after 15s smoke profiling. The change added a default-off `inference_generation_compile_mode` scout flag and tested `model.generation_config.compile_config = CompileConfig(mode="max-autotune")` while leaving `dynamic=None`, `fullgraph=False`, and the retained SDPA/static-cache generation path unchanged.

RTX 2080 Ti middle-15s SALVALAI comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `0cccf36` | `49136379` | `/work/imt11/Mapperatorinator/runs/smoke15-compile-default-0cccf36/beatmap1572085922b24c95b60ecf1f5abe5ec4.osu.profile.json` | 1,084 | 21.422s | 50.6 |
| candidate | `0cccf36` | `49136380` | `/work/imt11/Mapperatorinator/runs/smoke15-compile-maxautotune-0cccf36/beatmapf92dca264acf4af3bd5a08b78880c75a.osu.profile.json` | 1,084 | 34.126s | 31.8 |

Token equivalence PASS: all `1,084` generated main-generation token IDs matched. Throughput was `-37.2%` worse overall and post-warmup windows were also slower, for example seq9 fell from `104.1 tok/s` to `88.9 tok/s`. Stderr showed autotuning selected some Triton kernels, but that did not improve this Turing/SM75 single-token decode workload. Do not force `CompileConfig(mode="max-autotune")` for this baseline unless a future PyTorch/Transformers version changes the kernel choices and a fresh smoke run proves a win.

### Fullgraph generation compile config

Attempted in commit `149fe88` and reverted after 15s smoke profiling. The change forced `model.generation_config.compile_config = CompileConfig(fullgraph=True, mode="reduce-overhead")` while keeping the retained SDPA/static-cache generation path.

RTX 2080 Ti middle-15s SALVALAI comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `2d807c1` | `49137140` | `/work/imt11/Mapperatorinator/runs/smoke15-mask-base-2d807c1/beatmap4438946a6a864548b8d968d4a3a24682.osu.profile.json` | 1,084 | 21.093s | 51.4 |
| candidate | `149fe88` | `49137638` | `/work/imt11/Mapperatorinator/runs/smoke15-fullgraph-149fe88/beatmapa2f8469fd07e475aaa4d7f68477dd5ca.osu.profile.json` | 1,084 | 21.258s | 51.0 |

Token equivalence PASS: all `1,084` generated main-generation token IDs matched. The candidate was only `-0.8%` overall and post-warmup windows were mixed, with no plausible `>=10%` signal. Per the keep/revert policy, do not keep this complexity or force `CompileConfig(fullgraph=True)` unless a future PyTorch/Transformers version changes generation compile behavior and a fresh smoke run shows a meaningful win.

### Preallocated sample loop

Attempted in commits `6d6ba7c` and `f7b5222` and reverted after smoke profiling. The change added an opt-in Mapperatorinator override for Hugging Face `_sample` that preallocated `input_ids` and `decoder_attention_mask` buffers to avoid per-token `torch.cat` in the generation loop while preserving the compiled one-token forward path.

RTX 2080 Ti smoke comparison on DCC `gpu-common`:

| run | commit | job | node | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `f7b5222` | `49121863` | `dcc-core-ferc-s-z25-21` | `/work/imt11/Mapperatorinator/runs/smoke-pairbase-49121863-f7b5222/beatmap93c5d8f5a9504f92af53f1698a535fa8.osu.profile.json` | 2,894 | 31.368s | 92.3 |
| candidate | `f7b5222` | `49121864` | `dcc-core-gpu-ferc-s-h36-6` | `/work/imt11/Mapperatorinator/runs/smoke-prealloc2-49121864-f7b5222/beatmape999b779433b4f8395818898648fd657.osu.profile.json` | 3,000 | 32.367s | 92.7 |

`utils/summarize_inference_profile.py --compare` reported token equivalence FAIL: baseline generated `2,894` main-generation token IDs, candidate generated `3,000`, and the first mismatch was at token `1,350`. Instrumented profile records confirmed the preallocated path actually ran in all `20` candidate main-generation windows. Throughput was only `+0.5%` by token-normalized rate while model time was `+3.2%` worse, so this is both non-equivalent and too small. The candidate also logged one TorchDynamo recompile caused by timing/main model parameter shape mismatch. Do not reintroduce a mutable preallocated `_sample` loop unless a future prototype first proves exact token identity with compile disabled and then with compile enabled.

### Copy-compatible custom sample hook

Attempted in commit `07b36a5` and reverted after 15s smoke profiling. The change added an opt-in `inference_custom_decode_loop` path that temporarily replaced Hugging Face's internal `_sample` method with a local copy-compatible loop while still letting `model.generate()` perform its normal generation setup, logits-processor construction, stopping criteria, static-cache wiring, and generation compile checks.

RTX 2080 Ti middle-15s SALVALAI smoke comparisons on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| gate | run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| compile disabled | baseline | `07b36a5` | `49135145` | `/work/imt11/Mapperatorinator/runs/smoke15-loop-base2-compileoff-07b36a5/beatmapba4408b7bf2e444e9a7c761640053d40.osu.profile.json` | 1,084 | 17.564s | 61.7 |
| compile disabled | candidate | `07b36a5` | `49135146` | `/work/imt11/Mapperatorinator/runs/smoke15-loop-custom2-compileoff-07b36a5/beatmapcba4ff105a834900bb11cb79e288b873.osu.profile.json` | 1,084 | 19.711s | 55.0 |
| compile enabled | baseline | `07b36a5` | `49135380` | `/work/imt11/Mapperatorinator/runs/smoke15-loop-base-compileon-07b36a5/beatmap2dcd0585607145de8553c8d57889c856.osu.profile.json` | 1,084 | 21.390s | 50.7 |
| compile enabled | candidate | `07b36a5` | `49135381` | `/work/imt11/Mapperatorinator/runs/smoke15-loop-custom-compileon-07b36a5/beatmapc7f1c08725d445aaaf73d0e57e40eaa6.osu.profile.json` | 1,084 | 25.079s | 43.2 |

Token equivalence passed in both gates: `1,084 / 1,084` generated main-generation token IDs matched with compile disabled and with compile enabled. The path also recorded `custom_decode_loop_enabled=true`, so the candidate did exercise the local loop. It was still slower by `-10.9%` with compile disabled and `-14.7%` with compile enabled. Post-warmup compile-enabled windows were slower too, for example seq9 fell from `104.8 tok/s` to `93.5 tok/s`, so this was not only first-window compile noise.

Interpretation: a conservative local `_sample` hook is useful evidence that the HF sampling loop can be replaced without changing tokens, but a copy-compatible Python loop by itself adds overhead rather than removing it. Do not keep or retry a copy-only `_sample` monkeypatch as an optimization. Future custom runtime work must attack a real cost center such as CUDA graph capture discipline, stable preallocated buffers that preserve RNG/token identity, or compiled/exported one-token decoder forward execution.

### `torch.inference_mode` generation wrapper

Attempted in commit `02b2437` and reverted after full-song profiling. The change replaced `@torch.no_grad()` with `@torch.inference_mode()` around `model_generate` and `model_forward`.

RTX 2080 Ti full-song comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `3e9033c` | `49113713` | `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json` | 7,639 | 82.615s | 92.5 |
| candidate | `02b2437` | `49115936` | `/work/imt11/Mapperatorinator/runs/full-infermode-49115936-02b2437/beatmapd3e6077d36ea49e795275451d96d09d8.osu.profile.json` | 7,639 | 80.780s | 94.6 |

Smoke profiling looked promising (`+31.9%`), but the accepted full-song comparison was only `+2.3%` with token equivalence PASS for all `7,639` generated main-generation token IDs. This is below the keep threshold, so the change was reverted. Do not reintroduce inference-mode wrapping unless a future full-song result clears the acceptance threshold.

## Active-Prefix Decode Isolation

This section records the diagnostic evidence that led to the opt-in bucketed active-prefix decode loop. The full generated-token loop produced useful exact, post-warmup speed signals, but later validation kept it out of the retained cold single-song baseline; see "Bucketed active-prefix decode loop" above.

Commit `51f189f` added diagnostic switches to isolate active-prefix self-attention during direct-cache prefill vs the one-token decode step. This is not a production inference path and not a throughput claim; it is a correctness and fixed-step ceiling test.

DCC job `49140082`, node `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, commit `51f189f`:

- Run dir: `/work/imt11/Mapperatorinator/runs/active-prefix-isolate-49140082-51f189f`
- Logs: `/work/imt11/Mapperatorinator/logs/active-prefix-isolate-49140082.out` and `.err`
- Config: `profile_salvalai_smoke15`, `seq9`, `precision=fp32`, `attn_implementation=sdpa`, `use_server=false`, `parallel=false`, `cfg_scale=1.0`, `num_beams=1`

Correctness split:

| Variant | Compile | Result |
| --- | --- | --- |
| baseline | false | PASS, `max_abs=0.0`, top-k match |
| active-prefix prefill only | false | FAIL, `max_abs=15.413055`, top-k mismatch |
| active-prefix decode only | false | PASS, `max_abs=0.0`, top-k match |
| active-prefix prefill + decode | false | FAIL, `max_abs=15.413055`, top-k mismatch |
| baseline | true | PASS, `max_abs=2.2888e-05`, top-k match |
| active-prefix prefill only | true | FAIL, `max_abs=15.413059`, top-k mismatch |
| active-prefix decode only | true | PASS, `max_abs=2.2888e-05`, top-k match |
| active-prefix prefill + decode | true | FAIL, `max_abs=15.413059`, top-k mismatch |

Fixed-step graph timing for the passing decode-only variant:

| Compile | Report | Eager ms/step | Graph ms/step | Speedup |
| --- | --- | ---: | ---: | ---: |
| false | `graph_decode_compile_false.json` | `11.4996ms` | `3.7891ms` | `3.035x` |
| true | `graph_decode_compile_true.json` | `11.7837ms` | `3.7899ms` | `3.109x` |

Bucketed active-prefix follow-up job `49140217`, node `dcc-core-ferc-s-z25-21`, RTX 2080 Ti, commit `fb2b2ae`:

- Run dir: `/work/imt11/Mapperatorinator/runs/active-prefix-buckets-49140217-fb2b2ae`
- Logs: `/work/imt11/Mapperatorinator/logs/active-prefix-buckets-49140217.out` and `.err`

All tested graph-reusable decode bucket lengths preserved the one-token logits gate:

| Bucket | Compile | Gate max_abs | Gate top-k | Graph ms/step |
| ---: | --- | ---: | --- | ---: |
| 128 | false | `0.0` | PASS | `3.8920ms` |
| 256 | false | `0.0` | PASS | `4.1022ms` |
| 512 | false | `0.0` | PASS | `4.4178ms` |
| 128 | true | `2.2888e-05` | PASS | `3.7597ms` |
| 256 | true | `2.2888e-05` | PASS | `3.9840ms` |
| 512 | true | `2.2888e-05` | PASS | `4.4166ms` |

Interpretation:

- Do not apply active-prefix self-attention during static-cache prefill; it is not equivalent in the current model path.
- One-token decode-only active-prefix is logits-equivalent for this gate and gives a fixed-step graph ceiling near `264 tok/s` before real loop overhead.
- This is the first measured exact-calculation path with plausible arithmetic for a `200 tok/s` runtime project, but it still needs a real generated-token loop. The current graph POC replays one prepared `[1, 1]` step and does not handle changing tokens, changing prefix lengths, logits processors, sampling/RNG, EOS, or generated-token accounting.
- Bucketed decode is the preferred next strategy over exact-prefix graph captures because buckets preserve exact logits in the tested gate while giving reusable shape families.

Next implementation direction: leave prefill unchanged, then build an opt-in batch-1 direct decode loop using `osuT5.osuT5.inference.direct_decode` with active-prefix applied only around the one-token model forward. Use bucketed active-prefix lengths first, e.g. `ceil(prefix_length / bucket_size) * bucket_size`, because bucketed graph shapes are reusable and already passed the one-token gate.

## Runtime Backend Feasibility Notes

### Torch-TensorRT and TensorRT-RTX

The current DCC Mapperatorinator environment does not include Torch-TensorRT, TensorRT, ONNX, ONNX Runtime, or Polygraphy. The initial GPU-node probe was job `49134026` on `dcc-core-ferc-s-z25-21` with RTX 2080 Ti capability `(7, 5)`, PyTorch `2.10.0+cu128`, and repo commit `a2cf83a`. Its log is `/work/imt11/Mapperatorinator/logs/trt-probe-49134026.out`.

Package dry-runs on 2026-07-01 show that TensorRT work should use a separate isolated env rather than mutating `/hpc/group/romerolab/imt11/envs/mapperatorinator`:

| path | dry-run evidence | key resolved packages | concern |
| --- | --- | --- | --- |
| standard Torch-TensorRT | `/work/imt11/Mapperatorinator/logs/torch_tensorrt_210_unpinned_dryrun_report.json` | `torch_tensorrt==2.10.0`, `tensorrt==10.14.1.48.post1`, `cuda-toolkit==13.3.1`, `nvidia-cuda-runtime==13.3.29` | current DCC driver probe reported CUDA `13.2`, so import/runtime needs GPU-node validation |
| Torch-TensorRT-RTX | `/work/imt11/Mapperatorinator/logs/torch_tensorrt_rtx_dryrun_report.json` | `torch-tensorrt-rtx==2.12.1`, `torch==2.12.1`, `tensorrt_rtx==1.4.0.76`, CUDA `13.0` package family, `executorch==1.3.1` | changes the PyTorch line and pulls a large runtime stack |

Official docs describe Torch-TensorRT-RTX as experimental and installed via `torch-tensorrt-rtx`; the Python import remains `torch_tensorrt`, and NVIDIA's PyTorch walkthrough uses `torch.compile(..., backend="tensorrt")`. NVIDIA's TensorRT-RTX support matrix lists Turing/RTX 2080Ti compute capability `7.5` support, but its TensorRT-RTX 1.5 footnote says Turing does not support FP32 GEMMs in that release. Since the retained Mapperatorinator baseline is same-calculation FP32-style inference on RTX 2080 Ti, do not assume TensorRT-RTX accelerates the dominant decoder GEMMs without direct logits and token-equivalence evidence.

Follow-up isolated TensorRT-RTX validation created `/hpc/group/romerolab/imt11/envs/mapperatorinator-trt-rtx` without modifying the retained env. GPU import passed in job `49138760` on `dcc-core-ferc-s-z25-21`: `torch==2.12.1+cu130`, `torch_tensorrt==2.12.1`, `tensorrt_rtx==1.4.0.76`, RTX 2080 Ti capability `(7, 5)`, driver `595.71.05`.

That import success did not become a TensorRT optimization path. The first toy `torch.compile(..., backend="tensorrt")` smoke skipped lowering because the graph was below `min_block_size=5`. A stricter lowering smoke in job `49138769` tested a larger FP32 MLP and a tiny FP32 MLP with `min_block_size=1`; both produced correct outputs only after TensorRT conversion failed and the backend returned GraphModule/PyTorch fallback. Stderr included `MyelinCheckException: cudnn_graph_utils.h:405: CHECK(false) failed. cuDNN graph compilation failed`, `Unable to create TensorRT execution context`, and `Returning GraphModule forward instead`.

Current status: Torch-TensorRT-RTX is importable in an isolated env on the 2080 Ti, but it is rejected as a current same-calculation FP32 acceleration path because true TensorRT lowering failed before Mapperatorinator model work. Do not proceed to one-token decoder export until a toy graph proves real engine creation. Future TensorRT tests must fail loudly on fallback, for example with `pass_through_build_failures=True` or equivalent stderr/module-type checks.

The next TensorRT gate, if revisited with another package/CUDA/runtime combination, is engine validation before any inference speed claim:

1. Build an isolated DCC env such as `/hpc/group/romerolab/imt11/envs/mapperatorinator-trt`.
2. Run a Slurm GPU import/compile smoke on RTX 2080 Ti that prints driver, CUDA, PyTorch, `torch_tensorrt`, and TensorRT versions.
3. Compile a tiny fixed-shape CUDA module with `torch.compile(..., backend="tensorrt")`.
4. Compile/export only the repeated one-token decoder forward.
5. Compare logits, then run 15s smoke token-equivalence, then full-song token-equivalence before any speed claim can graduate.

See `notes/2026-07-01-tensorrt-packaging-probe.md` for the full resolver details.

## 200 tok/s Quick-Tweak Stop Decision

The first current-architecture 200 tok/s scouting loop stopped on 2026-07-01 by the documented stop condition. This is a historical stop decision for quick-tweak work before the renewed active-prefix runtime pass. The target was not reached, and the measured quick-tweak candidate families no longer showed a plausible remaining major full-song win on RTX 2080/2080 Ti.

The retained baseline at that time was SDPA plus `inference_generation_compile=true`: full-song job `49113713`, `7,639` main-generation tokens, `82.615s` synchronized model time, `92.465 tok/s`, and fixed-seed token equivalence PASS against the compile-disabled full-song baseline. After active-prefix follow-up validation, this remains the retained cold single-song baseline.

The final rejection set includes copy-compatible custom `_sample` hook, preallocated `_sample` loop, persistent static-mask mutation, compile-config variants (`dynamic=False`, `dynamic=True`, `mode="max-autotune"`, `fullgraph=True`), TensorRT-RTX current-env lowering, static-cache prefix trim, dynamic/default cache generation, final-position logits, stateful monotonic masking, and `torch.inference_mode` wrapping. Sampling/logits fusion remains below the profiling threshold in the post-warmup trace, and batching/parallel/server/window changes are non-equivalent unless separately proven token-identical.

Final read-only subagent `019f1c3b-a844-73d1-8e67-858ad3b51732` agreed with stopping: no remaining exact-calculation path was both plausible for a `>=10%` full-song win and worth running before the stop decision.

See `notes/2026-07-01-200tps-stop-decision.md` for the closure summary. Future attempts toward `200 tok/s` should start from the retained compile-only baseline and new profiling evidence, not by rerunning the documented failed scouts.

The renewed runtime/kernel pass later found new evidence in job `49140082`: decode-only active-prefix self-attention passed the one-token logits gate and cut fixed graph replay to about `3.79ms`, while active-prefix prefill failed. That evidence led to the opt-in generated-loop candidate, but full-song follow-up kept it as strategic runtime evidence rather than the retained cold single-song baseline.

## What The Profile Captures

Top-level `stages` report wall time for setup, model loading, audio loading, segmentation, timing generation, main generation, diffusion, postprocessing, and file writes.

The `generation` records report per-window or per-batch details:

- `profile_label`
- `context_type`
- `mode`
- `sequence_index` or `batch_start_index`
- `prompt_wall_seconds`
- `wall_seconds`
- `model_elapsed_seconds`
- `prompt_tokens_per_sample`
- `output_tokens_per_sample`
- `generated_tokens_per_sample`
- `tokens_per_second`
- `sync_model_timing`
- `torch_profiled`
- `torch_trace_index`, `torch_trace_path`, and `torch_trace_wall_seconds` when a torch profiler trace covered that generation window
- CUDA memory counters when CUDA is available

Torch profiler records include a bounded key-average event summary in the profile JSON. Use
`profile_torch_event_limit=<N>` to keep more than the default top 50 events, or `profile_torch_event_limit=0` to keep
all events. The bounded summary is sorted by the largest self/total CUDA or self CPU signal so nested semantic
`record_function` ranges are not dropped only because their self CUDA time is low. This changes only profile artifact
size; it is diagnostic and must not be used as a throughput claim.

Use `wall_seconds - model_elapsed_seconds` to spot server/IPC/queue overhead. Use token counts and `tokens_per_second` to separate model throughput problems from unusually long outputs. When `profile_sync_cuda=true`, profiling requests synchronized model timing inside `model_generate` so `model_elapsed_seconds` includes pending CUDA work. Do not use `wall_seconds` from records with `torch_profiled=true` as normal throughput; the profiler can inflate traced windows by orders of magnitude.

Profile metadata also captures reproducibility context when profiling is enabled: seed, hostname, Slurm job id/partition, git commit/branch, torch/CUDA versions, CUDA device name/capability, and cache/temp environment paths. Report the profile path plus this metadata when accepting or rejecting an optimization.
