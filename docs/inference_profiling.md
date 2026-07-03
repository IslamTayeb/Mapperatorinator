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
- Historical active long-range research target: `200 tok/s`, reached by the default-off exact opt-in stack below.
- Current single-song research target: `500 tok/s`, same-calculation only, not batching or multi-process aggregate throughput.

The first 100 tok/s loop ended below target by the documented stop condition because profiling showed no remaining plausible current-architecture quick win of `>=10%`. A renewed runtime pass found exact active-prefix and CUDA graph evidence, then job `49168188` added a stateful monotonic logits processor on top of that default-off path, job `49204568` reduced active-graph capture warmup to zero, job `49206207` reduced the active graph bucket size, and job `49213490` added q_len=1 BMM cross-attention for the simple fp32 active-graph path. The current fastest accepted exact opt-in path reaches `201.125 tok/s` with fixed-seed token equivalence. Reaching `200 tok/s` from the retained `92.465 tok/s` conservative baseline required cutting full-song synchronized main-generation model time from `82.615s` to about `38.195s`; the accepted q1 BMM opt-in path measured `37.981s`. The conservative cold default remains compile-only SDPA unless the user explicitly opts into the active-prefix graph stack.

### 500 tok/s single-song campaign

The post-200 campaign is explicitly about normal single-song inference speed on RTX 2080/2080 Ti. Do not count batching, multiple processes, multiple songs in parallel, precision changes, quantization, sampling-policy changes, RNG changes, output-policy changes, generated-token changes, or output-length changes as equivalent single-song speedups.

Use the fastest exact opt-in path as the campaign baseline unless a new full-song exact-equivalent single-song run beats it:

```bash
python inference.py --config-name profile_salvalai \
  inference_generation_compile=true \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=64 \
  inference_active_prefix_decode_cuda_graph=true \
  inference_active_prefix_decode_cuda_graph_warmup=0 \
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1 \
  inference_stateful_monotonic_logits_processor=true \
  inference_q1_bmm_cross_attention=true
```

For SALVALAI's `7,639` main tokens, `500 tok/s` implies about `15.278s` synchronized main-generation model time, a further `59.8%` reduction from the accepted q1-BMM path (`37.981s`). Treat this as a runtime/kernel campaign, not a quick-tweak loop. Re-profile the accepted stack first, use untraced `profile_inference` for speed claims, and use Nsight/torch profiler only for diagnosis.

The first campaign checkpoint adds default-off verifier infrastructure, not a speed claim:

- `inference_decode_session_runtime=false`
- `inference_decode_session_cuda_graph=false`
- `inference_decode_session_chunk_size=1`
- `inference_native_decode_kernels=false`
- `utils/verify_one_token_decode.py --candidate-decode-session`

`DecodeSession` lives in `osuT5.osuT5.inference.direct_decode` and currently delegates to the existing exact static-cache one-token helpers. Keep it verifier-first until generated-token identity, raw-logit/top-k equality, final RNG-state identity, and output/map behavior are proven across multiple sampled decode steps and windows.

Normal inference intentionally raises if any reserved DecodeSession/native-kernel flag is enabled before that path is implemented. Use `utils/verify_one_token_decode.py --candidate-decode-session` for the current one-token logits verifier path; do not treat the reserved Hydra flags as production runtime switches yet.

`DecodeSession` also has a multi-step direct-loop verifier path. DCC job `49222209` on RTX 2080 Ti, commit `c440416`, ran `utils/verify_direct_decode_loop.py --candidate-decode-session` for 64 sampled decode steps with active-prefix bucket64, CUDA graph replay, and q1 BMM cross-attention. The gate passed generated-token identity, raw-logit/top-k equality, and final CPU+CUDA RNG-state equality against HF `generate()`. This is verifier infrastructure only, not a throughput claim. See `notes/2026-07-03-decode-session-direct-loop-verifier.md`.

Prefer PyTorch built-ins first, then isolated C++/CUDA/CUTLASS/cuBLASLt kernels for measured dominant hotspots. Avoid Triton-first work unless there is no stable practical alternative and the path is default-off, isolated, exact, and strongly measured. Do not implement a PyTorch q1 BMM branch for active-prefix self-attention, even thresholded to long buckets, unless new full-song evidence changes the ceiling. Jobs `49222759`, `49222781`, and production-prefix sweep `49222883` showed self-attention BMM is slower than SDPA through `L=576`, only mildly faster at `L>=640`, and projects to only about `0.463s` saved on full-song SALVALAI (`201.125 -> ~203.6 tok/s`, `~1.2%`). q1 BMM remains justified for long unmasked cross-attention.

Fresh post-q1 500 tok/s profiling, DCC jobs `49221975` through `49221977` on RTX 2080 Ti, commit `b0ec059`, confirmed the current exact opt-in path remains around `200 tok/s` but did not improve it. Full-song untraced throughput was `200.646 tok/s` for `7,639` main tokens (`38.072s` model time), with main token equivalence PASS against job `49213490`; the retained baseline remains `201.125 tok/s`. Job `49221975` is marked Slurm `FAILED` only because the strict comparison command returned nonzero after detecting no-regression failures; the profile itself is valid. The torch diagnostic for `main_generation.seq9` classified actual kernel self-CUDA, excluding profiling ranges and API calls, as `36.0%` one-token GEMV/GEMM linear work, `32.9%` FMHA attention, `13.9%` elementwise, `10.1%` copy/index/cat, `3.1%` layernorm, and `1.7%` softmax. This supports prioritizing one-token linear launch reduction and q_len=1 active-prefix attention/cache layout before fused sampling. See `notes/2026-07-03-post-q1-500tps-profiling.md`.

The first post-q1 linear microprobe rejected simple PyTorch call-form cleanup. `utils/profile_decode_linear_kernels.py` captures actual one-token decoder linear inputs through the direct decode verifier path and compares equivalent call forms. DCC job `49222623` on RTX 2080 Ti, commit `41a1af0`, captured `73` decoder linears for `profile_salvalai_smoke15` seq9 with active-prefix length `128`, q1 BMM enabled, and logits replay `max_abs=0.0`. The captured one-token linears were `12x` self-attn `768 -> 2304`, `36x` `768 -> 768` projection linears, `12x` MLP `768 -> 3072`, `12x` MLP `3072 -> 768`, and `1x` output projection `768 -> 4069`. Rewriting individual `F.linear` calls to `matmul`, `addmm`, or `mv` was flat to tiny: the best per-signature isolated gains project to only about `0.22s` over full-song SALVALAI. A synthetic `fc1 -> GELU -> fc2` MLP block improved only `1.074x` isolated, about `1%` projected full-song. Do not pursue individual linears one by one; future linear work must be fused/native decoder-block or cuBLASLt/CUDA work with verifier gates. See `notes/2026-07-03-decode-linear-kernel-probe.md`.

The first post-q1 attention component probe rejected a PyTorch q1 BMM branch for active-prefix self-attention. `utils/profile_decode_attention_components.py` captures actual one-token decoder self/cross attention tensors through the direct decode verifier path and compares SDPA against explicit q1 `bmm -> softmax -> bmm`. DCC jobs `49222759`, `49222781`, and `49222883` on RTX 2080 Ti, commit `989359f`, showed the accepted cross-attention BMM shape remains `~2.4-2.6x` faster than SDPA, but self-attention BMM is slower through active-prefix `L=576` and only mildly faster at `L>=640`. Using full-song active64 replay counts from job `49207288`, a thresholded self-attention BMM policy projects to only `0.463s` saved over SALVALAI full-song main generation, or `201.125 -> ~203.6 tok/s`. Do not add this PyTorch branch; future self-attention work must improve the common `128..640` buckets through a real cache/layout/native-kernel path. See `notes/2026-07-03-decode-attention-component-probe.md`.

The first persistent DecodeSession verifier proved graph/cache reuse is exact across multiple windows, but only in verifier code. Full-song q1 diagnostic job `49223017` on RTX 2080 Ti, commit `976b0eb`, measured `198` CUDA graph captures, `11` normalized graph shapes, `7,552` decode replays, and a duplicate-capture ceiling of `2.866s` (`7.610%` of main model time), projecting `202.830 -> 219.537 tok/s` if duplicate captures disappeared with all else unchanged. `utils/verify_persistent_decode_session.py` then reused one `MapperatorinatorCache`, a stable encoder-output buffer, and one shared graph cache across smoke windows. Job `49223121` passed two windows x `64` tokens with exact generated tokens, raw logits/top-k, and final RNG, capturing `2` graphs for `2` prefixes. Job `49223151` passed four windows x `256` tokens, capturing `5` graphs for `5` prefixes and replaying them `1,020` times. Keep this as verifier infrastructure; production reuse still needs a 15s smoke and full-song exact no-regression run before any speed claim. See `notes/2026-07-03-persistent-decode-session-verifier.md`.

For normal total-stage comparisons, do not isolate Hugging Face/model-download caches unless the claim explicitly includes model-cache cold cost. Isolating compiler/TorchInductor/CUDA caches is useful for cold compile checks, but invalidating model caches can make total-stage wall comparisons fail for reasons unrelated to generation speed.

Only count a speedup as equivalent when fixed-seed generated token IDs match the baseline for the same audio/config slice. Do not claim wins from changed precision, sampling policy, output policy, model quality, windowing/overlap, or generated-token behavior unless the run is explicitly labeled non-equivalent.

If default Mapperatorinator generation nondeterminism makes exact token identity too strict for a valuable runtime or batch candidate, use a separate `documented-drift` result class instead of weakening the same-calculation gate. A documented-drift candidate must prove the config and generation semantics are otherwise unchanged, characterize baseline-vs-baseline drift separately from baseline-vs-candidate drift, preserve generated-token counts/output policy, and justify the review with a large operational speedup. Do not mix documented-drift numbers into same-calculation speedup tables or use them to replace the retained baseline without explicit approval.

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

The harness loads the model once, resets RNG before every `generate()` call, writes one profile JSON per run, and writes `suite_manifest.json` with first-run, warmed-run, aggregate throughput, per-song throughput, profile paths, token hashes, and token-equivalence status. Manifest schema v3 also records `main_first_record`, `main_remaining_records`, aggregate first-record/remaining-record metrics, timing-context metrics, and runtime cache/env metadata (`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `HF_HOME`, `TORCH_LOGS`, etc.). This makes active-prefix first-window cold tax visible instead of letting warmed aggregate numbers hide setup or graph-capture cost. `warm_repeat` compares token IDs against run 0. `serial_multi_song` compares token IDs against each song's first repeat, so every song has its own equivalence baseline. Its warmed and multi-song results are useful operational evidence, but they are not cold single-song acceptance evidence.

Compare suite manifests with:

```bash
python utils/summarize_inference_profile.py \
  --compare-suite "$BASE_SUITE/suite_manifest.json" "$CANDIDATE_SUITE/suite_manifest.json" \
  --suite-scope warmed_runs \
  --strict \
  --json-output "$RUN_DIR/compare-suite-warmed.json"
```

Use `--suite-scope warmed_runs` for warmed-repeat or long-lived process claims. Use `--suite-scope all_runs` only when claiming all-run aggregate performance. In suite mode, `--strict` checks schema/shape/run contract, paired token hashes/token counts, selected-scope aggregate main-generation non-regression, selected-scope first-record and remaining-record segment non-regression, selected-scope timing-context non-regression, and per-song main-generation non-regression. It also prints cold run0 diagnostics. Add `--gate-cold-run0` only when the suite claim includes cold run0 non-regression; cold single-song promotion still requires normal full-song profile strict gates against the retained cold baseline.

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

Multi-token direct decode loop correctness gate, DCC job `49161597` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, torch `2.10.0+cu128`, Transformers `4.57.3`, dirty test patch on commit `cb874ee`:

| gate | token match | RNG match | raw-logit steps | max_abs | wall |
| --- | --- | --- | ---: | ---: | ---: |
| compile false, plain loop | PASS | PASS | 8 | `0.0` | `8.110s` |
| compile true, plain loop | PASS | PASS | 8 | `0.0` | `40.794s` |
| compile false, buffered active512 | PASS | PASS | 8 | `0.0` | `7.713s` |
| compile true, buffered active512 | PASS | PASS | 8 | `0.0` | `49.090s` |

Run dir: `/work/imt11/Mapperatorinator/runs/direct-loop-gate-49161597-cb874ee`.

This is a testing-suite improvement, not an inference speed result. `utils/verify_direct_decode_loop.py` compares normal HF `generate()` against a candidate loop passed through `custom_generate`, so HF still constructs the final `logits_processor`, `stopping_criteria`, cache, and generation config for both paths. It resets CPU/CUDA RNG state before each path, captures raw logits before in-place processors, compares generated token IDs, compares per-step raw logits/top-k, and verifies final RNG state. Use it before 15s smoke for direct/custom decode loop changes, especially when touching buffering, active-prefix decode, sampling, or stopping behavior.

Direct CUDA graph replay gate, DCC job `49165810` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, commit `0602d32`:

| gate | token match | RNG match | raw-logit steps | max_abs | graph replays | wall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| compile false, direct active512 | PASS | PASS | 8 | `0.0` | n/a | `17.989s` |
| compile true, direct active512 | PASS | PASS | 8 | `0.0` | n/a | `47.984s` |
| compile false, graph active512 | PASS | PASS | 8 | `0.0` | `7` | `7.415s` |
| compile true, graph active512 | PASS | PASS | 8 | `1.068e-4` | `7` | `18.623s` |

Run dir: `/work/imt11/Mapperatorinator/runs/direct-graph-gate-49165810-0602d32`.

This remains a verifier-only result. It proves that `utils/verify_direct_decode_loop.py --candidate-cuda-graph-forward` can replay a captured active-prefix bucket512 one-token forward across changing sampled tokens and `cache_position` while preserving generated-token identity, final RNG state, logits allclose, and top-k order for this 8-token gate. It does not prove inference throughput because the verifier still pays short-run setup, input preparation, tensor-copy, and reporting overhead. Before any speed claim, extend the graph gate to a longer direct-loop sample, then run 15s smoke token equivalence and untraced `profile_inference` throughput.

Follow-up 64-token graph gate, DCC job `49165980` on `dcc-dhvimdcore-gpu-ferc-s-p15-10`, RTX 2080 Ti, commit `26ec057`:

| gate | token match | RNG match | raw-logit steps | max_abs | graph replays | wall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| compile false, graph active512 | PASS | PASS | 64 | `0.0` | `63` | `28.448s` |
| compile true, graph active512 | PASS | PASS | 64 | `1.068e-4` | `63` | `70.252s` |

Run dir: `/work/imt11/Mapperatorinator/runs/direct-graph-gate64-49165980-26ec057`. This strengthens the correctness signal for graph replay across longer sampled-token/RNG progression, but it still stayed inside one active-prefix bucket and is not a throughput result.

Bucket-transition CUDA graph gate, DCC jobs `49166191` and `49166213` on `dcc-dhvimdcore-gpu-ferc-s-p15-12`, RTX 2080 Ti, commit `13335e5`:

| gate | token match | RNG match | raw-logit steps | max_abs | graph captures | graph replays | buckets | wall |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| compile false, graph active512 | PASS | PASS | 512 | `0.0` | `2` | `511` | `512:414`, `1024:97` | `66.538s` |
| compile true, graph active512 | PASS | PASS | 512 | `1.373e-4` | `2` | `511` | `512:414`, `1024:97` | `82.194s` |

Run dirs: `/work/imt11/Mapperatorinator/runs/direct-graph-transition-seq0-49166191-13335e5` and `/work/imt11/Mapperatorinator/runs/direct-graph-transition-seq0-compile-49166213-13335e5`. This proves the verifier can recapture graph buckets when active-prefix decode crosses from `512` to `1024`, with generated tokens, final RNG state, logits allclose, and top-k order still matching. It remains a verifier-only result, not throughput: `prepare_inputs_for_generation()`, tensor copies, short-run setup, and report overhead are still outside the captured graph.

Production active-prefix CUDA graph loop validation, DCC jobs `49166771` and `49167356` on RTX 2080 Ti, commit `8e8757b`:

| run | main tokens | main model time | main tok/s | total timing+map stage | token equivalence | strict status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| retained compile-only baseline, job `49113713` | `7,639` | `82.615s` | `92.465` | `113.928s` | baseline | baseline |
| active512 graph full-song, job `49167356` | `7,639` | `71.981s` | `106.125` | `101.481s` | PASS, `7,639 / 7,639` | aggregate PASS, per-window FAIL |

Run dirs:

- 15s smoke: `/work/imt11/Mapperatorinator/runs/active-graph-immediate-smoke-49166771-8e8757b`
- Full song: `/work/imt11/Mapperatorinator/runs/active-graph-immediate-full-49167356-8e8757b`

The 15s smoke compared immediate-capture active512 graph against compile-only with isolated compiler/CUDA caches. It passed same-calculation metadata, token equivalence (`1,084 / 1,084`), and strict per-window no-regression, with main-generation throughput `29.556 -> 110.451 tok/s` in that paired run. This promoted the candidate to full-song validation.

The full-song current-branch validation improved main generation by `+14.8%` (`92.465 -> 106.125 tok/s`) and total timing+map profiled stage time by `-10.9%` (`113.928s -> 101.481s`) while preserving exact fixed-seed main-token IDs. The compare metadata contract failed only because the retained baseline profile predates newer metadata keys (`temperature`, `top_p`, `cfg_scale`, `lookback`, etc.); there were no mismatched metadata values.

The strict per-window zero-regression gate still failed on 11/87 map windows. The failing windows totaled `128ms` of model-time overhead, compared with `10.634s` total main-generation model-time savings. Treat this as a scoped micro-regression, not a strict-pass result. Do not silently use it as the cold default; it is an accepted default-off opt-in performance path for explicit profiling or user-enabled fast inference.

Use:

```bash
python inference.py --config-name profile_salvalai \
  inference_generation_compile=true \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512 \
  inference_active_prefix_decode_cuda_graph=true \
  inference_active_prefix_decode_cuda_graph_warmup=0 \
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1
```

Keep the hard restrictions from the implementation: batch size 1, `use_server=false`, `parallel=false`, `cfg_scale=1`, `num_beams=1`, static cache, and decode-only active-prefix. Sampling, logits processors, RNG consumption, EOS behavior, generated-token accounting, and timing generation remain outside the captured graph path.

Stateful monotonic processor validation on top of active512 graph, DCC jobs `49167587`, `49168158`, and `49168188`:

| run | main tokens | main model time | main tok/s | total timing+map stage | token equivalence | strict status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| retained compile-only baseline, job `49113713` | `7,639` | `82.615s` | `92.465` | `113.928s` | baseline | baseline |
| active512 graph baseline, job `49167356` | `7,639` | `71.981s` | `106.125` | `101.481s` | PASS, `7,639 / 7,639` | aggregate PASS, per-window micro-regression |
| active512 graph + stateful monotonic, job `49168188` | `7,639` | `56.639s` | `134.873` | `78.473s` | PASS, `7,639 / 7,639` | PASS, `87 / 87` |

Run dirs:

- Logits processor diagnostic: `/work/imt11/Mapperatorinator/runs/active-graph-logitdiag-smoke-49167587-269f7ea`
- 15s smoke: `/work/imt11/Mapperatorinator/runs/stateful-monotonic-smoke-49168158-a980c8d`
- Full song: `/work/imt11/Mapperatorinator/runs/stateful-monotonic-full-49168188-a980c8d`

Job `49167587` used diagnostic counters, not throughput claims, and showed `MonotonicTimeShiftLogitsProcessor` dominated active-graph logits-processor time (`5.373s / 5.602s`) on the 15s smoke. Job `49168158` then passed same-calculation metadata, fixed-seed token equivalence (`1,084 / 1,084`), and improved active512 graph smoke throughput from `132.593` to `149.321 tok/s`; the only per-window regression was a `0.299ms` one-token window. Job `49168188` promoted the change to full song on RTX 2080 Ti, torch `2.10.0+cu128`, Transformers `4.57.3`, and passed token equivalence against both the retained compile-only baseline and the active512 graph baseline.

Use:

```bash
python inference.py --config-name profile_salvalai \
  inference_generation_compile=true \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512 \
  inference_active_prefix_decode_cuda_graph=true \
  inference_active_prefix_decode_cuda_graph_warmup=0 \
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1 \
  inference_stateful_monotonic_logits_processor=true
```

Keep this path default-off and scoped to the active-prefix CUDA graph/simple batch-1 mode until fresh profiling validates broader modes. The older pre-graph stateful monotonic attempt stayed rejected for normal generation because it regressed smoke throughput; the new result was accepted only after active-graph diagnostics made the logits processor a target-sized cost.

Active graph zero-warmup validation, DCC jobs `49204447` and `49204568` on RTX 2080 Ti, commit `f56f2f5`:

| run | main tokens | main model time | main tok/s | timing tok/s | total timing+map stage | token equivalence | strict status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| active512 graph + stateful, prior control job `49204317` | `7,639` | `55.230s` | `138.311` | `58.753` | `74.376s` | baseline | baseline |
| active512 graph + stateful, isolated warmup3 job `49204568` | `7,639` | `55.732s` | `137.066` | `58.616` | `74.685s` | baseline | baseline |
| active512 graph + stateful, isolated warmup0 job `49204568` | `7,639` | `52.107s` | `146.602` | `75.130` | `68.283s` | PASS, `7,639 / 7,639` | aggregate PASS, per-window micro-regression |

Run dirs:

- 15s warmup sweep: `/work/imt11/Mapperatorinator/runs/active-warmup-sweep-49204447-f56f2f5`
- Full-song isolated validation: `/work/imt11/Mapperatorinator/runs/active-warmup0-full-isolated-49204568-f56f2f5`
- Previous untraced control: `/work/imt11/Mapperatorinator/runs/stateful-bottleneck-49204317-94fcb31`

The 15s smoke sweep compared active512 graph + stateful with `inference_active_prefix_decode_cuda_graph_warmup=3`, `1`, and `0` after a compile-only control. All active variants were exact against warmup3 and compile-only (`1,084 / 1,084` main tokens). Warmup0 was fastest on that smoke (`153.030 -> 162.401 tok/s`, `+6.1%` versus warmup3), so it promoted to full-song validation.

The full-song isolated run used separate TorchInductor, Triton, CUDA, and HF cache directories for warmup0 and warmup3. Warmup0 improved main generation by `+7.0%` versus same-job warmup3 and `+6.0%` versus the previous untraced active512 graph + stateful control. Timing context also improved by `+28.2%` versus same-job warmup3 and `+27.9%` versus the previous control, with `821 / 821` timing tokens matching. Total timing+map profiled stage time improved `74.376s -> 68.283s` versus the previous control.

Strict zero-tolerance per-window gates still failed on tiny late one-token windows: versus same-job warmup3, one map window regressed by `0.128ms` model time; versus the previous control, two map windows regressed by `0.126ms` and `0.487ms`, and one timing window regressed only in outer wall by `0.150ms` while model time improved. Treat these as scoped micro-regressions, not a strict-pass result. Because this is a simple default-off active-graph setting, exact, faster on main/timing/total aggregates, and strategically useful for the 200 tok/s runtime path, keep `inference_active_prefix_decode_cuda_graph_warmup=0` as the active graph default.

Use:

```bash
python inference.py --config-name profile_salvalai \
  inference_generation_compile=true \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512 \
  inference_active_prefix_decode_cuda_graph=true \
  inference_active_prefix_decode_cuda_graph_warmup=0 \
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1 \
  inference_stateful_monotonic_logits_processor=true
```

Post-warmup0 attribution, DCC job `49204765`, commit `bb11d9f`:

| run | main tokens | main model time | main tok/s | timing tok/s | token equivalence | note |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| full default omission control | `7,639` | `51.979s` | `146.963` | `70.061` | PASS vs warmup0 | validates default `warmup=0`; not a new accepted baseline because total stage/timing context differed under fresh cache/load state |
| full default diagnostic | `7,639` | `52.510s` | `145.476` | `75.020` | PASS vs control | diagnostic overhead about `1%` on main |

Run dir: `/work/imt11/Mapperatorinator/runs/active-w0-nextdiag-49204765-bb11d9f`.

The diagnostic counters ruled out several easy next targets:

| counter | full diagnostic wall time |
| --- | ---: |
| `token_append_stop_wall_cpu_s` | `35.276s` |
| `stopping_criteria_wall_cpu_s` | `34.826s` |
| `logits_processor_wall_cpu_s` | `4.452s` |
| `prepare_inputs_wall_cpu_s` | `3.874s` |
| `decode_forward_wall_cpu_s` | `3.444s` |
| `sampling_wall_cpu_s` | `1.167s` |
| `compile_lookup_wall_cpu_s` | `0.0106s` |

`compile_lookup_wall_cpu_s` is too small to optimize. The captured graph input shapes were already minimal (`decoder_input_ids`, `decoder_attention_mask`, `decoder_position_ids`, and `cache_position`), so pruning dead encoder/static inputs from graph replay is not target-sized. The seq9 torch-profiler trace showed `aten::_local_scalar_dense`/`aten::isin` around stopping, but the wall bucket overlaps required per-token synchronization/control; do not treat it as pure Python-loop overhead.

Rejected simple active-prefix stopping specialization, DCC job `49204960`, dirty local patch on top of commit `bb11d9f`:

| run | main tokens | main model time | main tok/s | timing tok/s | token equivalence | status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| active512 graph + stateful + warmup0 smoke baseline | `1,084` | `6.675s` | `162.401` | `47.556` | baseline | baseline |
| simple stopping candidate | `1,084` | `6.433s` | `168.512` | `48.209` | PASS, `1,084 / 1,084` | rejected |

Run dir: `/work/imt11/Mapperatorinator/runs/simple-stop-smoke-49204960-bb11d9f-simple-stop-dirty`. The 64-step direct-loop graph gate passed for generated tokens, raw logits, final RNG state, and stop reason. The candidate still improved main generation by only `+3.76%`, below the keep threshold for a custom stopping path, and the diagnostic run still reported `stopping_criteria_wall_cpu_s=4.265s` of `6.450s` main model time. The patch was reverted without full-song promotion. See `notes/2026-07-02-simple-stopping-scout.md`.

Accepted active-prefix graph bucket-size reduction, DCC jobs `49205143`, `49205622`, and `49206207` on RTX 2080 Ti, commit `39e85e4`:

| bucket | main tokens | main model time | main tok/s | timing tok/s | total timing+map stage | token equivalence vs bucket512 | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `512` | `7,639` | `51.887s` | `147.223` | `75.016` | `67.773s` | baseline | historical active control |
| `192` | `7,639` | `49.369s` | `154.733` | `77.070` | `65.225s` | PASS main and timing | safer fallback |
| `64` | `7,639` | `49.101s` | `155.578` | `76.524` | `64.946s` | PASS main and timing | fastest opt-in |

Run dirs:

- Broad 15s smoke sweep: `/work/imt11/Mapperatorinator/runs/active-bucket-sweep-49205143-39e85e4`
- Low-bucket 15s smoke sweep: `/work/imt11/Mapperatorinator/runs/active-low-bucket-sweep-49205622-39e85e4`
- Full-song validation: `/work/imt11/Mapperatorinator/runs/active-bucket-full-49206207-39e85e4`

All smoke candidates tested from bucket64 through bucket1024 matched bucket512 generated token IDs for both main and timing. Full-song bucket64 improved same-job main generation by `+5.7%` over bucket512 (`147.223 -> 155.578 tok/s`), improved aggregate timing by `+2.0%`, and reduced total timing+map stage wall by `4.2%`. Full-song bucket192 improved same-job main by `+5.1%`, improved aggregate timing by `+2.7%`, and reduced total stage by `3.8%`.

Strict per-window checks still failed with scoped small regressions. Bucket64 failed `12 / 87` main windows with `67ms` total failed-window model overhead and `29 / 87` timing windows with `185ms` total failed-window model overhead. Bucket192 failed the same `12 / 87` main windows with `71ms` failed-window overhead but only `3 / 87` timing windows with `25ms` failed-window overhead. Because bucket64 is config-only, exact, improves full-song main/timing/total aggregates, and has small absolute failed-window overhead relative to `2.786s` main model-time savings and `2.828s` total-stage savings, keep it as the fastest opt-in setting. Use bucket192 as the safer fallback when timing-context per-window stability is more important than maximum main-generation throughput. See `notes/2026-07-02-active-bucket-size-sweep.md`.

Use the fastest exact opt-in path:

```bash
python inference.py --config-name profile_salvalai \
  inference_generation_compile=true \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=64 \
  inference_active_prefix_decode_cuda_graph=true \
  inference_active_prefix_decode_cuda_graph_warmup=0 \
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1 \
  inference_stateful_monotonic_logits_processor=true \
  inference_q1_bmm_cross_attention=true
```

Accepted q_len=1 BMM cross-attention, DCC jobs `49212482`, `49212884`, `49213018`, `49213078`, and `49213490` on RTX 2080 Ti, commit `3af8d69`:

| run | main tokens | main model time | main tok/s | timing tok/s | total timing+map stage | token equivalence | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| active64 graph + stateful control | `7,639` | `49.279s` | `155.014` | `75.652` | `65.200s` | baseline | control |
| q1 BMM cross-attention candidate | `7,639` | `37.981s` | `201.125` | `84.160` | `52.638s` | PASS main and timing | fastest opt-in |

Run dirs:

- BMM attention microbench: `/work/imt11/Mapperatorinator/runs/attn-bmm-sm75-49212482`
- One-token logits gate: `/work/imt11/Mapperatorinator/runs/q1bmm-logit-gate-49212884-3af8d69-len128`
- Direct-loop token/logit/RNG gate: `/work/imt11/Mapperatorinator/runs/q1bmm-direct-gate-49213018-3af8d69`
- 15s smoke: `/work/imt11/Mapperatorinator/runs/q1bmm-smoke15-49213078-3af8d69`
- Full-song validation: `/work/imt11/Mapperatorinator/runs/q1bmm-full-49213490-3af8d69`

The trigger was the active64 no-graph trace showing repeated unmasked cross-attention SDPA calls with Q shape `[1, 12, 1, 64]` and K/V shape `[1, 12, 1024, 64]`. The isolated SM75 microbench showed explicit fp32 `bmm -> softmax -> bmm` was much faster than SDPA for that exact q_len=1 cross-attention shape (`0.1486ms -> 0.0548ms`, `2.71x`, max abs `1.64e-7`). Longer q_len=1 lengths favored BMM even more (`L=2560`: `0.3643ms -> 0.0565ms`), while small lengths such as `64` and `96` favored SDPA; therefore the accepted production hook is cross-attention-only, not a broad self-attention replacement.

Correctness gates passed before the speed claim. The one-token direct static-cache logits gate passed with matching top-k and `max_abs=4.196e-05`. The 64-token direct-loop gate passed generated-token identity, final RNG-state identity, logits allclose, and top-k equality with active-prefix bucket64 plus CUDA graph replay. The 15s smoke matched `1,084 / 1,084` main tokens and improved active64 graph main generation `168.853 -> 224.910 tok/s`; timing-context also improved `44.143 -> 48.686 tok/s`.

Full-song validation reached the long-range `200 tok/s` target for this default-off exact opt-in path: active64 graph + stateful control `155.014 -> 201.125 tok/s` (`+29.7%`), main model time `49.279s -> 37.981s`, timing-context `75.652 -> 84.160 tok/s`, and total timing+map stage `65.200s -> 52.638s`. Fixed-seed generated token IDs matched for main (`7,639 / 7,639`) and timing (`821 / 821`).

Strict zero-tolerance per-window gates still failed on tiny late one-token windows. Main generation failed `4 / 87` windows with only `4.4ms` total failed-window model overhead, versus `11.298s` aggregate main model-time savings. Timing failed `1 / 87` windows with `1.0ms` failed-window model overhead, versus `1.097s` aggregate timing model-time savings. Treat this as a scoped micro-regression, not a strict-pass result. Because the candidate is default-off, exact at the generated-token level, improves main/timing/total aggregates, and crosses the target, keep it as the fastest opt-in path. See `notes/2026-07-03-q1-bmm-cross-attention.md`.

Five-song before/after validation, DCC jobs `49218365` through `49218368` on RTX 2080 Ti, commit `8a2de72`, compared original-repo-equivalent flags disabled against the current optimized opt-in stack on Lambada, PEGASUS, Ela ke Leitada, SALVALAI, and Nube Negra. Official TPS came from untraced `profile_inference`; each job also wrote 1s `nvidia-smi` telemetry and an Nsight Systems diagnostic report.

| class | before main tok/s | after main tok/s | speedup | equivalence | note |
| --- | ---: | ---: | ---: | --- | --- |
| separate cold, five fresh processes | `64.802` | `195.545` | `+201.8%` | PASS, all five songs | cold single-song evidence |
| together first run | `64.614` | `201.749` | `+212.2%` | PASS | first song in long-lived process |
| together all runs | `60.541` | `194.791` | `+221.8%` | PASS | serial multi-song, two repeats |
| together warmed runs | `56.834` | `194.116` | `+241.5%` | PASS | batch-adjacent evidence, not true batching |

All five separate cold main-generation strict comparisons passed, and suite `all_runs`, `warmed_runs`, and `all_runs --gate-cold-run0` strict comparisons passed. Separate timing-context aggregate throughput improved on every song, but `timing/sequential/seq0` regressed for each song, so keep that as a scoped first-record timing caveat. See `notes/2026-07-03-five-song-before-after-profile.md`.

Post-bucket64 diagnostics, DCC jobs `49207288` and `49208036` on RTX 2080 Ti, commit `cf4f87e`:

| diagnostic run | main tokens | main model time | main tok/s | timing tok/s | note |
| --- | ---: | ---: | ---: | ---: | --- |
| active64 diagnostic full song | `7,639` | `48.914s` | `156.171` | `75.903` | close to accepted active64 throughput |
| active64 seq66 torch profile | `7,639` | `53.156s` | `143.709` | `63.175` | diagnostic only; torch profiler overhead |

Run dir: `/work/imt11/Mapperatorinator/runs/active64-nextdiag-49207288-cf4f87e`. The diagnostic full-song run showed:

| counter | wall time |
| --- | ---: |
| `token_append_stop_wall_cpu_s` | `30.269s` |
| `stopping_criteria_wall_cpu_s` | `29.823s` |
| `decode_forward_wall_cpu_s` | `4.990s` |
| `logits_processor_wall_cpu_s` | `4.426s` |
| `prepare_inputs_wall_cpu_s` | `3.848s` |
| `sampling_wall_cpu_s` | `1.156s` |
| `compile_lookup_wall_cpu_s` | `0.010s` |

CUDA graph diagnostics reported `198` captured graphs, `3.328s` total capture time, `7,552` decode replays, and `115` bucket transitions. Normalizing by active-prefix length and static tensor input shapes collapses those captures to only `11` graph shapes. The duplicate-capture ceiling is `3.153s`, or `6.446%` of active64 main model time; if duplicate capture disappeared with all other costs unchanged, estimated throughput would be about `166.9 tok/s`. This capture tax is now a real target-sized cost. The current `graph_cache` is local to each `active_prefix_decode_generate()` call, so captures repeat across generation windows; however, replay also closes over per-window cache and encoder-output objects, so persistent cross-window graph reuse requires a bufferized direct runtime rather than a small dictionary hoist.

The seq66 trace again showed no single embarrassing small-kernel cleanup. Fused attention and one-token linear/GEMV-heavy decoder work dominate CUDA time: `mapperatorinator.active_prefix.decode_forward.cuda_graph` had `694.950ms` self CUDA, the SM75 FMHA kernel had `379.462ms` self CUDA, and GEMV/GEMM families contributed tens of milliseconds each in that one traced window. CPU-side `stopping_criteria` and `cudaStreamSynchronize` remain synchronization/control symptoms, consistent with the rejected simple stopping specialization.

Tiny-bucket smoke job `49208036`, run dir `/work/imt11/Mapperatorinator/runs/active-tiny-bucket-sweep-49208036-cf4f87e`, tested buckets `16`, `32`, `48`, `80`, and `96` against bucket64. All were token-equivalent on the 15s smoke, but none deserved full-song promotion: bucket16/32/48 regressed main generation, bucket80 was flat, and bucket96 improved smoke main by only `+0.7%`. Keep bucket64 as the current full-song opt-in bucket. See `notes/2026-07-02-active64-post-bucket-diagnostics.md`.

Follow-up tiny-bucket duplicate-capture diagnostic job `49209984`, run dir `/work/imt11/Mapperatorinator/runs/active-tiny-bucket-diag-49209984-24342d0`, repeated buckets `16`, `32`, `48`, `64`, `96`, `128`, and `192` with `profile_active_prefix_decode_diagnostics=true` and isolated compiler/cache dirs. All candidates matched bucket64 main tokens (`1,084 / 1,084`). Bucket96 was fastest on aggregate but only `+1.0%` over bucket64 (`169.290 -> 170.973 tok/s`) and still had strict per-window noise failures. The duplicate-capture projections were more important: bucket96 projected only `176.136 tok/s` without duplicate captures, while bucket64 projected `175.853 tok/s`; smaller buckets had larger duplicate-capture ceilings but worse actual throughput. This confirms that persistent graph/cache reuse may be a 5-10% cleanup or future batch-serving project, but it is not enough to reach `200 tok/s` by itself and should start as a verifier-only multi-window runtime, not production wiring. See `notes/2026-07-03-tiny-bucket-duplicate-capture-diagnostics.md`.

Active64 kernel attribution jobs `49207288` and `49210611` make the next target clearer. In the accepted CUDA-graph trace, classified kernel self-CUDA after excluding the graph replay range was `57.4%` FMHA attention, `17.2%` GEMV linear, `4.3%` SGEMM linear, `9.9%` elementwise, `6.7%` device copies, `2.0%` layernorm, `1.9%` top-p sort, and `0.1%` softmax. A no-graph component trace of `main_generation.seq9` was directionally similar: `63.6%` FMHA attention, `18.0%` GEMV/SGEMM linear, `9.3%` elementwise, and `6.0%` device copies. Treat this as diagnostic-only attribution, but it strongly argues that the next `200 tok/s` path is q_len=1 attention/cache-layout/backend work plus one-token linear/GEMV launch reduction, not fused sampling. See `notes/2026-07-03-active64-kernel-attribution.md`.

FlashInfer SM75 feasibility jobs `49212042` and `49212270`, run dirs `/work/imt11/Mapperatorinator/runs/flashinfer-sm75-light-49212042` and `/work/imt11/Mapperatorinator/runs/flashinfer-fp32-h12-49212270`, rejected FlashInfer as a current same-calculation fp32 backend. An isolated target with `flashinfer-python==0.6.14`, `apache-tvm-ffi==0.1.12`, and `nvidia-ml-py==13.610.43` imported on RTX 2080 Ti, but q_len=1 `torch.float32` `single_decode_with_kv_cache` failed before timing with `KeyError(torch.float32)`, including the actual traced model head shape `H=12`, `D=64`. The fp16 path then required JIT tooling and would be non-equivalent to the current fp32 objective regardless. Do not integrate or retry FlashInfer as a quick attention swap unless a future isolated probe first proves fp32 decode support on SM75 and a static-bucket valid-length/mask story that can pass the one-token logits gate. See `notes/2026-07-03-flashinfer-sm75-feasibility.md`.

Rejected delayed-capture variant, DCC job `49166715`, commit `f712b59`: setting `inference_active_prefix_decode_cuda_graph_min_decode_steps=16` remained token-equivalent on 15s smoke but failed per-window no-regression on 5/10 windows and reduced graph-path aggregate throughput versus immediate capture. Keep `min_decode_steps=1` as the default. Only revisit delayed capture if a future profiler trace shows graph capture itself has become the dominant cost and a better adaptive policy can avoid medium-window regressions.

Rejected active-prefix mask fast path, DCC jobs `49158276` and `49158365` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti:

| run | main tokens | main model time | tok/s | token equivalence | status |
| --- | ---: | ---: | ---: | --- | --- |
| cold compile-only | `1,084` | `22.369s` | `48.461` | baseline | baseline |
| active512 mask fast path | `1,084` | `30.544s` | `35.490` | PASS | rejected |

The candidate moved active-prefix decode input preparation under the active-prefix context and capped static-cache mask construction to bucket length. One-token logits gates passed for compile disabled and enabled (`seq9`, active-prefix decode length `512`), and the 15s smoke matched all `1,084 / 1,084` generated main-token IDs. It still regressed cold main generation by `-26.8%`: the first long map window worsened (`seq3 16.259s -> 26.186s`) despite faster post-warm windows (`seq9 103.007 -> 150.046 tok/s`). The patch was reverted. See `notes/2026-07-01-active-prefix-mask-fastpath.md`.

Rejected simple measured active-prefix primer, DCC job `49159121` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti:

| run | main tokens | main model time | tok/s | token equivalence | status |
| --- | ---: | ---: | ---: | --- | --- |
| cold compile-only | `1,084` | `22.599s` | `47.967` | baseline | baseline |
| active512 | `1,084` | `30.266s` | `35.816` | PASS | baseline active-prefix candidate |
| active512 + measured primer | `1,084` | `31.477s` | `34.438` | PASS | rejected |

The primer used a scratch cache, fresh logits processors, and an RNG fork. It moved work out of the first long active-prefix map window (`seq3 25.909s -> 14.923s`) but paid setup in earlier records (`seq0=11.705s`) and regressed the active512 aggregate by `-3.8%`. The code was reverted. This supports the graph/capture/specialization diagnosis but shows that naive throwaway-generate priming is not enough; future priming needs better bucket/shape coverage and explicit profile schema support. See `notes/2026-07-01-active-prefix-primer.md`.

Rejected active-prefix direct per-layer static-cache update dispatch bypass, DCC job `49160690` on `dcc-core-ferc-s-z25-21`, RTX 2080 Ti:

| run | main tokens | main model time | tok/s | token equivalence | status |
| --- | ---: | ---: | ---: | --- | --- |
| cold compile-only | `1,084` | `22.406s` | `48.380` | baseline | baseline |
| active512 old path | `1,084` | `30.906s` | `35.074` | PASS | baseline active-prefix candidate |
| active512 direct-layer update | `1,084` | `31.330s` | `34.599` | PASS | rejected |

The candidate bypassed `Cache.update(..., layer_idx, ...)` dispatch by calling the selected static-cache layer update directly during active-prefix VarWhisper SDPA self-attention decode. One-token logits gates passed with compile disabled (`max_abs=0.0`) and compile enabled (`max_abs=2.2888e-05`), and the 15s smoke matched all `1,084 / 1,084` generated main-token IDs. It still regressed active512 by `-1.4%`, did not improve the first long map window (`seq3 26.539s -> 26.926s`), and slightly worsened a post-warm window (`seq9 1.574s -> 1.599s`). The code was reverted. See `notes/2026-07-01-active-prefix-direct-layer-cache-update.md`.

Rejected active-prefix buffered input-id preallocation, DCC job `49162138` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti:

| run | main tokens | main model time | tok/s | token equivalence | status |
| --- | ---: | ---: | ---: | --- | --- |
| cold compile-only | `1,084` | `21.460s` | `50.513` | baseline | baseline |
| active512 old path | `1,084` | `29.006s` | `37.372` | PASS | baseline active-prefix candidate |
| active512 buffered input IDs | `1,084` | `29.802s` | `36.373` | PASS | rejected |

The candidate preallocated the generated `input_ids` buffer inside the active-prefix custom decode loop to avoid per-token `torch.cat`. The multi-token direct-loop gate had already shown this can preserve token/logit/RNG semantics, but 15s profile evidence showed it regressed active512 main generation by `-2.7%` and remained `-28.0%` below the compile-only smoke baseline. Timing generation improved in this specific run (`4.4 -> 7.0 tok/s`), but the objective is main-generation throughput and the main path got worse. The code was reverted. See `notes/2026-07-01-active-prefix-buffered-input-ids.md`.

Rejected naive TorchInductor cudagraph-tree disabling, DCC jobs `49162564`, `49162714`, and `49162877` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti:

| run | main tokens | main model time | tok/s | timing model time | token equivalence | status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| compile-only baseline | `1,084` | `21.855s` | `49.600` | `34.243s` | baseline | baseline |
| active512 default | `1,084` | `29.820s` | `36.352` | `39.420s` | PASS | baseline active-prefix candidate |
| active512 global no-cudagraph-trees | `1,084` | `16.755s` | `64.696` | `73.607s` | PASS | rejected, timing regression |
| active512 MAP-only no-cudagraph-trees | `1,084` | `63.546s` | `17.059` | `37.651s` | PASS | rejected |

`torch._inductor.config.triton.cudagraph_trees=False` is diagnostic evidence, not an accepted optimization. Setting it globally from process start made active-prefix main generation exact and much faster on the 15s smoke (`+30.4%` vs compile-only main generation), but it more than doubled timing-context model time. A follow-up default-off scoped patch that limited active-prefix/no-cudagraph-trees to `MAP` preserved token equivalence and kept timing closer to baseline, but it catastrophically regressed main generation (`-64.5%`). The code was reverted. The useful conclusion is that active-prefix still needs deeper graph-state isolation, likely bucket-scoped compiled calls or a direct decode runtime, not a naive global or per-call cudagraph-tree switch. See `notes/2026-07-01-active-prefix-cudagraph-trees.md`.

Rejected bucket-scoped active-prefix `torch.compile` wrappers, DCC job `49163585` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, commit `b4f3f11` with the code reverted in `ea1c422`:

| variant | main tokens | main model time | tok/s | token equivalence | decision |
| --- | ---: | ---: | ---: | --- | --- |
| compile-only | `1,084` | `21.773s` | `49.786` | baseline | retained smoke baseline |
| active512 shared compile | `1,084` | `29.741s` | `36.448` | PASS vs bucket | existing active-prefix path |
| active512 bucket-scoped compile | `1,084` | `31.946s` | `33.933` | PASS | rejected |

The direct-loop gate passed first (`token_match=true`, `logits_pass=true`, `rng_match=true`, `max_abs=0.0`, `8` steps), so the wrapper was exact. It still made cold 15s smoke performance worse: `-6.9%` main-generation throughput versus active512 shared compile and `-31.8%` versus compile-only, with strict per-window non-regression failing for all active-shared-vs-bucket main records. This means simply giving each active-prefix bucket its own compiled wrapper adds more compile/graph overhead than it removes. See `notes/2026-07-01-active-prefix-bucket-compile-scope.md`.

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

`configs/inference/profile_salvalai_smoke15.yaml` sets `start_time=71000`, `end_time=86000`, `seed=12345`, `attn_implementation=sdpa`, `use_server=false`, `inference_generation_compile=true`, `inference_active_prefix_decode_loop=false`, and `profile_record_token_ids=true`. Use this for first-pass scouting; promote to the 30s smoke or full-song configs only when the 15s result is token-equivalent and plausibly meaningful. For compile/runtime changes with one-time specialization costs, inspect post-warmup windows before rejecting a token-equivalent candidate, but full-song non-regression is still required before acceptance.

Reference 15s smoke profiles from commit `ce92ebb` on RTX 2080 Ti node `dcc-core-ferc-s-z25-21`:

| run | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| compile-disabled | `49132862` | `/work/imt11/Mapperatorinator/runs/smoke15-nocompile-49132862-ce92ebb/beatmap3ccc5112a05940db8fdd747994c4bef2.osu.profile.json` | 1,084 | 22.068s | 49.1 |
| retained compile baseline | `49132861` | `/work/imt11/Mapperatorinator/runs/smoke15-compile-49132861-ce92ebb/beatmapfb8360ef206b41f4a6cc1fe7a6a735ed.osu.profile.json` | 1,084 | 13.335s | 81.3 |

Use the comparison helper as a real gate, not only a readable summary:

```bash
python utils/summarize_inference_profile.py \
  --compare "$BASELINE_PROFILE" "$CANDIDATE_PROFILE" \
  --label main_generation \
  --strict \
  --json-output "$WORK/runs/profile-compare-${SLURM_JOB_ID}.json"
```

`--strict` exits nonzero when same-calculation metadata differs, generated token IDs are missing or different, generated-token or record counts change, aggregate throughput/model/wall/stage time regresses, or any selected-label per-window record regresses. Run it at least for `main_generation` before promotion; full-song acceptance still needs timing-generation and total-stage inspection for non-regression.

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
  --sequence-index 9 \
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
the direct candidate step. Add `--include-no-cache-diagnostics` only when debugging the ABI; it runs slower no-cache
full-prefix forwards that are not part of the pass condition. Passing this gate is
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

For active-prefix cold-overhead attribution, add:

```bash
profile_active_prefix_decode_diagnostics=true
```

This is default-off and only records diagnostics when `inference_active_prefix_decode_loop=true`. It adds an `active_prefix_decode_diagnostics` object to each generation profile record with CPU-side wall counters for compile lookup, prepare inputs, prefill, decode forward, cache/model-kwargs update, logits processors, sampling, token append, and stopping checks, plus `decode_steps`, `bucket_lengths_seen`, and `bucket_transition_count`. It does not call `.cpu()`, `.item()`, or `torch.cuda.synchronize()` inside the decode loop. Pair it with `profile_generation_detail_ranges=true` when NVTX/`record_function` ranges are needed; otherwise it still writes JSON counters without enabling the broader VarWhisper detail ranges. This mode is diagnostic only and should not be used for throughput claims.

Use `python utils/summarize_active_prefix_diagnostics.py PROFILE.json --label main_generation --json-output active_diagnostics.json` to aggregate per-record active-prefix counters, CUDA graph capture/replay counts, duplicate-capture ceiling by normalized graph shape, bucket usage, and per-logits-processor wall time without writing ad hoc parsers for every DCC run.

DCC validation job `49164750` on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, commit `c7ab3b8`, compared active512 15s smoke with and without `profile_active_prefix_decode_diagnostics=true`. The Slurm job exited `FAILED` because the final ad-hoc report snippet had a quoting bug, but both profiles and compare reports were written under `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-smoke-49164750-c7ab3b8`. Main-generation token equivalence passed (`1,084 / 1,084`) and diagnostics changed aggregate main-generation throughput only `34.629 -> 34.623 tok/s` (`-0.02%`). Timing-context token equivalence also passed (`164 / 164`). Zero-tolerance per-window no-regression failed on sub-percent to about `1.3%` noise, so treat this as diagnostic validation rather than a throughput claim.

The useful attribution from that run: across 20 generation records, `decode_forward_wall_cpu_s` summed to `54.791s`, with `first_decode_forward_wall_cpu_s=30.560s` and `steady_decode_forward_wall_cpu_s=24.231s`. For map records specifically, first decode forward was `11.657s`, steady decode forward was `12.296s`, logits processors were `5.250s`, and sampling was `0.243s`. The first long map window (`seq3`) paid `11.538s` in first decode forward, while later map windows reached about `116-138 tok/s`. This supports graph/runtime stabilization as the next active-prefix target before fused sampling/logits work.

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

Historical pre-q1-BMM 200 tok/s phase stop rule: stop the long-running optimization loop only when either the full-song RTX 2080/2080 Ti run reaches at least `200 tok/s` with identical fixed-seed tokens, or profiling across multiple exact-calculation optimization families shows no remaining plausible major exact-calculation path. This rule is now historical because job `49213490` reached `201.125 tok/s` on the default-off exact opt-in path; use the `500 tok/s single-song campaign` section for current work.

Historical status after active-prefix validation, before q1-BMM cross-attention:

- Retained cold single-song baseline: SDPA plus `inference_generation_compile=true`, active-prefix disabled. Full-song job `49113713` generated `7,639` main tokens in `82.615s`, `92.465 tok/s`, token equivalence PASS against the compile-disabled baseline. Same-commit paired job `49150185` measured compile-only at `92.998 tok/s`.
- Current retained flags: `attn_implementation=sdpa`, `inference_generation_compile=true`, `inference_active_prefix_decode_loop=false`, `seed=12345`, `profile_record_token_ids=true`.
- Active-prefix is an opt-in strategic candidate, not the retained cold baseline. Bucket `256` in job `49150185` looked like a large exact win (`92.998 -> 121.926 tok/s`), but later full-song runs showed order/warm-state sensitivity. Job `49151748` measured bucket `256` cold at `88.699 tok/s`, bucket `128` at `81.396 tok/s`, and bucket `512` warm at `108.313 tok/s`, all token-equivalent against bucket `256`.
- Bucket `512` cold-first validation job `49152465` generated `7,639` main tokens in `78.108s`, `97.800 tok/s`, token equivalence PASS against a same-job compile-only run. That is only about `+5.8%` against the retained `92.465 tok/s` compile-only baseline and it still regressed the first main-generation window and timing/total stage against clean retained runs, so it stays default-off.
- At this checkpoint, reaching `200 tok/s` from the retained `92.465 tok/s` baseline required reducing main-generation model time to `38.195s`, so about `44.420s` or `53.8%` of retained model time remained to remove. The later q1-BMM path reached `37.981s`.

For that historical 200 tok/s phase, the next experiments were ranked this way:

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

## Historical 100 tok/s Goal Prompt

Superseded by the 200 tok/s and then 500 tok/s campaigns. Keep this only as campaign history.

```text
Optimize Mapperatorinator inference on RTX 2080/2080 Ti for same-calculation speedups only. Current baseline is roughly 65-78 tok/s; first success target is 100 tok/s main-generation throughput, strong milestone is 120 tok/s, and 150+ tok/s is stretch. Do not claim speedups from changed precision, sampling policy, output policy, model quality, windowing/overlap, or generated-token behavior unless explicitly labeled non-equivalent.

Use a profiling-first loop. Start with a middle-30-second song smoke slice, prove fixed-seed generated tokens match baseline, and only promote promising changes to full-song SALVALAI runs. Use full-song runs for accepted results. Keep SDPA as the baseline unless profiler evidence strongly contradicts it. Separate true model time from torch.profiler overhead.

Scout improvement ideas with subagents and web research as useful, but accept only measured wins. Prioritize big wins in generation-loop structure, cache behavior, mask construction, logits processors, repeated small kernels, memory movement, and avoidable per-token setup. Keep changes that improve RTX 2080 full-song main-generation throughput by >=10%; keep 5-10% only if very simple; remove 1-3% complexity. Commit and push clean checkpoints for accepted wins, document why each win worked or failed in docs/inference_profiling.md, update AGENTS.md with durable conventions, write notes under notes/, and stop when the 100 tok/s goal is reached or profiling shows no remaining plausible >=10% exact-calculation improvement.
```

## Historical 200 tok/s Goal Prompt

Superseded by the accepted `201.125 tok/s` exact opt-in path and the current 500 tok/s single-song campaign. Keep this only as campaign history.

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

Post-active-prefix attribution job `49150687` traced the bucket-256 path with detailed ranges and `TORCH_LOGS=recompiles,cudagraphs`. It is diagnostic only, because torch profiler/logging inflated the traced `seq9` wall time to `107.836s`. The useful evidence at that point was that sampling/logits processors remained small, while graph/runtime churn was large: `1,058` graph-recording log lines, `20,504` TorchDynamo cache lookups, `16,310` CUDA graph launches, and a visible `sdpa_attention_forward` recompile from full static key length `2560` to active length `1024`. See `notes/2026-07-01-post-active-prefix-attribution.md`.

The later manual CUDA graph path changed the cost mix enough that logits processors became target-sized. Job `49168188` accepted `inference_stateful_monotonic_logits_processor=true` only in that active512 graph context; it should not be generalized back to normal generation without new evidence.

## Rejected Exact-Calculation Experiments

### Historical pre-graph stateful monotonic time-shift masking

This rejection applies to the old normal-generation/pre-graph implementation. It was superseded in the active-prefix CUDA graph context by job `49168188`, where diagnostics showed monotonic masking had become a dominant CPU-side cost and the new default-off flag passed full-song token equivalence and non-regression.

Attempted in commit `9d7e5b7` and reverted after smoke profiling. The change replaced the per-token full-prefix scan in `MonotonicTimeShiftLogitsProcessor` with a stateful batch-size-1 path that tracks the last time-shift token after the last SOS token.

RTX 2080 Ti smoke comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `01c18d6` | `49109301` | `/work/imt11/Mapperatorinator/runs/smoke-base-49109301-01c18d6/beatmap6f980906005d441fb87edde94f269b83.osu.profile.json` | 2,894 | 41.707s | 69.4 |
| candidate | `9d7e5b7` | `49109743` | `/work/imt11/Mapperatorinator/runs/smoke-cand-49109743-9d7e5b7/beatmapd81b370ad0ac422cb1b5a01b3d3a093d.osu.profile.json` | 2,894 | 43.487s | 66.5 |

`utils/summarize_inference_profile.py --compare` reported token equivalence PASS for all `2,894` generated main-generation token IDs, but throughput was `-4.1%` worse. The likely reason is that removing `torch.isin`/full-prefix work also added per-token state, mask, slice, `masked_fill`, and `torch.where` work; this did not pay back in normal generation. Do not reintroduce this shape of stateful logits processor in normal generation without profiler evidence that the replacement removes more work than it adds.

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

The final rejection set includes copy-compatible custom `_sample` hook, preallocated `_sample` loop, persistent static-mask mutation, compile-config variants (`dynamic=False`, `dynamic=True`, `mode="max-autotune"`, `fullgraph=True`), TensorRT-RTX current-env lowering, static-cache prefix trim, dynamic/default cache generation, final-position logits, old pre-graph stateful monotonic masking, and `torch.inference_mode` wrapping. Sampling/logits fusion stayed below the profiling threshold in the post-warmup trace, and batching/parallel/server/window changes are non-equivalent unless separately proven token-identical. Later active-prefix CUDA graph diagnostics changed the cost mix and justified reintroducing stateful monotonic masking only as a default-off active-graph path.

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
