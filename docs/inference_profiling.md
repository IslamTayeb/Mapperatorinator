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

The current RTX 2080/2080 Ti baseline after the accepted generation-compile win is roughly `92 tok/s` for full-song main-generation throughput. Use these targets:

- Starter success target: `100 tok/s` on a full-song run.
- Strong first milestone: `120 tok/s`.
- Stretch target: `150+ tok/s`.
- Long-range north star: `200 tok/s`, but do not use it as the first stop condition.

Only count a speedup as equivalent when fixed-seed generated token IDs match the baseline for the same audio/config slice. Do not claim wins from changed precision, sampling policy, output policy, model quality, windowing/overlap, or generated-token behavior unless the run is explicitly labeled non-equivalent.

Keep SDPA as the current baseline unless profiler evidence strongly contradicts it. Torch profiler wall time is diagnostic only; compare normal `profile_inference` model elapsed time and token throughput.

## Smoke-To-Full Profiling Loop

Start with the middle 30s SALVALAI smoke config:

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

Promote a change to a full-song SALVALAI run only when smoke results are stable, token IDs match, and the speedup is plausibly meaningful. For compile-like changes with one-time first-window costs, inspect post-warmup per-window throughput before rejecting a weak total smoke result. Keep changes that improve RTX 2080 full-song main-generation throughput by about 10% or more. Keep 5-10% wins only when they are simple and well-contained. Remove 1-3% complexity by default.

Stop the long-running optimization loop when either the full-song RTX 2080 run reaches at least `100 tok/s` with identical fixed-seed tokens, or profiling across multiple exact-calculation optimization families shows no remaining plausible `>=10%` improvement.

## Codex Goal Prompt

```text
Optimize Mapperatorinator inference on RTX 2080/2080 Ti for same-calculation speedups only. Current baseline is roughly 65-78 tok/s; first success target is 100 tok/s main-generation throughput, strong milestone is 120 tok/s, and 150+ tok/s is stretch. Do not claim speedups from changed precision, sampling policy, output policy, model quality, windowing/overlap, or generated-token behavior unless explicitly labeled non-equivalent.

Use a profiling-first loop. Start with a middle-30-second song smoke slice, prove fixed-seed generated tokens match baseline, and only promote promising changes to full-song SALVALAI runs. Use full-song runs for accepted results. Keep SDPA as the baseline unless profiler evidence strongly contradicts it. Separate true model time from torch.profiler overhead.

Scout improvement ideas with subagents and web research as useful, but accept only measured wins. Prioritize big wins in generation-loop structure, cache behavior, mask construction, logits processors, repeated small kernels, memory movement, and avoidable per-token setup. Keep changes that improve RTX 2080 full-song main-generation throughput by >=10%; keep 5-10% only if very simple; remove 1-3% complexity. Commit and push clean checkpoints for accepted wins, document why each win worked or failed in docs/inference_profiling.md, update AGENTS.md with durable conventions, write notes under notes/, and stop when the 100 tok/s goal is reached or profiling shows no remaining plausible >=10% exact-calculation improvement.
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

### Dynamic/default cache generation

Attempted in commit `52b8871` and reverted after smoke profiling. The change added an `inference_static_cache=false` path that skipped the preallocated `StaticCache` and let generation use the default/dynamic cache behavior while keeping `inference_generation_compile=true`.

RTX 2080 Ti smoke comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `3e9033c` | `49113275` | `/work/imt11/Mapperatorinator/runs/smoke-compile-49113275-3e9033c/beatmapa00e6b0a1f394a05a1bc95ab28a1dac4.osu.profile.json` | 2,894 | 41.190s | 70.3 |
| candidate | `52b8871` | `49114897` | `/work/imt11/Mapperatorinator/runs/smoke-dyncache-49114897-52b8871/beatmap36600487b37a414b9f239d9a8f6e9586.osu.profile.json` | 1,633 | 23.522s | 69.4 |

`utils/summarize_inference_profile.py --compare` reported token equivalence FAIL: baseline generated `2,894` main-generation token IDs, candidate generated `1,633`, and the first mismatch was at token `0`. This is not an equivalent speed result. Do not use dynamic/default-cache generation for accepted same-calculation claims unless a future implementation proves fixed-seed token equivalence first.

### `torch.inference_mode` generation wrapper

Attempted in commit `02b2437` and reverted after full-song profiling. The change replaced `@torch.no_grad()` with `@torch.inference_mode()` around `model_generate` and `model_forward`.

RTX 2080 Ti full-song comparison on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

| run | commit | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | `3e9033c` | `49113713` | `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json` | 7,639 | 82.615s | 92.5 |
| candidate | `02b2437` | `49115936` | `/work/imt11/Mapperatorinator/runs/full-infermode-49115936-02b2437/beatmapd3e6077d36ea49e795275451d96d09d8.osu.profile.json` | 7,639 | 80.780s | 94.6 |

Smoke profiling looked promising (`+31.9%`), but the accepted full-song comparison was only `+2.3%` with token equivalence PASS for all `7,639` generated main-generation token IDs. This is below the keep threshold, so the change was reverted. Do not reintroduce inference-mode wrapping unless a future full-song result clears the acceptance threshold.

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

Use `wall_seconds - model_elapsed_seconds` to spot server/IPC/queue overhead. Use token counts and `tokens_per_second` to separate model throughput problems from unusually long outputs. When `profile_sync_cuda=true`, profiling requests synchronized model timing inside `model_generate` so `model_elapsed_seconds` includes pending CUDA work. Do not use `wall_seconds` from records with `torch_profiled=true` as normal throughput; the profiler can inflate traced windows by orders of magnitude.

Profile metadata also captures reproducibility context when profiling is enabled: seed, hostname, Slurm job id/partition, git commit/branch, torch/CUDA versions, CUDA device name/capability, and cache/temp environment paths. Report the profile path plus this metadata when accepting or rejecting an optimization.
