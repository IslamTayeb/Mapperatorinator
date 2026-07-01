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

Validated on DCC `gpu-common` with `--gres=gpu:2080:1`; the allocated node exposed an `NVIDIA GeForce RTX 2080 Ti`.

Observed full SALVALAI run on 2026-06-30:

- Slurm elapsed: 3m20s for a 156.992s MP3.
- Sequence count: 87.
- Map generation: 137.324s stage wall, 8,816 generated tokens, 64.6 tok/s.
- Timing generation: 21.094s stage wall, 821 generated tokens, 39.5 tok/s.
- Model load: 10.152s main model, 8.489s timing model from cache.
- Audio load and postprocessing were sub-second to low-subsecond and not the bottleneck.

The dominant bottleneck is autoregressive map generation. The first map window was the slowest record in this run at 7.655s for 520 generated tokens; most other slow windows were around 2.0-2.4s.

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
  profile_torch_generation_limit=3
```

This writes Chrome traces for the first selected `model.generate` calls and adds a `torch_profiles` section to the normal `.profile.json`. Keep `profile_torch_generation_limit` small for full songs; each trace includes CPU/CUDA activities, shapes, memory events, NVTX/record-function ranges, and the top profiler events by self CUDA time.

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
- CUDA memory counters when CUDA is available

Use `wall_seconds - model_elapsed_seconds` to spot server/IPC/queue overhead. Use token counts and `tokens_per_second` to separate model throughput problems from unusually long outputs.
