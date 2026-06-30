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
