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
