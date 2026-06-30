# AGENTS.md

- Keep inference profiling opt-in via `profile_inference`; default inference behavior should not emit profile artifacts.
- Do not commit generated beatmaps, audio files, model weights, or `*.profile.json` outputs.
- For expensive profiling, use a GPU host and the `configs/inference/profile_salvalai.yaml` style of reproducible Hydra config rather than running local Mac inference.
- Preserve server and non-server inference paths when changing profiling code; both should retain prompt, output, generated-token, and elapsed-time stats.
- Keep profiling run instructions and schema notes in `docs/inference_profiling.md` when the workflow changes.
- For DCC profiling jobs, put the conda env `bin` directory on `PATH` so `ffmpeg`/`ffprobe` are visible, and keep Hugging Face cache variables consistent between warmup and measured runs.
