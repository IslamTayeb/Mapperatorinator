# AGENTS.md

- Keep inference profiling opt-in via `profile_inference`; default inference behavior should not emit profile artifacts.
- Do not commit generated beatmaps, audio files, model weights, or `*.profile.json` outputs.
- For expensive profiling, use a GPU host and the `configs/inference/profile_salvalai.yaml` style of reproducible Hydra config rather than running local Mac inference.
- Preserve server and non-server inference paths when changing profiling code; both should retain prompt, output, generated-token, and elapsed-time stats.
- Keep profiling run instructions and schema notes in `docs/inference_profiling.md` when the workflow changes.
- DCC source paths: repo at `/hpc/group/romerolab/imt11/projects/Mapperatorinator`, env at `/hpc/group/romerolab/imt11/envs/mapperatorinator`, and active data/cache/runs/logs under `/work/imt11/Mapperatorinator`.
- Run expensive DCC work through Slurm only. Use login nodes for git, file inspection, and job submission; capture Slurm stdout/stderr under `/work/imt11/Mapperatorinator/logs` and report job id, GPU node/type, profile paths, and `sacct` status.
- For DCC profiling jobs, put the conda env `bin` directory on `PATH` so `ffmpeg`/`ffprobe` are visible, and keep Hugging Face cache variables consistent between warmup and measured runs.
- For DCC FlashAttention 2 builds, use node-local `/tmp`, `FLASH_ATTN_CUDA_ARCHS=80`, and an A5000/5000 Ada runtime validation; login-node `is_flash_attn_2_available()` can be false because CUDA is unavailable there.
