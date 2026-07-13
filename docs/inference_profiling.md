# Inference Verification

Current conclusions live in `notes/inference-status.md`; decisions live in `notes/inference-experiment-ledger.md`.

## Contract

- Exact means identical token IDs/counts, stopping, RNG, timing/main semantics, request-local state, and final `.osu` bytes.
- Single-song performance is synchronized, untraced main-generation model time.
- Queue claims report scheduler and complete request-to-output wall. Projections and traces are diagnostic only.
- Compare identical commits/configs/cache state in reciprocal launch order.

## Local gate

```bash
.venv/bin/python -m pytest -q tests
```

V32 must keep optimized/native modules cold. Fork controls are `inference_engine=v32|optimized` and `profile_inference=true`.

## DCC gate

Before submission, verify live values rather than copying old Slurm settings:

```bash
hostname
whoami
sinfo -o "%P %D %t %G %m %c"
squeue -u "$USER"
```

Use clean pushed worktrees, one GPU job, `/work` caches, and the stable `/hpc/group` environment.
`scripts/dcc/verify_inference_smoke.sbatch` requires explicit worktrees and commits.

For a direct comparison:

```bash
python utils/summarize_inference_profile.py \
  --compare BASE.profile.json CANDIDATE.profile.json \
  --labels main_generation,timing_context \
  --regression-tolerance-pct 5
```

Record commit, job/GPU, config, cache, wall/TPS, memory, hashes, decision, and revisit condition.
Promote component -> tensors -> loop -> smoke -> full song -> queue; stop on the first exactness, ownership, memory, wall, or 5% failure.
Run Nsight separately from primary timing.
