# Inference Experiment Runbook

This file contains procedure, not research history. Current conclusions live in
[`notes/inference-status.md`](../notes/inference-status.md); decisions and
revisit conditions live in the
[`experiment ledger`](../notes/inference-experiment-ledger.md).

## Metrics

- Single song: synchronized, untraced main-generation model time.
- Offline queue: aggregate main tokens divided by direct wall from first main start to last main finish.
- Traces, CUDA events, projections, and component timings are diagnostic; they are not end-to-end throughput.
- Keep cold, warm, serial queue, offline batch, and server measurements separate.

An exact result preserves tokens, stopping, RNG, timing/main semantics,
request-local mutable state, and final `.osu` bytes. Internal FP32 allclose is
allowed only when the claim is explicitly `exact-output`; observable drift is
reported separately.

## Local checks

Run the smallest relevant tests first, then the full affected suite:

```bash
.venv/bin/python -m pytest
```

Default and compile-only V32 checks must confirm that optimized/native modules
and extension machinery remain unloaded.

## DCC preflight

Use the login node only for inspection, Git, and submission:

```bash
hostname
whoami
git status --short
sinfo -o "%P %D %t %G %m %c"
squeue -u "$USER"
```

Verify the account, partition, and exact GPU GRES live. Use one GPU job at a
time. Every experiment branch gets a persistent DCC worktree, and every Slurm
wrapper receives that checkout explicitly.

Canonical locations:

```bash
REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
WORK=/work/imt11/Mapperatorinator

export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
```

Keep model, compiler, extension, graph, and Hugging Face cache state identical
within a comparison. Push the exact commit before submission and make wrappers
fail if their checkout or commit does not match.

## Promotion ladder

Before implementation, write down:

1. the measured bottleneck;
2. the avoidable fraction and physical/fantasy floor;
3. the projected end-to-end ceiling;
4. the result that would falsify the idea.

Then advance one boundary at a time:

1. component measurement;
2. real-tensor logits/cache verifier;
3. short, then 256-step token/logit/RNG loop;
4. 15-second smoke;
5. reciprocal full-song comparison;
6. direct multi-song queue wall.

A lower gate authorizes only the next gate. Stop immediately when exactness,
ownership, memory, direct wall, or the `5%` signal fails.

## Required record

For every retained result, record:

- commit, branch, Slurm job/status, node, and GPU;
- config, flags, seed, precision, backend, and execution mode;
- model/compiler/extension/graph cache state;
- untraced wall/TPS, tokens, memory, and useful diagnostics;
- token/RNG/state and final output equivalence;
- artifact paths and hashes;
- accept/reject decision and the evidence required to revisit it.

Use `utils/summarize_inference_profile.py --compare BASE CANDIDATE
--strict-full-song` for full-song comparisons. Run tracing separately from
primary timing. Never use stale server sockets, sample stopped/dummy rows, or
share mutable request state in an exact batch claim.
