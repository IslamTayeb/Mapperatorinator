# Nsight Compute Decoder-Stack Permission Probe

## Purpose

Check whether DCC RTX 2080 Ti jobs can collect Nsight Compute hardware counters for the accepted graph-replayed decoder-stack path.

This was a tooling/permission probe, not an inference speed result.

## Run

- Job: `49232763`
- Commit: `7a74b53`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1`
- Run dir:

```text
/work/imt11/Mapperatorinator/runs/ncu-decoder-stack-20260703-141345-7a74b53
```

Command shape:

```bash
/opt/apps/rhel9/cuda-12.8/bin/ncu \
  --target-processes all \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --launch-skip 200 \
  --launch-count 80 \
  --csv \
  --log-file "$RUN/ncu_decoder_stack.csv" \
  --export "$RUN/ncu_decoder_stack" \
  python utils/profile_decode_decoder_stack_island.py ...
```

The first attempt, job `49232752`, failed immediately because `module` was unavailable in the batch shell and the script relied on `module load CUDA/12.8`. The rerun used the absolute NCU path.

## Result

NCU launched and connected to the Python process, but hardware counter access failed:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

Slurm state:

```text
49232763|FAILED|00:01:48|dcc-core-ferc-s-z25-20|1:0
```

The decoder-stack utility still wrote `decoder_stack_island.json`, but those timings were heavily profiler-perturbed and should not be used. Example graph replay times inflated to about `48ms/call`, far from the normal validated `~1.85ms/call`.

## Decision

Do not retry Nsight Compute hardware-counter profiling on these DCC RTX 2080 Ti nodes unless counter permissions are enabled or a node/partition with accessible counters is identified.

Use these alternatives meanwhile:

- Nsight Systems with NVTX ranges for timeline/kernel names;
- torch profiler for diagnostic op/kernel attribution, not throughput;
- CUDA-event microbench utilities for target sizing;
- untraced `profile_inference` for throughput claims.
