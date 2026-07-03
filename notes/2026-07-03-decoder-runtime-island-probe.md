# Decoder Runtime Island Probe

## Hypothesis

After the `270.475 tok/s` fused RoPE/cache baseline, narrow sampling,
graph-cache, and individual linear tweaks are too small. A broader
decoder-layer/runtime island might be the next reusable boundary for native
kernel work, but it should be sized as verifier/profiling infrastructure before
production integration.

## Probe

Added `utils/profile_decode_decoder_layer_island.py --candidate-decoder-runtime-island`.
The flag is diagnostic-only. It adds a manual whole-layer callable that composes
the same self-attention, cross-attention, and MLP module calls as
`VarWhisperDecoderLayer`, then compares it to the repo layer under the existing
CUDA graph replay profiler.

This is not an inference throughput claim. It is a layer-island allclose and
projection check using the real seq9/prefix640 tensors from the 15s SALVALAI
smoke config.

## Results

### Manual whole-layer island

- Job: `49231993`
- Commit: `5d62ec7`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
  `GPU-ba708720-cba9-538c-6e0f-ecaea3486d09`
- Report:
  `/work/imt11/Mapperatorinator/runs/decoder-runtime-island-49231993-5d62ec7/decoder_runtime_island.json`
- Result: PASS, logits replay `max_abs=0.0`, active prefix `640`
- Repo decoder layer projected replay: `15.974386552734375s`
- Manual island projected replay: `15.971617470703125s`
- Projected saved time: `0.0027690820312500364s`
- Projected TPS: `270.50062098250913`

The boundary is valid and effectively identical to the repo layer. It does not
by itself reduce runtime.

### In-place residual variant

- Job: `49232003`
- Commit: `59a3f19`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
  `GPU-8de8e67b-289d-a1ae-3745-25f40db81f55`
- Report:
  `/work/imt11/Mapperatorinator/runs/decoder-inplace-island-49232003-59a3f19/decoder_inplace_island.json`
- Result: PASS, logits replay `max_abs=0.0`, active prefix `640`
- Repo decoder layer projected replay: `18.068206669921874s`
- Manual island projected replay: `18.07352876953125s`
- In-place residual projected replay: `18.114389150390622s`
- In-place residual projected saved time: `-0.0461824804687474s`
- In-place residual projected TPS: `270.0325470795798`

The in-place residual/no-op-dropout variant was exact but slower. It was removed
from the branch.

## Decision

Keep the manual whole-layer island as verifier/profiling infrastructure only.
It gives future native/CUDA/CUTLASS decoder-layer candidates a single exact
replacement boundary and proves the harness overhead is not hiding a Python-call
win. Do not promote it as a speed optimization.

Reject in-place residual rewrites for the current path. They did not improve the
CUDA graph replay projection and should not be wired into production.

## Next

The next plausible `>5%` work still needs to change real decoder-layer compute,
not Python/module organization. Candidate directions remain a fused native MLP
or broader decoder-layer kernel/runtime island, but only if an isolated profiler
probe projects comfortably above the `~1.41s` full-song `5%` threshold before
production integration.
