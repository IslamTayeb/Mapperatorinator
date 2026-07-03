# Model Generate CUDA Ledger

## Hypothesis

Before starting more native runtime/kernel work, check whether the gap between
the accepted `270.475 tok/s` path and a `500 tok/s` target is mostly host/runtime
control around `model.generate()` or real queued CUDA decoder work.

## Change

Added diagnostic `model.generate()` CUDA-event ledger fields:

- `model_generate_cpu_elapsed_seconds`
- `model_generate_cuda_event_seconds`
- `model_generate_host_gap_seconds`

These are now behind `profile_model_generate_cuda_ledger=true`. The flag is
default-off and validated through `inference.py`; runtime wiring flows through
`processor.py` into `server.py:model_generate()`.

## Evidence

- Job: `49234401`
- Repeat job: `49234699`
- GPU: RTX 2080 Ti, `GPU-70edd336-9ffb-9a85-c0ee-c04d77b76094`
- Node: `dcc-gehmlab-gpu-ferc-s-z25-19`
- Commit: `a6309f88b1cea4ef1d3d0f8caaea102c66ce8181`
- Candidate profile:
  `/work/imt11/Mapperatorinator/runs/generate-ledger2-49234699/candidate.profile.json`
- Strict compare:
  `/work/imt11/Mapperatorinator/runs/generate-ledger2-49234699/compare_strict_full.txt`

Repeat full-song SALVALAI main generation:

| Metric | Value |
| --- | ---: |
| Generated main tokens | `7,639` |
| Main model time | `27.855077s` |
| Main tok/s | `274.240845` |
| CUDA-event time inside `model.generate()` | `27.847836s` |
| Host gap | `0.007241s` |
| CUDA-event fraction | `0.999740` |
| Main token equivalence | PASS, `7,639 / 7,639` |

Timing context in the same repeat:

| Metric | Value |
| --- | ---: |
| Generated timing tokens | `821` |
| Timing model time | `8.116974s` |
| Timing tok/s | `101.146068` |
| CUDA-event time inside `model.generate()` | `8.109725s` |
| Host gap | `0.007249s` |
| Timing token equivalence | PASS, `821 / 821` |

Strict comparison against the retained accepted profile failed because:

- output artifact equivalence could not be checked; the older baseline profile
  lacks output hash fields;
- main per-window no-regression failed on two tiny windows despite better
  aggregate main time;
- timing-context aggregate regressed slightly (`101.988 -> 101.146 tok/s`) and
  failed several strict per-window checks.

## Interpretation

For main generation, the current fastest exact stack is not meaningfully waiting
on Python host work outside `model.generate()`: only about `7ms` of `27.855s`
is outside the CUDA-event interval. That rejects host-loop cleanup as a major
path to `500 tok/s`.

The next plausible work should target broad decoder CUDA compute/native runtime
boundaries: self-attention residual, cross-attention residual, MLP residual, and
the generated-token control boundary together. Narrow sampling, graph-input
copying, or wrapper cleanup is not target-sized from this evidence.

## Decision

Keep the ledger as default-off diagnostic infrastructure only. Do not count it
as an optimization. Normal throughput claims should keep
`profile_model_generate_cuda_ledger=false` unless the run is explicitly for
attribution.
