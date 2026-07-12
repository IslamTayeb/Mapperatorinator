# Post-270 MLP Breakdown

## Purpose

Size whether a fused/native decoder MLP residual island is worth implementing
after the accepted `270.475 tok/s` exact single-song opt-in baseline.

This is diagnostic-only. No inference speedup graduated.

Baseline entering this pass:

- Job: `49230082`
- Main tokens: `7,639`
- Full-song synchronized main model time: `28.243s`
- Throughput: `270.475 tok/s`
- Equivalence: main/timing token identity PASS and byte-identical `.osu`

For this baseline, a `5%` full-song throughput improvement requires saving
about `1.35s`; the project convention rounds this to a `~1.4s` target before
production native-kernel work is worth carrying.

## Code Change

Extended `utils/profile_decode_decoder_layer_island.py` with MLP component
breakdown variants:

- `mlp_input_norm_only`
- `mlp_fc1_only`
- `mlp_activation_only`
- `mlp_fc2_only`
- `mlp_residual_add_only`
- `mlp_fc2_residual_segment`

These are verifier/profiling variants only. They are not wired into production
inference and do not affect normal generation.

Local checks:

```text
python -m py_compile utils/profile_decode_decoder_layer_island.py
git diff --check
```

## Job 49232374

Run dir:

```text
/work/imt11/Mapperatorinator/runs/post270-mlp-breakdown-20260703-131844-59b622a
```

Environment:

- Branch/commit: `experiment/post270-next-probe`, `59b622a`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
  `GPU-825b182c-b59e-7d16-c8ec-6084dc8199b8`
- Driver: `595.71.05`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Precision/backend: `fp32`, `attn_implementation=sdpa`

Gate result:

- `pass=true`
- logits replay allclose PASS
- logits max abs `0.0`
- captured decoder layers: `12`
- active prefix length: `128`

CUDA graph replay projection from seq9/prefix128:

| component | ms/layer | projected full-song s |
| --- | ---: | ---: |
| repo decoder layer | `0.160515` | `14.5465` |
| manual decoder runtime island | `0.143100` | `12.9683` |
| self-attention residual | `0.047013` | `4.2605` |
| cross-attention residual | `0.036057` | `3.2677` |
| MLP residual | `0.047108` | `4.2691` |

MLP component projections:

| component | ms/layer | projected full-song s |
| --- | ---: | ---: |
| MLP input RMSNorm | `0.004176` | `0.3784` |
| fc1 | `0.020602` | `1.8671` |
| activation | `0.003940` | `0.3571` |
| fc2 | `0.019362` | `1.7547` |
| residual add | `0.003908` | `0.3541` |
| fc2 + residual segment | `0.020358` | `1.8450` |

The first run suggested a target-sized manual-layer saving:

```text
manual island saved 1.578s projected
projected TPS 286.48
```

Because prior post-270 diagnostics had measured the same manual boundary as
flat, this result required a repeat before choosing a production direction.

## Job 49232386

Repeat run dir:

```text
/work/imt11/Mapperatorinator/runs/post270-mlp-repeat-20260703-132132-59b622a
```

The repeat used longer CUDA graph replay timing (`warmup=100`, `iters=2000`)
and ran both the computed active-prefix bucket and explicit prefix `128`.

Summary:

| pass | active prefix | repo ms/layer | manual ms/layer | projected manual saved s | projected MLP residual s |
| --- | ---: | ---: | ---: | ---: | ---: |
| computed | `128` | `0.163379` | `0.154692` | `0.7872` | `4.2310` |
| explicit128 | `128` | `0.163619` | `0.155275` | `0.7561` | `4.2222` |

The repeat kept MLP residual stable at about `4.22s`, but the manual
decoder-runtime island saving dropped below the `~1.4s` production threshold.
Combined with the earlier post-270 job `49232331`, which measured only
`0.038s` saved for the same manual island, this is not stable enough to justify
production work.

## Decision

No optimization graduated.

Keep the MLP component breakdown as profiling infrastructure. It is useful for
future native/CUTLASS/cuBLASLt work because it sizes the exact residual island
instead of repeating rejected individual-linear swaps.

Do not implement a production native MLP residual island from this evidence.
The MLP segment is real (`~4.22s`), but an MLP-only production kernel would need
to save roughly one third of the whole segment to clear the full-song `5%` bar.
The current component split shows the two matrix-vector projections dominate,
but prior individual-linear/native-linear attempts were too small, so any future
MLP work must be a broad fused/native island with a verifier-only projected
saving above `~1.4s` before production integration.

Do not treat the manual Python decoder-layer island as an accepted speed path.
Its measured projected saving has been unstable across jobs:

- `49232331`: `0.038s`
- `49232374`: `1.578s`
- `49232386`: `0.756-0.787s`

This remains diagnostic sizing only. The next plausible major path is still a
broader exact DecodeSession/native decoder-layer runtime that attacks both
decoder compute and the outside-forward runtime/control boundary.
