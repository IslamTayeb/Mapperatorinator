# Weighted Decoder Stack Roofline

## Purpose

Refresh the theoretical minimum check using the actual full-song active-prefix
bucket distribution, not only one representative decoder-layer report. This
keeps the 500 tok/s work pointed at the current bottleneck and separates
potentially avoidable runtime/kernel headroom from necessary decoder math and
memory traffic.

This is diagnostic-only. It is not an inference throughput claim.

## Utility

Added `utils/summarize_weighted_decoder_stack_roofline.py`.

Inputs:

- weighted decoder-stack summary from `utils/profile_decode_decoder_stack_island.py`;
- optional decoder-layer ABI report to infer `D`, FFN size, encoder length, and
  decoder layer count;
- nominal RTX 2080 Ti fp32 peak and memory bandwidth assumptions.

The script sums per-prefix full-song replay counts, computes a lower-bound
FLOP/byte model for each one-token decoder-stack component, then reports compute
floor, bandwidth floor, above-floor headroom, and whether the weighted stack
roofline alone can hit the target.

## Validation

Run:

```text
/work/imt11/Mapperatorinator/runs/weighted-stack-roofline-20260704135313-721b4c3
```

Command:

```text
python3 utils/summarize_weighted_decoder_stack_roofline.py \
  /work/imt11/Mapperatorinator/runs/decoder-stack-weighted-49233687/weighted_decoder_stack_summary.json \
  --decoder-layer-report /work/imt11/Mapperatorinator/runs/decoder-layer-cache-fingerprint-20260704134048-7f633ef/decoder_layer_abi_fingerprint_report.json \
  --json-output /work/imt11/Mapperatorinator/runs/weighted-stack-roofline-20260704135313-721b4c3/weighted_decoder_stack_roofline.json
```

Output summary:

```text
Weighted Decoder Stack Roofline Summary
  accepted baseline: 7639 tokens, 28.243s, 270.474 tok/s
  target: 500.0 tok/s requires 15.278s (12.965s saved)
  measured weighted decoder stack: 16.666s (59.0% of model time)
  stack-only target pressure: 77.8% of measured stack replay would need to disappear
  roofline floors: compute=0.145s, bandwidth=6.348s, floor=6.348s (bandwidth)
  above-floor headroom: 10.319s, model_at_floor=17.924s, tps_at_floor=426.179
  decision: verifier_only_go=True stack_roofline_reaches_target=False
```

Assumptions:

| Field | Value |
| --- | ---: |
| model dim | `768` |
| FFN dim | `3072` |
| encoder length | `1024` |
| decoder layers | `12` |
| decoder heads / head dim | `12` / `64` |
| weighted decode replays | `7,552` |
| prefix buckets | `11` |
| peak fp32 | `13.45 TFLOP/s` |
| peak bandwidth | `616 GB/s` |

Largest component bandwidth floors:

| Component | Bandwidth floor |
| --- | ---: |
| `mlp_fc1` | `1.392s` |
| `mlp_fc2` | `1.391s` |
| `self_qkv_linear` | `1.044s` |
| `cross_q1_attention` | `0.927s` |
| `self_rope_cache_attention` | `0.537s` |
| `self_out_linear` | `0.348s` |
| `cross_q_linear` | `0.348s` |
| `cross_out_linear` | `0.348s` |

## Interpretation

The weighted stack still has target-sized above-floor headroom:

- measured weighted decoder stack: `16.666s`;
- optimistic roofline floor: `6.348s`;
- above-floor headroom: `10.319s`.

That is enough to justify a verifier-only broad native decoder-layer/stack
math-memory candidate if one is designed carefully and passes the existing ABI,
cache-write fingerprint, logit, token, RNG, and output gates.

It is not enough to claim that decoder-stack work alone reaches `500 tok/s`.
Even if the whole weighted stack hit this optimistic bandwidth roofline and
everything outside the stack stayed the same, full-song SALVALAI would be about
`426.179 tok/s`, still short of the `500 tok/s` target.

So the stop/go rule becomes sharper:

1. Do not run more small wrapper, tail, sampling, graph-cache, per-linear,
   MLP-only, or self-attention-only experiments from the current evidence.
2. The next implementation-worthy work is still a whole-layer or whole-stack
   native math/memory verifier, but it must project at least the normal
   `~1.35-1.4s` 5% full-song saving before production integration.
3. A true `500 tok/s` path likely needs both broad decoder stack improvement and
   additional runtime/control or architecture-level savings; reaching the
   weighted stack roofline alone projects only `426 tok/s`.
