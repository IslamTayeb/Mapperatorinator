# Decoder Layer Roofline

## Purpose

Add a physical-floor check before any broad native decoder-layer work. The
existing pressure audit showed the broad decoder-layer replay boundary is the
only current 500-capable idealized boundary, but that did not answer whether
the measured layer cost is avoidable or necessary math/data movement.

This note is diagnostic only. It is not an inference throughput claim.

## Utility

Added `utils/summarize_decoder_layer_roofline.py`.

The utility reads a `utils/profile_decode_decoder_layer_island.py` report and
estimates an optimistic roofline floor for the captured one-token decoder-layer
bucket. It is parameterized by `ffn_dim`, `decoder_heads`, dtype bytes, peak
fp32 TFLOP/s, memory bandwidth, and whether cross-attention K/V is recomputed.

Default hardware assumptions are nominal RTX 2080 Ti values:

- `13.45` fp32 TFLOP/s;
- `616` GB/s memory bandwidth;
- fp32, `4` bytes per element.

## Validation Command

```text
python3 utils/summarize_decoder_layer_roofline.py \
  /tmp/mapperatorinator_decoder_layer_seq9.json \
  --json-output /tmp/mapperatorinator_decoder_layer_roofline.json
```

## Result

Input report: `/tmp/mapperatorinator_decoder_layer_seq9.json`, copied from the
current decoder-layer replay artifacts.

Key inferred shape:

- `D=768`;
- `ffn_dim=3072` by default (`4 * D`);
- `12` decoder layers;
- active self-attention prefix `128`;
- encoder length `1024`;
- full-song decode replays `7,552`.

Summary:

| Item | Value |
| --- | ---: |
| accepted baseline | `7,639` tokens, `28.243s`, `270.474 tok/s` |
| target | `500 tok/s`, `15.278s`, `12.965s` saved |
| measured decoder-layer replay | `13.018s` |
| layer-only fraction that must disappear for 500 | `99.6%` |
| compute floor | `0.135s` |
| bandwidth floor | `5.928s` |
| optimistic roofline floor | `5.928s`, bandwidth-limited |
| removable above floor | `7.090s` |
| model time if layer reached roofline and everything else stayed | `21.153s` |
| TPS if layer reached roofline and everything else stayed | `361.132 tok/s` |

## Interpretation

The result keeps one bounded whole-layer verifier alive: there is theoretical
headroom above a nominal bandwidth floor, and it is above the 5-10% keep bars.

It also rules out a too-optimistic story. Broad decoder-layer work alone does
not reach `500 tok/s` under this roofline; even an optimistic peak-bandwidth
layer floor leaves the model at about `361 tok/s`. Hitting `500 tok/s` would
need decoder-layer improvement plus additional production/runtime or
architecture-level changes, or a better-than-this roofline assumption.

Do not start narrow kernels from this. The only justified implementation-class
experiment remains a verifier-only whole decoder-layer math/memory island. It
must show CUDA-graph replay savings before production wiring, and any production
candidate still needs the normal exact logits, token, RNG, full-song throughput,
and output-byte gates.
