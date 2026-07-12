# Decoder-Layer Island Probe

## Context

Current accepted exact opt-in baseline remains DCC job `49225493`: full-song SALVALAI `237.111 tok/s`, `7,639` main tokens, `32.217s` synchronized model time, token equivalence PASS for main/timing, and byte-identical `.osu`.

The self-attention, cross-attention, and MLP graph replay probes showed useful but bounded ceilings:

- self-attention island: `~6.075s`
- cross-attention island: `~3.358s`
- one-layer MLP island across 12 layers: `~3.724s`

The next question was how much of main-generation model time is inside the whole one-token decoder layer after including layernorms, residuals, dropout identities, and layer-local glue.

## Utility

Added `utils/profile_decode_decoder_layer_island.py`.

The utility captures the real one-token decoder layer boundary from the direct decode verifier path under active-prefix/native-q1/q1-BMM settings, then benchmarks the captured layer with optional CUDA graph replay. It also times representative RMSNorm calls for scale only. This is diagnostic-only and not an inference throughput claim.

## DCC Result

DCC job `49228458` on commit `df21b90` completed on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133`, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`.

- Run dir: `/work/imt11/Mapperatorinator/runs/decoder-layer-49228458-df21b90`
- Logits replay: PASS, `max_abs=0.0`
- Active prefix length: `128`
- Captured decoder layers: `12`
- Encoder hidden shape: `[1, 1024, 768]`

CUDA graph replay timings:

| component | graph ms/layer |
| --- | ---: |
| full decoder layer | `0.172552ms` |
| self-attn RMSNorm only | `0.004338ms` |
| cross-attn RMSNorm only | `0.004297ms` |
| MLP RMSNorm only | `0.004292ms` |

Full-song projection over `12` layers and `7,552` one-token decode replays:

- Whole one-token decoder layers: `15.637s`, `48.5%` of accepted model time.
- Making every one-token decoder layer free would raise throughput only to about `461 tok/s`.
- The three RMSNorms alone project to about `1.17s` full-song.

## Decision

No optimization graduated.

This is a ceiling/strategy result. It strongly argues against expecting `500 tok/s` from another narrow kernel. Even perfect elimination of the captured one-token decoder layers is not enough for 500 tok/s on the current full-song profile; the target needs near-total decoder-layer acceleration plus at least `~1.3s` outside-layer savings.

The practical implication is:

- Narrow self-attn, cross-attn, MLP, or linears can still produce incremental wins, but they cannot individually be the 500 tok/s path.
- A serious 500 tok/s attempt must be a broader decoder-runtime/backend project: fused decoder-layer or multi-layer decode, better GEMV/GEMM scheduling, cache/layout changes, and output/sampling/window overhead cleanup.
- Before production kernel work, size the outside-layer model time: decoder final norm, output projection, logits processors/sampling, prefill/first-token paths, and any remaining non-layer graph/runtime overhead.

Keep the utility as verifier-first profiling infrastructure. Do not implement a production decoder-layer rewrite until a graph-replayed prototype or native/CUTLASS plan shows a realistic `>5%` full-song gain and preserves generated-token identity.
