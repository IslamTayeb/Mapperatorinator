# Native Decoder-Layer Verifier Scope

## Purpose

Define the next implementation scope after the segment-pressure auditor. The
current bottleneck evidence points at broad decoder-layer math/memory work, but
all narrow paths tried so far were too small. This note is a handoff for the
first native verifier candidate, not a throughput claim.

## Scope

First implementation should be profiler-only:

- add an optional native candidate under `utils/profile_decode_decoder_layer_island.py`;
- keep it out of `inference.py`, `server.py`, `config.py`, and production
  generation flags until it clears the CUDA-graph replay stop/go bar;
- expose native helper code through a separate module such as
  `osuT5/osuT5/inference/native_decoder_layer.py`;
- validate only batch-1, q_len=1, fp32, active-prefix decode with `StaticCache`,
  cached cross-attention K/V, no CFG, no beams, no server, no parallel mode.

The candidate should replace multiple adjacent operation classes together:

```text
self-attn norm -> Wqkv -> RoPE/cache write/q1 attention -> Wo -> residual
cross-attn norm -> Wq -> cached q1 attention -> Wo -> residual
final norm -> fc1 -> exact GELU -> fc2 -> residual
```

Do not start with another production path for only one `Linear`, only
`fc1+GELU`, only MLP residual, only self-attention, only cross-attention, graph
cache cleanup, static input copy, logits tail, or sampling. Those have already
been measured below threshold or cannot reach the target alone.

## Files

Candidate files:

| File | Responsibility |
| --- | --- |
| `osuT5/osuT5/inference/native_decoder_layer.py` | default-off native/C++/CUDA/CUTLASS/cuBLASLt diagnostic helper |
| `utils/profile_decode_decoder_layer_island.py` | add `--candidate-native-decoder-layer-island`, benchmark it beside `repo_decoder_layer`, and project CUDA-graph replay savings |
| `utils/validate_decoder_layer_abi.py` | include the candidate in cache-write validation when present |

Avoid production control-plane files until after the verifier clears the bar:

```text
inference.py
osuT5/osuT5/inference/server.py
config.py
```

## Gates

The verifier candidate must pass:

```text
python utils/profile_decode_decoder_layer_island.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --active-prefix-bucket-size 64 \
  --native-q1-rope-cache-self-attention \
  --candidate-native-decoder-layer-island \
  --cuda-graph-replay \
  --include-cache-write-fingerprint \
  --verify-cache-write-candidates \
  ...

python utils/validate_decoder_layer_abi.py REPORT.json \
  --require-cache-write-fingerprint \
  --require-candidate-cache-write-checks
```

Required report fields:

- top-level `pass=true`;
- `logits_replay_allclose=true`;
- candidate hidden output allclose;
- candidate CUDA-graph replay allclose;
- `candidate_cache_write_checks_pass=true`;
- candidate key/value slot SHA256s match the reference.

## Stop/Go

Stop before production wiring unless the candidate saves at least:

```text
~1.35-1.4s projected full-song model time
```

Prefer:

```text
>=2.6-2.8s projected full-song model time
```

If the win appears only in eager timing, breaks CUDA graph replay, changes cache
slot fingerprints, fails logits replay, or saves less than the 5% bar, keep or
remove it as diagnostic code only and document the rejection.

Production integration, if ever justified, still requires:

- one-token logits gate;
- direct-loop token/logit/RNG gate;
- 15s fixed-seed token equivalence;
- full-song SALVALAI token equivalence;
- byte-identical output artifact;
- no meaningful timing-context, stage-wall, or per-window regression.
