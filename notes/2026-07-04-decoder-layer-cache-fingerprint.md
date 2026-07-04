# Decoder Layer Cache-Write Fingerprint Gate

## Purpose

Tighten the whole decoder-layer ABI before any native/CUDA implementation work.
Output allclose is not enough for a replacement layer because a candidate could
return the right hidden state for one token while corrupting the `StaticCache`
slot needed by the next token.

This is verifier infrastructure only. It is not an inference throughput claim.

## Code Change

`utils/profile_decode_decoder_layer_island.py` now accepts:

```text
--include-cache-write-fingerprint
```

When enabled, each representative `native_decoder_layer_abi` entry records a
SHA256 fingerprint and tensor metadata for the q_len=1 self-attention K/V cache
slot written at `cache_position`. It hashes only the slot view
`keys[:, :, cache_position:cache_position + 1, :]` and the matching value slot,
not the full cache.

`utils/validate_decoder_layer_abi.py` now accepts:

```text
--require-cache-write-fingerprint
```

That strict mode requires the fingerprint metadata to be present, verifies the
reported cache position, active prefix, cache length, slot shape, dtype, CUDA
metadata, byte count, and SHA256 format, and keeps old reports valid when the
flag is not requested.

## Validation

DCC job:

```text
49250272
```

Run details:

| Field | Value |
| --- | --- |
| node/GPU | `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133` |
| branch/commit | `codex/decoder-layer-cache-fingerprint`, `7f633ef` |
| run dir | `/work/imt11/Mapperatorinator/runs/decoder-layer-cache-fingerprint-20260704134048-7f633ef` |
| report | `/work/imt11/Mapperatorinator/runs/decoder-layer-cache-fingerprint-20260704134048-7f633ef/decoder_layer_abi_fingerprint_report.json` |
| validation | `/work/imt11/Mapperatorinator/runs/decoder-layer-cache-fingerprint-20260704134048-7f633ef/decoder_layer_abi_fingerprint_validation.json` |
| Slurm state | `COMPLETED`, elapsed `00:02:06`, `gres/gpu:2080=1` |
| config | `profile_salvalai_smoke15`, sequence `9`, bucket `64`, fp32, SDPA, fused RoPE/cache self-attention |

Command shape:

```text
python utils/profile_decode_decoder_layer_island.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --active-prefix-bucket-size 64 \
  --native-q1-rope-cache-self-attention \
  --candidate-decoder-runtime-island \
  --cuda-graph-replay \
  --warmup 10 \
  --iters 50 \
  --include-cache-write-fingerprint \
  --report-path /work/imt11/Mapperatorinator/runs/decoder-layer-cache-fingerprint-20260704134048-7f633ef/decoder_layer_abi_fingerprint_report.json \
  audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3

python utils/validate_decoder_layer_abi.py \
  /work/imt11/Mapperatorinator/runs/decoder-layer-cache-fingerprint-20260704134048-7f633ef/decoder_layer_abi_fingerprint_report.json \
  --require-cache-write-fingerprint \
  --json-output /work/imt11/Mapperatorinator/runs/decoder-layer-cache-fingerprint-20260704134048-7f633ef/decoder_layer_abi_fingerprint_validation.json
```

Result:

```text
pass=True
logits_replay_allclose=True
logits_replay_max_abs=0.0
validation_pass=True
failures=0
warnings=0
signature=hidden1x1x768_encoder1x1024x768_prefix128_mask1x1x1x2560
members=12
representative=transformer.model.decoder.layers.0
cache_position=[84]
active_prefix_length=128
max_cache_len=2560
cache_fingerprint=True
```

Cache slot facts:

| Field | Value |
| --- | --- |
| key slot shape / stride | `[1, 12, 1, 64]` / `[1966080, 163840, 64, 1]` |
| value slot shape / stride | `[1, 12, 1, 64]` / `[1966080, 163840, 64, 1]` |
| dtype/device | `torch.float32`, CUDA |
| storage offset | `5376` |
| key SHA256 | `8a5e5d554cd8ad876f463c6d846235e071e2a43174589599b53a599be5efa6e5` |
| value SHA256 | `6118158b057dd715c552038881e1993180207aafe2621ec52a2e1a1fe3e59f07` |

## Bottleneck Interpretation

This keeps the whole-layer native verifier path alive only as correctness
infrastructure. It does not make the current manual whole-layer replacement a
speed path:

| Replay boundary | Projected full-song seconds |
| --- | ---: |
| repo decoder layer | `15.768982s` |
| manual decoder runtime island | `15.697874s` |
| projected saving | `0.071108s` |
| projected TPS | `271.157 tok/s` |

The saving is far below the `5%` keep bar for production/runtime complexity.
Future whole-layer work must improve real native math or memory behavior, pass
this cache-write fingerprint gate, and project at least `~1.35-1.4s` full-song
synchronized model-time saving before production integration is worth trying.
