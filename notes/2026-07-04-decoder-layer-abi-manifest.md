# Decoder Layer ABI Manifest

## Purpose

Before writing a whole decoder-layer native/CUDA verifier, record the exact
tensor and cache contract that such a candidate must replace. This moves the
500 tok/s work forward without starting another speculative kernel.

This is verifier infrastructure only. It is not a throughput claim.

## Code Change

`utils/profile_decode_decoder_layer_island.py` now emits
`native_decoder_layer_abi` inside each `signature_reports[*]` entry.

The manifest records:

- layer boundary operations expected of a whole-layer native candidate;
- hidden, mask, encoder, cache-position, position-id, and output tensor
  metadata;
- RMSNorm, self-attention, cross-attention, and MLP module weight/bias metadata;
- self/cross `StaticCache` key/value tensor metadata for the captured layer;
- guardrails for batch-1, fp32, static-cache presence, cross-K/V reuse, cache
  write preservation, and output matching.

It intentionally stores shapes, strides, dtype, device, and contiguity only; it
does not serialize model weights, audio, generated maps, or profile JSON outputs
into the repo.

## DCC Validation

Final validation job:

- DCC job: `49250220`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
  `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Commit: `069217a`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/decoder-layer-abi-20260704092623-069217a`
- Report:
  `/work/imt11/Mapperatorinator/runs/decoder-layer-abi-20260704092623-069217a/decoder_layer_abi_report.json`
- Config: `profile_salvalai_smoke15`, sequence `9`, active prefix bucket `64`,
  fused RoPE/cache self-attention enabled, manual decoder runtime island enabled,
  CUDA graph replay enabled, `warmup=10`, `iters=50`
- Required DCC overrides:
  `audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3` and
  `PATH=/hpc/group/romerolab/imt11/envs/mapperatorinator/bin:$PATH` for
  `ffmpeg`/`ffprobe`

Result:

```text
pass=True
logits_replay_allclose=True
logits_replay_max_abs=0.0
captured_decoder_layer_count=12
active_prefix_length=128
signature=hidden1x1x768_encoder1x1024x768_prefix128_mask1x1x1x2560
members=12
representative=transformer.model.decoder.layers.0
abi_schema=1
```

Guardrails:

```json
{
  "batch_size_1": true,
  "cross_kv_expected_reused": true,
  "fp32_hidden_states": true,
  "has_static_cache_layers": true,
  "native_candidate_must_match_output": true,
  "native_candidate_must_preserve_cache_write": true
}
```

Key ABI facts:

| Field | Value |
| --- | --- |
| hidden shape / stride | `[1, 1, 768]` / `[768, 768, 1]` |
| attention mask shape / stride | `[1, 1, 1, 2560]` / `[2560, 2560, 2560, 1]` |
| self heads / head dim | `12` / `64` |
| cross heads / head dim | `12` / `64` |
| self cache keys shape / stride | `[1, 12, 2560, 64]` / `[1966080, 163840, 64, 1]` |
| cross cache keys shape / stride | `[1, 12, 1024, 64]` / `[786432, 65536, 64, 1]` |
| cache types | `MapperatorinatorCache`, `StaticCache`, `StaticCache` |

## Failed Setup Attempts

- Job `49250201` failed before model work because the DCC run missed
  `audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3`.
- Job `49250209` fixed the audio path but failed before model work because
  batch `PATH` did not include the env `ffprobe`.

These are environment/setup failures, not verifier failures.

## Interpretation

The manifest gives the next native whole-layer verifier an explicit ABI. A
candidate should be rejected before CUDA work if it cannot satisfy these
guardrails, especially fp32 batch-1 static cache, exact output matching, and
self-cache write preservation.

This does not change the current speed conclusion: the accepted full-song
baseline remains `270.475 tok/s`, and a whole decoder-layer native experiment
still needs CUDA-graph replay savings before any production integration.
