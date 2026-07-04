# Decoder Layer ABI Validator

## Purpose

Turn the native decoder-layer ABI manifest into an executable gate before any
whole-layer native/CUDA work. This keeps future kernel work tied to the measured
bottleneck and rejects unsupported shapes or missing exactness metadata before
implementation.

This is verifier infrastructure only. It is not an inference throughput claim.

## Utility

Added `utils/validate_decoder_layer_abi.py`.

The validator reads a `utils/profile_decode_decoder_layer_island.py` JSON report
and checks:

- top-level logits replay exactness metadata;
- `native_decoder_layer_abi.schema_version == 1`;
- required guardrails are true: batch-1, fp32 hidden states, static cache
  layers, cross-K/V reuse, output matching, and cache-write preservation;
- hidden/output/encoder/mask/cache-position tensor shape, dtype, CUDA,
  contiguity, and last-stride assumptions;
- self-attention/cross-attention head layout and linear dimensions;
- RMSNorm and MLP dimensions;
- self/cross `StaticCache` key/value layout and initialization.

It intentionally validates metadata only. It does not load model weights,
deserialize tensors, or claim speed.

## Validation

Report copied from DCC job `49250220`:

```text
/work/imt11/Mapperatorinator/runs/decoder-layer-abi-20260704092623-069217a/decoder_layer_abi_report.json
```

Command:

```text
ssh dcc 'cat /work/imt11/Mapperatorinator/runs/decoder-layer-abi-20260704092623-069217a/decoder_layer_abi_report.json' \
  > /tmp/mapperatorinator_decoder_layer_abi_report.json

python3 -m py_compile utils/validate_decoder_layer_abi.py
python3 utils/validate_decoder_layer_abi.py \
  /tmp/mapperatorinator_decoder_layer_abi_report.json \
  --json-output /tmp/mapperatorinator_decoder_layer_abi_validation.json
```

Output:

```text
Decoder Layer ABI Validation
  pass: True
  signatures: 1
  failures: 0
  warnings: 0
  hidden1x1x768_encoder1x1024x768_prefix128_mask1x1x1x2560: pass=True members=12 D=768 ffn=3072 heads=12 prefix=128 encoder=1024
```

## Interpretation

Future whole-layer native work should run this validator before CUDA coding or
DCC timing. Failing the validator means the candidate is outside the current
accepted simple batch-1 fp32 static-cache DecodeSession path and must either be
re-scoped or documented as a separate non-equivalent/unsupported experiment.

Passing this validator is necessary but not sufficient for a speed claim. A
candidate still needs one-token logits, direct-loop token/logit/RNG, 15s smoke,
full-song token/output equivalence, and untraced throughput gates.
