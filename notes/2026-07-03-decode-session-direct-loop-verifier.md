# DecodeSession Direct-Loop Verifier

## Purpose

Extend `DecodeSession` from a one-token wrapper into a verifier-first state owner for the multi-step direct decode loop. This is infrastructure for future stable-buffer/CUDA-graph/native-kernel work. It is not a production inference path and it is not a speed claim.

## Code

Commits:

- `f5d78ce`: added `--candidate-decode-session` to `utils/verify_direct_decode_loop.py`.
- `c440416`: tightened session metadata, separated prefill/current cache-position metadata, added CUDA RNG state to the session ledger, avoided per-token `.tolist()` sync in session recording, and made `last_token_logits()` clone its fp32 output.

The new verifier flag wraps the existing exact candidate loop in `DecodeSession` while leaving computation unchanged:

- Same `prepare_inputs_for_generation`.
- Same model forward / optional active-prefix context.
- Same CUDA graph replay path.
- Same logits processors.
- Same sampling and RNG consumption.
- Same stopping criteria and output tensor.

Correctness remains enforced by the existing direct-loop gate: generated-token identity, raw-logit allclose, top-k equality, and final CPU+CUDA RNG-state equality against HF `generate()`.

## DCC Gates

Final cleaned validation:

| field | value |
| --- | --- |
| job | `49222209` |
| state | `COMPLETED`, exit `0:0` |
| node | `dcc-core-ferc-s-z25-20` |
| GPU | RTX 2080 Ti `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1` |
| driver | `595.71.05` |
| torch | `2.10.0+cu128` |
| transformers | `4.57.3` |
| commit | `c440416` |
| run dir | `/work/imt11/Mapperatorinator/runs/decode-session-direct-gate-20260703-021358-c440416` |
| report | `direct_gate_64.json` |

Command shape:

```bash
python utils/verify_direct_decode_loop.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --max-new-tokens 64 \
  --candidate-decode-session \
  --candidate-active-prefix-decode \
  --candidate-active-prefix-decode-bucket-size 64 \
  --candidate-cuda-graph-forward \
  --candidate-cuda-graph-warmup 0 \
  --candidate-q1-bmm-cross-attention \
  audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3 \
  use_server=false \
  attn_implementation=sdpa \
  precision=fp32 \
  inference_generation_compile=true \
  profile_record_token_ids=true \
  seed=12345
```

Result:

| check | result |
| --- | --- |
| overall pass | PASS |
| generated-token identity | PASS |
| final RNG state | PASS |
| raw-logit/top-k gate | PASS |
| generated steps | `64` |
| candidate generated tokens | `64` |
| prompt length | `84` |
| output shape | `[1, 148]` |
| prefill cache-position shape | `[84]` |
| current cache-position shape | `[1]` |
| session CUDA RNG state count | `1` |

Earlier validation:

- Job `49222187`, commit `f5d78ce`, passed the same 64-token gate before the metadata cleanup. Keep `49222209` as the final evidence for the cleaned implementation.

## Decision

Keep the verifier infrastructure. It strategically unlocks future DecodeSession work by proving the session wrapper can sit around the exact active-prefix bucket64 + CUDA graph + q1 BMM direct loop without changing tokens, logits, or RNG.

Do not use `--candidate-decode-session` timings as throughput evidence. This path records verifier metadata and intentionally stays outside production inference.

Next useful runtime work:

1. Move more direct-loop state into `DecodeSession` without changing the pass condition.
2. Add stable GPU buffers for one-token input/mask/cache metadata under the same verifier gate.
3. Only after the verifier proves exactness, use the runtime boundary to reduce one-token linear/GEMV launches or q_len=1 attention/cache overhead.
