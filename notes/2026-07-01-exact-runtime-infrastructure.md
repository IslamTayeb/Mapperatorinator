# Exact Runtime Infrastructure Checkpoint

## Summary

Added the first PyTorch-first infrastructure needed for renewed `200 tok/s` runtime work. This checkpoint is not a speed win and should not be reported as one. It adds gates and profiling controls so future decoder/runtime experiments can be measured without relaxing same-calculation policy.

## What Changed

- Added opt-in detailed generation ranges via `profile_generation_detail_ranges=true`.
- Added SDPA backend forcing via `profile_sdpa_backend=flash|efficient|math|cudnn` for dispatch audits.
- Added `utils/verify_one_token_decode.py`, a raw-logits gate comparing full-prefix logits against the static-cache `q_len=1` decode step on the real inference prompt path.
- Extended profile metadata and `utils/summarize_inference_profile.py --compare` so candidate runs print a same-calculation metadata contract before token-equivalence output.

## Why This Exists

The retained baseline is still SDPA plus `inference_generation_compile=true`: job `49113713`, `7,639` full-song SALVALAI main tokens, `82.615s` synchronized model time, `92.465 tok/s`, fixed-seed token equivalence PASS.

The post-warmup trace showed the compiled one-token forward, f32 SDPA kernels on SM75, and many one-token GEMV/GEMM launches dominate. Sampling/logits work was not target-sized. That means the next plausible path needs to attack the real decoder step, but any rewrite needs a logits gate before full generated-token equivalence testing.

## Gate Semantics

`utils/verify_one_token_decode.py` deliberately avoids `generate()` as its logits reference. A generation call can consume sampling RNG and mutate caches before the real test. Instead, the gate:

1. Builds a real SALVALAI prompt through `Preprocessor` and `Processor`.
2. Chooses a deterministic probe token from the full-prompt raw logits unless `--probe-token-id` is provided.
3. Computes reference raw logits from a full-prefix no-cache forward.
4. Computes candidate raw logits from a prefilled `StaticCache` plus one `q_len=1` decode step.
5. Requires `torch.allclose` within tolerance and identical top-k token ordering.

Passing this gate is necessary for custom decoder-step work, but it is not sufficient for a speed claim. Fixed-seed 15s generated-token equivalence and full-song generated-token equivalence are still required.

## Current Status

No DCC GPU runs have been made for this checkpoint yet. The next run should be:

```bash
python utils/verify_one_token_decode.py \
  --config-name profile_salvalai_smoke15 \
  --report-path "$WORK/runs/one-token-decode-${SLURM_JOB_ID}.json" \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/profile-smoke15-${SLURM_JOB_ID}" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1
```

Then run a diagnostic trace with `profile_generation_detail_ranges=true` and, separately, SDPA backend audit smokes with `profile_sdpa_backend=flash`, `efficient`, and `math`. Use untraced `profile_inference` runs for throughput claims.
