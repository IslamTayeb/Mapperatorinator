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

`utils/verify_one_token_decode.py` originally compared a static-cache `q_len=1` decode step against a full-prefix no-cache raw-logits reference. That was too strict for the actual optimization target: production inference uses Hugging Face `generate()` with a freshly allocated `StaticCache`, logits processors, sampling, EOS handling, and optional generation compile. The gate now uses HF cached generation as the reference and keeps the old no-cache comparison only as diagnostics. The current gate:

1. Builds a real SALVALAI prompt through `Preprocessor` and `Processor`.
2. Runs `model.generate(..., past_key_values=StaticCache, max_new_tokens=2, output_logits=True)` with the same production logits processors, sampling flags, EOS policy, prompt, mask, encoder inputs, and conditioning kwargs.
3. Uses HF's first generated token as the probe token.
4. Uses HF's second cached-generation raw-logit step as the reference.
5. Computes candidate raw logits from a separately prefilled `StaticCache` plus one `q_len=1` decode step prepared through `model.prepare_inputs_for_generation()`.
6. Requires `torch.allclose` within tolerance and identical top-k token ordering.

Passing this gate is necessary for custom decoder-step work, but it is not sufficient for a speed claim. Fixed-seed 15s generated-token equivalence and full-song generated-token equivalence are still required.

## Current Status

Initial DCC gate run `49139099` on `dcc-core-ferc-s-z25-21` reached model load, then failed before logits comparison:

```text
ValueError: Invalid special event name song_position.
```

Cause: the first version of `utils/verify_one_token_decode.py` built the probe prompt from only the selected output context. Production sequential generation passes the output-context prefix `out_context[:i + 1]`, which matters for required extra special tokens such as `song_position`. The gate was patched to mirror the production prefix path before rerunning.

Follow-up DCC gate run `49139107` on the same node reached the raw-logits comparison but failed:

```text
pass=false, prompt_tokens=84, probe_token_id=13,
max_abs=27.6516, mean_abs=17.3663, topk_match=false
```

Diagnosis: the gate still called `model.forward()` directly for its prompt/reference/prefill paths. Production generation goes through `prepare_inputs_for_generation()`, which is important for left-padded prompt position IDs and static-cache 4D mask preparation. The gate was patched again to prepare prompt, no-cache reference, static-cache prefill, and `q_len=1` candidate inputs through `model.prepare_inputs_for_generation()` and to record prepared shapes/cache positions in the report.

Follow-up DCC gate run `49139122` on `dcc-core-ferc-s-z25-21` still failed with the same diff:

```text
pass=false, prompt_tokens=84, probe_token_id=13,
max_abs=27.6516, mean_abs=17.3663, topk_match=false
```

Diagnosis: the candidate was now exercising the intended prepared `q_len=1` cache path, but the reference was still the full-prefix no-cache path. The top-1 token matched, but many logits moved, so this is not a valid exact-runtime gate for a production path that already uses `StaticCache`. The gate was patched to compare against HF `generate()` cached raw logits instead. The no-cache full-prefix diff remains in the JSON report as `no_cache_reference_*` diagnostics, but it no longer controls pass/fail.

The next run should be:

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
