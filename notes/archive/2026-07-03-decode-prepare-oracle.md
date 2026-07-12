# Decode Prepare Oracle

## Summary

Add and validate a verifier-only fast one-token decode input builder. This checks whether a future `DecodeSession` runtime can bypass the full HF-style `prepare_inputs_for_generation` call during post-prefill decode while producing the exact same prepared tensors.

This is not a throughput claim and not a production optimization yet.

## Infrastructure

- Commit: `8cb1fd6`
- Helper: `osuT5.osuT5.inference.direct_decode.prepare_one_token_decode_inputs_fast`
- Verifier: `utils/verify_decode_prepare_oracle.py`

The helper is intentionally conservative:

- batch-1 only;
- cached post-prefill one-token decode only;
- preserves wrapper-level extra kwargs such as `frames`;
- preserves cache and encoder object identity;
- still constructs the full static-cache 4D causal mask through the existing model helper;
- does not change sampling, logits processors, EOS/stopping, cache mutation, generated tokens, or model compute.

## DCC Result

- Passing job: `49231765`
- Commit: `8cb1fd6`
- Node: `dcc-core-ferc-s-z25-20`
- GPU: RTX 2080 Ti, UUID `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1`
- Run dir: `/work/imt11/Mapperatorinator/runs/decode-prepare-oracle-49231765-8cb1fd6`
- Report: `/work/imt11/Mapperatorinator/runs/decode-prepare-oracle-49231765-8cb1fd6/prepare_oracle.json`
- Config: `profile_salvalai_smoke15`, sequence `9`, `max_new_tokens=32`, fp32, SDPA, generation compile enabled, stateful monotonic enabled, q1 BMM cross-attention enabled, native q1 self-attention enabled, fused RoPE/cache self-attention enabled

Result:

- PASS
- Prompt tokens: `84`
- Checked decode steps: `31`
- Failed comparisons: `0`
- Total verifier wall: `35.041s`
- Diagnostic HF prepare CPU wall: `0.016127s`
- Diagnostic fast-builder CPU wall: `0.009144s`

The first checked step compared these keys and all matched:

- `cache_position`
- `decoder_attention_mask`
- `decoder_input_ids`
- `decoder_position_ids`
- `encoder_outputs`
- `frames`
- `past_key_values`
- `use_cache`

## Failed Setup Attempts

- Job `49231686` failed before model work because the sbatch missed `audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3`.
- Job `49231700` failed because `ffprobe` was not on `PATH`.
- Job `49231711` failed because the first manual prefill used `input_features` instead of wrapper-level `frames`.
- Job `49231743` produced a full non-passing report because the helper used `model.proj_out` on the wrapper instead of `model.get_output_embeddings()`.

These were verifier harness issues, not candidate-equivalence failures.

## Interpretation

This validates the exact tensor-preparation ABI for a small runtime slice. It does not prove speed, because the verifier still calls the HF prepare path, the candidate path, and the full model forward for reporting.

The diagnostic wall delta over 31 checked decode steps is about `6.98ms`, or roughly `225us` per checked step. If this scaled perfectly over the accepted `7,552` full-song decode steps, the ceiling would be around `1.7s`, near the current `5%` keep threshold for the `28.243s` accepted fused-stack model time. That projection is only a target-size estimate; any speed claim needs an actual untraced `profile_inference` run.

## Next Step

The next reasonable experiment is a default-off active-prefix runtime flag that uses the fast prepared-input builder only after prefill, only in the same simple batch-1/non-server/active-prefix graph stack. Graduation gates:

1. `utils/verify_direct_decode_loop.py` with generated-token, raw-logit/top-k, and final RNG-state equality.
2. `profile_salvalai_smoke15` token equivalence.
3. Full-song SALVALAI token equivalence, byte-identical output sanity, and untraced throughput.

Keep only if it clears `>=5%` full-song improvement or if it is extremely simple and unlocks the broader `DecodeSession` runtime path.
