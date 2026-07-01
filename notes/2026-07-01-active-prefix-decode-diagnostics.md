# Active-Prefix Decode Diagnostics

## Summary

Added a default-off active-prefix diagnostic flag:

```text
profile_active_prefix_decode_diagnostics=true
```

This is profiling infrastructure only. It does not change the retained cold single-song baseline and does not claim an inference speedup.

## What It Records

When active-prefix decode is enabled, generation profile records can now include `active_prefix_decode_diagnostics` with CPU-side wall counters for:

- cache-position setup and max-cache-shape lookup;
- generation compile-call lookup;
- `prepare_inputs_for_generation`;
- normal prefill forward;
- first and steady active-prefix decode forward;
- `_update_model_kwargs_for_generation`;
- logits extraction and logits processors;
- sampling;
- token append and stopping checks;
- `decode_steps`, `bucket_lengths_seen`, and `bucket_transition_count`.

The decode loop does not add tensor `.cpu()`, `.item()`, `.tolist()`, RNG inspection, or CUDA synchronization. NVTX/`record_function` spans use stable names such as `mapperatorinator.active_prefix.decode_forward`; bucket lengths are recorded in JSON counters instead of dynamic per-token range names.

## Use

Use it for torch-profiler or Nsight diagnosis only:

```bash
profile_active_prefix_decode_diagnostics=true \
profile_generation_detail_ranges=true
```

For JSON counters without broader VarWhisper ranges, use only `profile_active_prefix_decode_diagnostics=true`.

## Validation Plan

- Compile and local dependency-light tests.
- DCC import/compile checks.
- Direct-loop token/logit/RNG gate with active-prefix diagnostics enabled.
- 15s smoke token-equivalence check before treating any diagnostic output as trustworthy.
