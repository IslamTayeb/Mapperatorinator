# 200 tok/s Next Steps

## Current Baseline

The retained same-calculation baseline is SDPA plus `inference_generation_compile=true`:

- Full-song SALVALAI job: `49113713`
- Profile: `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json`
- Main generation: `7,639` tokens, `82.615s` synchronized model time, `92.465 tok/s`
- Fixed-seed generated-token equivalence against compile-disabled baseline: PASS

Reaching `200 tok/s` requires the same `7,639` tokens in about `38.195s`, which means removing about `44.420s` or `53.8%` of retained model time. This is not a wrapper-cleanup target; the retained profile has only `0.178s` outside synchronized `model.generate` across `87` main-generation windows.

## Policy

Treat `200 tok/s` as a serious runtime/kernel research target, not permission to change the calculation. Accepted speedups must keep precision, sampling policy, output policy, model quality, windowing/overlap, generated-token behavior, output length, and RNG behavior equivalent unless the run is explicitly labeled non-equivalent.

Use `configs/inference/profile_salvalai_smoke15.yaml` for the first pass of quick scouting. It covers the middle 15s of SALVALAI with `seed=12345`, `attn_implementation=sdpa`, `use_server=false`, and `profile_record_token_ids=true`.

For custom runtime work, prove equivalence in this order before claiming speed:

1. Compile-disabled 15s smoke token equivalence.
2. Compile-enabled 15s smoke token equivalence.
3. Full-song SALVALAI token equivalence.

## Ranked Projects

1. Exact custom decode loop with CUDA graph discipline.
   - Opt-in only, and hard-disable CFG, beams, parallel mode, and server batching in v1.
   - Preserve HF `generate` semantics: top-p/top-k, temperature processors, monotonic time shift, lookback EOS behavior, static cache, RNG consumption, EOS stopping, and generated-token accounting.
   - Use preallocated buffers and stable-shape one-token forward calls.
   - Graduate only after 15s smoke equivalence plus a plausible `>=10%` smoke win, then full-song equivalence plus `>=10%` full-song win.

2. Fused sampling/logits-processor spike.
   - First add profiler subranges around logits processors and sampling.
   - Stop if sampling is not at least `10%` of synchronized main-generation time.
   - If it is large enough, prototype an exact fused path. If RNG/token identity cannot be preserved, document it only as non-equivalent.

3. Torch-TensorRT / TensorRT-RTX feasibility spike.
   - Export or compile only the repeated one-token decoder forward first.
   - Check RTX 2080 Ti / Turing compatibility and static KV-cache feasibility.
   - Graduate only if logits match within the chosen dtype tolerance and end-to-end generated token IDs match.

4. Backend/version refresh after runtime probes.
   - Do not blindly retry FA2, dynamic cache, or prefix trimming.
   - Revisit attention backends only if new traces show attention is the limiting cost under the custom/runtime path.

## Goal Prompt

```text
Optimize Mapperatorinator inference toward 200 tok/s main-generation throughput on RTX 2080/2080 Ti, same-calculation only. Current retained baseline is SDPA + `inference_generation_compile=true`: 7,639 full-song SALVALAI main tokens, 82.615s synchronized model time, 92.465 tok/s, fixed-seed token equivalence PASS against compile-disabled baseline.

Treat 200 tok/s as a serious research target, not permission to change the calculation. Do not claim speedups from changed precision, sampling policy, output policy, model quality, windowing/overlap, generated-token behavior, output length, or non-equivalent RNG behavior unless explicitly labeled non-equivalent.

Use a profiling-first loop. Start with a middle-15-second SALVALAI smoke slice, prove fixed-seed generated main-token IDs match the retained baseline, and promote only promising exact-calculation changes to longer smoke or full-song SALVALAI runs. Use full-song runs for accepted results. Keep SDPA + generation compile as the baseline unless profiler evidence and full-song token-equivalent runs strongly justify replacing it. Separate true model time from torch.profiler overhead.

Prioritize deeper runtime/kernel work that could plausibly remove about 54% of retained full-song model time: exact custom decode loop, CUDA graph discipline, fused sampling/logits processors, static-cache/layout work, and Torch-TensorRT/TensorRT-RTX feasibility. Do not reintroduce rejected quick tweaks unless new profiler evidence explains why the old negative or non-equivalent result no longer applies.

For custom runtime work, require compile-disabled 15s smoke token equivalence first, then compile-enabled 15s smoke token equivalence, then full-song token equivalence before any speed claim graduates. Keep changes that improve RTX 2080 full-song main-generation throughput by >=10%; keep 5-10% only if simple and strategic toward the custom runtime; remove 1-3% complexity by default.

Commit and push clean checkpoints for accepted wins, document every accepted/rejected experiment in docs/inference_profiling.md and notes/, update AGENTS.md with durable conventions, and stop only when 200 tok/s is reached or profiling shows no remaining plausible exact-calculation path toward a major gain.
```
