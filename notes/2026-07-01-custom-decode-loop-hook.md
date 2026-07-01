# Custom Decode Loop Hook Spike

## Idea

Add a conservative opt-in `inference_custom_decode_loop` path that temporarily replaces Hugging Face's internal `_sample` loop with a local copy-compatible loop. The goal was not to claim a speed win immediately; it was to prove whether a custom decode insertion point can preserve exact generated-token identity before attempting CUDA graphs, preallocated buffers, or lower-level runtime work.

The prototype was hard-guarded to the narrow v1 surface:

- `use_server=false`
- `parallel=false`
- `cfg_scale=1.0`
- `num_beams=1`
- batch size `1`
- tensor output only, no streamers or auxiliary generate outputs

## Results

Commit tested: `07b36a5`

Both gates used the middle-15s SALVALAI smoke slice with `seed=12345`, `attn_implementation=sdpa`, `use_server=false`, and recorded generated token IDs.

### Compile-disabled gate

- Baseline job: `49135145`
- Candidate job: `49135146`
- Node: `dcc-core-ferc-s-z25-21`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke15-loop-base2-compileoff-07b36a5/beatmapba4408b7bf2e444e9a7c761640053d40.osu.profile.json`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke15-loop-custom2-compileoff-07b36a5/beatmapcba4ff105a834900bb11cb79e288b873.osu.profile.json`
- Token equivalence: PASS, `1,084 / 1,084` generated main-token IDs matched
- Throughput: `61.716 -> 54.993 tok/s`, `-10.9%`
- Model time: `17.564s -> 19.711s`, `+12.2%`

### Compile-enabled gate

- Baseline job: `49135380`
- Candidate job: `49135381`
- Node: `dcc-core-ferc-s-z25-21`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke15-loop-base-compileon-07b36a5/beatmap2dcd0585607145de8553c8d57889c856.osu.profile.json`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke15-loop-custom-compileon-07b36a5/beatmapc7f1c08725d445aaaf73d0e57e40eaa6.osu.profile.json`
- Token equivalence: PASS, `1,084 / 1,084` generated main-token IDs matched
- Throughput: `50.677 -> 43.223 tok/s`, `-14.7%`
- Model time: `21.390s -> 25.079s`, `+17.2%`

Post-warmup compile-enabled windows were slower too. For example, seq9 moved from `104.8 tok/s` baseline to `93.5 tok/s` candidate, so the regression was not just first-window compile noise.

## Decision

Rejected and reverted.

The useful result is equivalence: replacing the HF `_sample` loop can preserve fixed-seed generated-token identity if `model.generate()` still owns normal setup, stopping criteria, logits processors, static-cache wiring, and generation compile checks.

The rejected part is the copy-compatible hook itself. It adds overhead and does not remove a true cost center. Keeping a default-off slower loop would create complexity without moving toward 200 tok/s.

Future custom runtime work should not repeat a copy-only `_sample` monkeypatch. It needs to attack a real target:

1. CUDA graph capture around stable one-token forward calls.
2. Preallocated buffers that preserve exact HF token/RNG behavior.
3. A compiled/exported one-token decoder forward path with logits and end-to-end token equivalence.
