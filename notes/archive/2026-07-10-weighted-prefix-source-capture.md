# Weighted Bucket Real-Prefix Capture Gate

## Scope

This is a capture-only prerequisite for the production-weighted B2 scout. It
does not execute or authorize bucket-576 H8 timing, a scheduler, runtime wiring,
or server work. The default V32 path remains unchanged.

The source is the accepted exact Lambada repeat01 profile from job `49543717`,
commit `a709b86`, plus its accepted audio. The capture utility pins their file
SHA-256, byte sizes, audio sample count, timing/main transcript hashes, and the
first 42 seq9 token IDs.

## Reconstruction Gate

`utils/verify_optimized_weighted_prefix_source.py` reproduces the production
state rather than padding a synthetic prompt:

1. replay accepted timing token IDs with the base V32 tokenizer
   (`auto_select_gamemode_model=false`, vocab `4097`) and run the same
   timing-context event finalization and trimming used by `Processor.generate()`;
2. inject accepted main seq0 through seq8 IDs into the prior output context and require every
   recorded prompt-token count using the gamemode-0 tokenizer (vocab `4069`);
3. reconstruct Lambada main seq9 prompt length `478` and retain its real frames;
4. advance a private seed-12345 CUDA generator in two typed multinomial
   segments: `132 x [1,4097]` timing draws, retain the intermediate state, then
   `1,042 x [1,4069]` main draws;
5. sample accepted seq9 tokens t0 through t41 through a fresh retained normal
   logits processor, failing on the first token, EOS, or RNG-chain mismatch;
6. execute exactly 41 model decodes for t0 through t40, prove every self-cache
   tensor is populated at position `518` and still zero at `519`, then prepare
   accepted t41 (`924`) with cache and decoder position `519`.

The resulting capture prefix has length `520`; a future eight-step follow-up remains
inside bucket `576` and the accepted seq9 transcript. The contract records
typed shape/stride/dtype/hash descriptors for prompt, mask, frames, generator
states, raw logits, prepared static tensors, and every self/cross cache tensor.
It states `prior_context_tokens_source=accepted_profile` and separately proves
`target_prefix_forced_tokens=false`; the 42 target tokens must be sampled.

The accepted profile does not record a baseline RNG-state hash. Therefore the
known one-multinomial-per-token draw count plus 42 exact sampled tokens strongly
supports the reconstructed private RNG state, but does not independently
compare that state to a production-recorded final RNG hash.

## Promotion Boundary

The report embeds `phase_a_authorized=false`. Its canonical contract digest is
only an observed artifact identifier. Before any H8 code or job may run, a
separate reviewed commit must pin that digest and the tensor descriptors. A
capture-created digest cannot approve itself.

The first GPU capture attempt, job `49555503` at commit `5930c41`, failed after
`29s` on an RTX 2080 Ti before producing any capture report. It replayed timing
IDs with the gamemode-0 tokenizer, so the accepted base-tokenizer BEAT/MEASURE
IDs were decoded as different events and `Postprocessor.generate_timing()`
received no timing points. This is a setup-only failure, not exactness or
performance evidence. Logs are
`/work/imt11/Mapperatorinator/logs/weighted-prefix-source-49555503.{out,err}`.

The next gate is the model-free CPU wrapper
`scripts/dcc/verify_weighted_prefix_context.sbatch`. It loads both tokenizers,
reconstructs timing and every main prompt, pins the exact event/marker counts,
and records CPU prompt/mask/frame descriptors without loading a model or using
CUDA.

## Reviewed CPU Context Result

Job `49558528` at commit `c6b45fd` completed on the CPU-only `common`
partition in `1m59s` with exit `0:0`. Independent validation reproduced the
strict report and its canonical JSON digest:

- canonical report SHA-256:
  `bf2e787bff309e8ce62382e2e3a00af3e4230b00b210331399c471cc4236e72b`;
- pretty-file SHA-256, retained for audit only:
  `d47bc29cfb2755133bc48fa722f4475d06a94bfd467bc2b077e9bdfa9e7573de`;
- target prompt: `int64 [1,478]`, stride `[478,1]`, SHA
  `216fde48e17e2b4b9cebebadb454efe81b80a86bd2c1c81e4c710db35bb27663`;
- target mask: `bool [1,478]`, stride `[478,1]`, SHA
  `d9ad68f2a11712547d5eb0fa4bb5c0e80e319cfab464516aa02b0e141ee8b5f4`;
- target frames: `float32 [1,262016]`, stride `[262016,1]`, SHA
  `a2c94c30626130130d5b90c6a458a7bf5bed5f752cfdab91073837c9f874d642`.

All ten prompt counts matched `[48,400,441,446,450,471,472,475,480,478]`.
The timing ledger matched `122` pretrim events, `60` retained events, `30`
time shifts, `22` beats, `8` measures, `30` markers from `71,237` through
`85,875 ms`, and exactly one resulting timing point. The report records
`model_loaded=false`, `cuda_used=false`, and `gpu_capture_authorized=false`.
Its artifact is
`/work/imt11/Mapperatorinator/runs/weighted-prefix-context-49558528-c6b45fd/weighted-prefix-context.json`;
logs are `/work/imt11/Mapperatorinator/logs/weighted-prefix-context-49558528.{out,err}`.

The post-report commit pins the canonical digest and every observed context
descriptor. A full capture must freshly reconstruct the CPU report from the
pinned profile/audio and match its canonical digest before CUDA configuration,
CUDA seeding, native-extension work, or model loading. After model loading, its
live GPU prompt/mask/frame, tokenizer, timing, and source evidence must again
equal the reviewed CPU fields. There is no CLI report or digest override.

This authorizes exactly one capture-only GPU job. It still does not authorize
H8 timing, scheduler/runtime/server code, or a performance claim.

## First Reviewed-Context Capture Attempt

Job `49559157` at commit `f2cea66` reached an RTX 2080 Ti and failed after
`44s` at the final prepared-mask shape assertion. It produced telemetry but no
capture JSON. Control flow proves that the fresh CPU canonical digest, live GPU
context descriptors, FP32 model/tokenizer contracts, segmented prior RNG
advancement, all 42 accepted sampled tokens with no EOS, 41 cached decode
calls, prefix/bucket bounds, cache position `518` populated and `519` empty,
and prepared token/position `924/519` passed before the failure. Those values
were not persisted into a validated source-contract artifact, so this remains
setup-only partial evidence rather than an accepted exact capture.

The failed `[1,1,1,1024]` assertion confused encoder source capacity with
decoder target capacity. The corrected retry now pins gamemode-0 checkpoint
snapshot
`74f22583400d259bf424819e11027c17933efe54` has config SHA
`ff96b8c059179978c93b6f938e39fac945d5682a3495789c00cc159d571a1a22`,
`max_source_positions=1024`, and `max_target_positions=2560`.
`get_cache()` therefore creates a decoder self `StaticCache` of length `2560`.
For cached `q_len=1` at position `519`, `prepare_inputs_for_generation()` must
produce `[1,1,1,2560]`: positions `0..519` are zero/allowed and `520..2559`
are FP32 minimum/masked. The active-prefix attention implementation later
slices that mask to bucket `576`, leaving 520 allowed and 56 masked positions.

The corrected verifier pins the snapshot/config, requires model target capacity
and every self-cache sequence dimension to equal `2560`, derives the full mask
width from that cache, checks the exact full and bucket-sliced values, and
records both descriptors. It does not crop or change the prepared input. This
derivation authorizes one corrected capture-only retry; all earlier gates must
repeat and the job must stop again with `phase_a_authorized=false` and
`h8_executed=false` for separate artifact review.

## Reviewed Capture Result

Corrected capture-only job `49559386` at commit `1896b19` completed in `58s`
on an RTX 2080 Ti. The report is
`/work/imt11/Mapperatorinator/runs/weighted-prefix-source-49559386-1896b19/weighted-prefix-source.json`.
Independent validation reproduced:

- canonical source-contract SHA-256
  `de55e2d16c4b085c5cce62e83be8cd4f83bb23f2a40e87f80be23a88f216b99a`;
- pretty report-file SHA-256
  `8757f97874fd7249420feb1d2e03de43a646da8187cdebb34d3ee306baf7de62`;
- 42 unforced sampled tokens, no EOS, accepted/sampled transcript SHA
  `66960797a93b3f094f36cffadfcc44cbf27ad837eb30cad3d9cdde95a1381f06`;
- continuous RNG chain from pre-target
  `3903731b09c872d94364baa79c33a2794a0d7b129abf900cb5dace10cba1133d`
  to post-replay
  `61758aa77b7fe8f0603235093d2b38c9ecc3658344b31c6e2757e15a305f030b`;
- raw pre-last-sample logits SHA
  `8e0fee1b926d5f9f7f25f02042983e47c7d391e9c439beffe26f134782ed01ae`;
- prepared full mask SHA
  `7411b76017def4ea0c62a52705ba9e3d5c8a665f4769b85358a52cc31c0b2418`
  and prospective bucket-576 slice SHA
  `6c2d8cf911b5d720b855b872ac580f64a9af211e7a56c1e0441ce603da9889b5`;
- 24 self and 24 cross cache tensors, aggregate SHAs
  `77ec075467a970497a99cd6cbb0b266269fe51db53674480272b8058aa74b466`
  and `0b4307698b3556f9881590cb6ce9ff9a9a24330dc33883b1b994006702920ed4`.

The post-capture pin commit stores the job/commit, canonical/file digests, and
critical prompt, RNG, logits, mask, and cache hashes. Its loader accepts no
digest override: it checks the raw report file, strict source contract,
canonical digest, runtime commit/GPU/context provenance, two RNG-advance
segments, unchanged extension cache, and every capture-only boundary.

This result class is an exact sampled-prefix/source-state contract. It is not a
full `exact-output` result because only 42 of seq9's 327 tokens were replayed
and it does not cover final production stop/RNG or `.osu` bytes. It is not a
bitwise optimization comparison and contains no throughput evidence. It
authorizes only a separately reviewed bucket-576 H8 Phase-A verifier
implementation. Phase B, production runtime, scheduler, offline engine, and
server work remain unauthorized.
