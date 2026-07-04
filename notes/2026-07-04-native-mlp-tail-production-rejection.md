# Native Decoder MLP Tail Production Rejection

## Summary

The verifier-only native decoder-layer MLP tail looked target-sized in CUDA graph replay, but the production opt-in path did not meet the promotion bar. It was exact on the corrected gates and 15s smoke, but the measured `profile_inference` gain was only `+2.8%` main-generation throughput and strict no-regression failed. The production flag wiring was reverted; keep only the verifier/profiler infrastructure.

## Evidence

- Branch: `codex/native-decoder-layer-verifier`
- Candidate commit: `7e89f17`
- Reverted production commits: `127f246`, `7bf0c68`, `7e89f17`
- Corrected one-token gate: DCC job `49258638`
  - Path: `/work/imt11/Mapperatorinator/runs/native-mlp-tail-one-token-20260704151426-7e89f17-normalprefill/one_token/one_token_gate.json`
  - Result: PASS
  - `max_abs=1.9073486328125e-05`, top-k match PASS
  - Uses normal prefill plus active-prefix decode length `128`, matching production more closely than the failed active-prefix-prefill harness.
- Direct-loop gate: DCC job `49258622`
  - Path: `/work/imt11/Mapperatorinator/runs/native-mlp-tail-gates-20260704150827-7e89f17-fullflags/direct_loop/direct_loop_gate.json`
  - Result: PASS
  - `256` sampled steps, generated-token match PASS, raw-logit/top-k PASS, final RNG PASS
  - CUDA graph diagnostics: `capture_count=5`, `decode_replays=255`, `total_capture_seconds=0.06556504499167204`
- 15s smoke profile: DCC job `49258644`
  - Path: `/work/imt11/Mapperatorinator/runs/native-mlp-tail-smoke15-20260704151908-7e89f17`
  - Control profile: `control/beatmape56b17c78d3f429d824a4c209949d7d5.osu.profile.json`
  - Candidate profile: `candidate/beatmap8999178229e94275977faab9c6692487.osu.profile.json`
  - Compare: `compare_main_generation.json` and `compare_main_generation.txt`

## Smoke Result

| metric | control | candidate | delta |
| --- | ---: | ---: | ---: |
| main-generation tokens | 1,084 | 1,084 | exact |
| main model time | 3.833s | 3.728s | -0.105s |
| main tok/s | 282.783 | 290.752 | +2.8% |
| output artifact | 4,144 bytes | 4,144 bytes | sha256 PASS |
| generated token IDs | 1,084 | 1,084 | PASS |
| outer wall | 67.468s | 70.036s | +3.8% worse |
| total stage wall | 74.777s | 76.930s | +2.9% worse |
| per-window no-regression | PASS | FAIL | seq0 and seq2 regressed |

The smoke gain extrapolates to roughly `0.74s` over `7,639` full-song SALVALAI main tokens, or about `270.5 -> 277.8 tok/s` if it scaled perfectly. That is below the `5%` keep threshold and far below the 500 tok/s target. The strict comparison failed because total stage wall and two per-window records regressed.

## Decision

Reject the production `inference_native_decoder_layer_mlp_tail` flag path and do not run a full-song promotion profile. The bottleneck proof remains useful, but the production result shows this MLP-tail-only native island is not the next bottleneck. The next serious path needs a broader decoder-layer/runtime island that reduces multiple segments together, not a narrow native MLP tail.

Keep:

- `osuT5/osuT5/inference/native_decoder_layer.py`
- `utils/profile_decode_decoder_layer_island.py --candidate-native-decoder-layer-island`
- `utils/validate_decoder_layer_abi.py` candidate cache-write validation

Reverted:

- user-facing `inference_native_decoder_layer_mlp_tail` config/Hydra flags
- `inference.py` and `server.py` production routing
- `Processor` profile metadata for the production flag
- model-layer production hook
- one-token/direct-loop verifier flags specific to the production hook

## Lesson

Graph-replay component ceilings can overstate production value when the candidate only replaces a narrow part of the layer and leaves broader control, graph, and neighboring segment costs intact. Before productionizing another native island, require the current bottleneck table plus a projected full-song saving comfortably above the keep threshold, then use 15s smoke to confirm that the saving appears in untraced `profile_inference`.
