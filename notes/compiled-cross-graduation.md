# Compiled-cross graduation (relaxed Track A)

**Status:** measured independent wall win — graduate for serial composition / confirmation, **not** a solo 500-TPS finish.

## Winner

| Field | Value |
| --- | --- |
| Job | `49910316` (h36-9, COMPLETED 0:0) |
| Scout | `49906034` — `max-autotune-no-cudagraphs` PASS; nested `reduce-overhead` fails under outer capture |
| Branch tip | `codex/500tps-arena-compiled-cross-last-mile` @ `0dbab9e5` |
| Confirmation pin | `codex/500tps-compiled-cross-confirmation` @ `9cc30063` (= `0dbab9e5` + this pin note) |
| Artifacts | `/work/imt11/Mapperatorinator/runs/selected-arena-compiled-cross-reciprocal-49910316/` |
| Gate | same-commit reciprocal; baseline `run_k4_shared_rope_fp16_cross_shared_arena.py`; candidate `..._compiled_bmm.py`; `REQUIRE_COMPILED_CROSS_INCREMENTAL=true` |

### Measured deltas (job-local reciprocal medians)

- main TPS **477.96 → 483.61** (+5.65)
- main_model **−0.208s**; fixed-8294 main **−0.203s**; complete-request wall **−0.319s**
- `final_map_equal`; main/timing token parity; declared capture-hit dispatch pass (relaxed / documented drift)

## Independence assumptions (do not treat as proven composition)

1. **Orthogonal to failed DP4A self-QKV** (`49908852` DROP). Do **not** merge `codex/500tps-arena-dp4a-last-mile` (`22faee3b`): `weight_only_runtime.py` / engine / wrappers conflict; DP4A regressed tokens + TPS.
2. **Same-commit selected control** already is the composition surface: selected shared-arena tip lineage (`3cd327df` ⊂ `e9a1d9a2` ⊂ `0dbab9e5`) plus opt-in compiled cross BMM inside outer CUDA graphs. No further cherry-pick needed for selected+compiled-cross.
3. **Setup/compile ownership stays offline/AOT.** Scout setup ~13.3s must remain outside song wall; cold one-off compile is not a promotion claim.
4. **Absolute TPS is node/wrapper contingent.** Job `49910316` baseline median ~**478 TPS** is **not** the campaign selected reference **494.27 TPS** (DP4A control on h36-6 saw ~492). Same-job reciprocal delta is evidence of a win; **additive projection onto 494.27 is forbidden** until a pinned reciprocal on the true selected tip reproduces both the high baseline and the candidate delta.
5. **Exactness:** relaxed Track A only; tiny numeric drift expected. Exact production track stays separate.

## Combo plan (selected tip — clean)

| Item | Value |
| --- | --- |
| Base tip | selected shared-arena lineage @ `e9a1d9a2` / ancestor `3cd327df` |
| Cherry-pick / tip | `0dbab9e5` (already applied); pin `9cc30063` |
| DP4A merge | **blocked** — multi-file conflicts; candidate failed |
| Reciprocal env | `BASELINE_RUNNER=utils/run_k4_shared_rope_fp16_cross_shared_arena.py`; `CANDIDATE_RUNNER=utils/run_k4_shared_rope_fp16_cross_compiled_bmm.py`; `REQUIRE_COMPILED_CROSS_INCREMENTAL=true`; `CROSS_CANDIDATE_MODE=fp16_packed_projections`; unique `RUN_LABEL` + `tmp/reciprocal-$JOBID` + job-local `torch_extensions` |
| 500 TPS expectation after composition | **Plausible only if** the −0.208s main delta transfers onto a real ~494 TPS baseline (~0.198s gap). **Not expected** from `49910316` alone (483.61). Remeasure required. |

## Solo confirmation submit (post-graduation)

| Field | Value |
| --- | --- |
| Job | `49952708` (pinned `h36-9`, isolated `RUN_LABEL=selected-arena-compiled-cross-confirm-solo`) |
| Tip | same runtime as `0dbab9e5` / pin `9cc30063` |
| Parallel | DP4A retry `49952461` on `h36-6` — ignored for this path |
| Expectation | reconfirm same-job delta; **do not** treat as 494→500 proof unless baseline lands near campaign reference |

## Non-goals

- No merge to `main`
- No overlapping authoritative multi-candidate confirmation on one GPU
- Do not wait on DP4A retry jobs for this graduation path
