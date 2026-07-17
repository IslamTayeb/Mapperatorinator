# §47 W-KV keep-accepted-KV + O(1) rollback — handoff

**Status:** **IMPLEMENTING / GATING** — 2026-07-17  
**Branch / WT:** `codex/turbo-keep-accepted-kv` @ `/work/projects/Mapperatorinator-worktrees/turbo-keep-accepted-kv`  
**Base:** `codex/turbo-integrator` tip `e6898743`  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge**.

## Ruling (binding)

| Claim | Binding |
| --- | --- |
| Sampled turbo keep-accepted-KV fp16 ULP (verify-written vs fused-kernel rows) | **DOCUMENTED DRIFT** under §34; TIER1 KS covers distribution equivalence |
| Greedy TIER1a canary | stays **crop-to-L + aligned-Q1 rebuild** — certification unaffected |
| Process | Do **not** block sampled keep-KV on bit-parity again |

## What landed

| Piece | Detail |
| --- | --- |
| Mode split | `turbo_kv_commit_mode=keep_accepted_o1_rewind` when **not** aligned-greedy; `crop_to_L_full_rebuild` when canary |
| O(1) rollback | `rewind_self_cache`: StaticCache zeros only rejected band `[new_len, occupied_end)` (≤γ); DynamicCache crop; `cache_position` popped + mask rewind |
| No accepted reforward | Sampled path never replays kept tokens on teacher or draft; residual/bonus is at most one Q=1 |
| q1 bucket | Residual/bonus `forward_q1` uses rolled-back (+extra) `decoder_input_ids` length for active-prefix bucket |
| Persistent caches | `TurboDecodeSession` holds `teacher_cache`, `draft_cache`, `verify_fp` across windows; in-place reset only |

## Gates

| Gate | How measured |
| --- | --- |
| teacher forwards/cycle == **1** | `turbo_teacher_forwards_per_cycle` = verify forwards / steps (sampled K-path ⇒ 1); `turbo_teacher_accepted_reforwards_window == 0` |
| **−10 ms/cycle** vs §45 | phase timers vs §45 ref ~**45 ms/cycle** (`50148770` sustained) |
| TIER1a canary PASS | `jobs/s47-turbo-tier1a-canary.sbatch` (crop-rebuild path) |

## Kill

If teacher forwards/cycle cannot be 1 without collapsing canary mode separation → **STOP** with the measured number; do not grind.

## Jobs

| Job | Script | Purpose |
| --- | --- | --- |
| TIER1a canary | `jobs/s47-turbo-tier1a-canary.sbatch` | Prove canary path still PASS |
| FP16 scout | `jobs/s47-turbo-fp16-perf-scout.sbatch` | Path hits + forwards/cycle + ms/cycle delta |

## Not this lever

W-VG (§48) graph-native K=γ verify; W-DG (§49) graphed draft; tip graduation; 500 claim; merge to main.
