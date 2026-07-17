# §48 Graph-native K=γ verify (W-VG) — handoff

**Status:** **STOP_KILL** (2026-07-17) — graph-native path hit; c_verify still **1.669×** > 1.35 kill  
**Branch / WT:** `codex/turbo-graph-native-verify` @ **`be491a4f`**  
**Campaign tip:** still `55949274` / FP16 **366.11** — **no 500 claim**

## Measured (in-loop, not harness-only)

| Metric | Value | Evidence |
| --- | --- | --- |
| Path | **`graph_native_k`** | `50150290` |
| prepare_inputs hot-path | **0** | `50150290` |
| Q1 step | **1.842 ms** | same job |
| in-loop c_verify | **3.075 ms** | CUDA-event avg over replays |
| ratio | **1.669×** | vs Q1; gate ≤1.2 / kill >1.35 |
| Decision | **STOP_KILL** | stuck above 1.35 after graph-native attempt |

Auth §41 cudagraph K=5 was 3.125 ms / 1.686× — graph-native matches that band; glue removal did not clear ≤1.2×.

## What landed

| Piece | Change |
| --- | --- |
| Gate lift | `forward_k` allows CUDA graphs for K>1 by default |
| Graph-native path | Static `{ids[1,γ], cache_position[γ]}`; 4D mask filled in-graph; no HF prepare on hot path |
| Capture | Production side-stream + warmup=1 |
| Persistent cache | `TurboRuntime.verify_fastpath` shared across windows |
| In-loop meter | CUDA-event `turbo_verify_forward_ms_avg` |

## Jobs

| Job | Role | Result |
| --- | --- | --- |
| `50150048` | in-loop r1 | FAILED `proj_out` on wrapper |
| `50150203` | in-loop r2 | FAILED missing `cache_position` for HF update |
| `50150290` | in-loop auth | **STOP_KILL** 3.075 ms / **1.669×** |

## Not this lever / revisit

- Do not grind further graph-native variants without a new mechanism (shared arenas / fused K-verify kernel / different γ).
- Keep-accepted-KV (§47), graphed draft (§49), baseline glue (§46) remain independent.
- Tip graduation / merge / 500 claim: **no**.
