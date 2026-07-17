# §48 Graph-native K=γ verify (W-VG) — handoff

**Status:** **OPEN / measuring** (2026-07-17)  
**Branch / WT:** `codex/turbo-graph-native-verify`  
**Campaign tip:** still `55949274` / FP16 **366.11** — **no 500 claim**

## What landed

| Piece | Change |
| --- | --- |
| Gate lift | `forward_k` no longer hard-disables CUDA graphs for K>1 |
| Graph-native path | Static replay inputs `{decoder_input_ids[1,γ], cache_position[γ]}`; 4D mask filled in-graph; **no** HF `prepare_inputs_for_generation` on hot path |
| Capture | Production side-stream + `GRAPH_CAPTURE_WARMUP=1` (§41 zero-logits class) |
| Persistent cache | `TurboRuntime.verify_fastpath` shared across windows/sessions |
| In-loop meter | CUDA-event `turbo_verify_forward_ms_avg` (excludes capture + host rejection glue) |
| Probe | `utils/s48_graph_native_verify_inloop.py` + `jobs/s48-graph-native-verify-inloop.sbatch` |

## Gates

| Gate | Threshold |
| --- | --- |
| PASS | in-loop c_verify / Q1 ≤ **1.2** (≤**2.22 ms** @ Q1=1.853) |
| KILL | ratio > **1.35** after graph-native attempt → STOP |
| Path hit | `turbo_verify_path == graph_native_k` under `do_sample` |
| Hot path | `prepare_inputs_calls == 0` on graph-native verifies |

## Jobs

| Job | Role | Result |
| --- | --- | --- |
| _(pending)_ | in-loop c_verify | — |

## Not this lever

- Keep-accepted-KV (§47), graphed draft (§49), baseline glue (§46)
- Tip graduation / merge / 500 claim
