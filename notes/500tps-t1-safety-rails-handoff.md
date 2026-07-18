# §59 T1 SAFETY RAILS — handoff

**Status:** **WIRED** (2026-07-18)  
**Branch / WT:** `codex/turbo-on-tiger-pr120` (local tip after this commit)  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**Base:** tiger PR #120 `d01cdd27` (local turbo scaffold on top)  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify**  
**Remote:** `islamtayeb` only — **no push to Tiger14n / origin / PR #120**  
**No merge.**

## What fixed

| Rail | Detail |
| --- | --- |
| Force SDPA | `inference.py` forces `attn_implementation='sdpa'` when `fast_decoder_loop` / `super_timing_fast_loop`; `require_sdpa_for_fast_path` gates capture |
| Loud capture | CaptureError → stderr + traceback + `warnings.warn`; latch refuses silent stock fallback unless `MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1` |
| Graph-cache VRAM | `_decoder_cache` / `_k_decoder_cache` are `OrderedDict` of `(decoder, cache, nbytes)`; shared budget default **2048 MiB** (`MAPPERATORINATOR_GRAPH_CACHE_BUDGET_MB`); LRU + `close()` + `empty_cache` |
| CFG left-pad KV | Pad-aware graph keeps full-length **2D** mask; replay sets new pos=1 + RoPE via `(mask.cumsum-1).clamp(0)`; CFG mask merge uncond∥cond; CFG token `repeat(2,1)` into (2B,1) graph buffer |

Preserved: `neutralize_dynamic_rope`, `_IncrementalMonotonicMask`, `CUDAGraphDecoder` q_len=K, buckets, turbo `get_k_decoder`.

## CFG bug (local doc)

Capture previously passed `decoder_attention_mask=None`, so decode attended pad K/V written at prefill. Stock HF extends the 2D pad mask each step. Unequal CFG left-pad → pad leak. Fix is local-only vs PR #120 until an upstream stacked PR is authorized.

FA2 `kv[:, :cache_position[-1]+1]` dynamic slice (modeling_varwhisper) is why the fast path forces SDPA.

## Smoke

| Item | Value |
| --- | --- |
| Scout | `utils/s59_t1_safety_rails_scout.py` |
| Job | `jobs/s59-t1-safety-rails.sbatch` |
| Asserts | require_sdpa / CFG mask+cumsum / eviction@1MiB / loud latch / force-SDPA |
| Env | `unset MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK` |
| Decision | fill after harvest |

## Do not

- Push to `origin` / `tiger14n` / PR #120
- Merge
- Touch tip `55949274`
- Claim 500 TPS
