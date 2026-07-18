# §59 T1 SAFETY RAILS — handoff

**Status:** **SMOKE PASS** (2026-07-18)  
**Branch / WT:** `codex/turbo-on-tiger-pr120` @ **`72d6ed72`** (rails `26e465b3`)  
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

## Smoke import fix (resubmit)

Prior job `50194265` FAILED before rails: scout put `REPO/osuT5` ahead of `REPO` on `sys.path`, so `import osuT5` bound the inner package and `osuT5.osuT5` raised `ModuleNotFoundError`.

Fix in `72d6ed72`:
- `utils/s59_t1_safety_rails_scout.py` — insert **repo root only** (same pattern as s58 c-verify)
- `jobs/s59-t1-safety-rails.sbatch` — `export PYTHONPATH=$REPO` + `import_ok` preflight check

## Smoke

| Item | Value |
| --- | --- |
| Scout | `utils/s59_t1_safety_rails_scout.py` |
| Job | `jobs/s59-t1-safety-rails.sbatch` → **`50215788`** (**COMPLETED**, ExitCode `0:0`, Elapsed `00:00:39`, Node `dcc-core-ferc-s-z25-20`, GPU **2080 Ti**) |
| Asserts | require_sdpa / CFG mask+cumsum / eviction@1MiB / loud latch / force-SDPA — **all PASS** |
| Env | `unset MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK` |
| DCC sync | job commit `72d6ed72` (rails `26e465b3`) |
| Decision | **PASS** |

Prior FAIL: `50194265` (import only; no rail asserts).

## Harvest seal

- **Sealed:** 2026-07-18
- **Job:** `50215788` State=`COMPLETED` ExitCode=`0:0` Start=`2026-07-18T14:51:41` End=`2026-07-18T14:52:20` Node=`dcc-core-ferc-s-z25-20`
- **Smoke verdict:** **PASS** (`OK=true`, `DECISION=PASS`, `RC=0`)
- **Rails:** `require_sdpa=PASS`, `cfg_left_pad_mask=PASS`, `graph_cache_eviction=PASS` (1 eviction under 1MiB budget), `loud_capture_failure=PASS`, `force_sdpa_setup=PASS`
- **Commit (job):** `72d6ed72987f3ea8f1ca28edacadfd3852bb935c`
- **Campaign tip frozen:** `55949274` — untouched
- **Preflight:** `import_ok CaptureError`; unique TMPDIR/TORCH_EXTENSIONS under `/work/imt11/Mapperatorinator/{tmp,torch_extensions}/s59-t1-rails-50215788`
- **Remote run dir:** `/work/imt11/Mapperatorinator/runs/s59-t1-rails-50215788/` (`asserts.json`, `stdout.txt`, `stderr.txt`, `summary.json`, `preflight.txt`)
- **Remote logs:** `/work/imt11/Mapperatorinator/logs/s59-t1-rails-50215788.{out,err}`
- **Local artifacts:** `notes/s59-artifacts/` (`verdict-50215788.txt`, `summary.json`, `asserts.json`, `stdout.txt`, `stderr.txt`, `preflight.txt`, `sacct-50215788.txt`, `s59-t1-rails-50215788.out`, `s59-t1-rails-50215788.err`, plus `*-50215788.*` tagged copies)
- **No 500 claim.** No PR #120 push.

## Do not

- Push to `origin` / `tiger14n` / PR #120
- Merge
- Touch tip `55949274`
- Claim 500 TPS
