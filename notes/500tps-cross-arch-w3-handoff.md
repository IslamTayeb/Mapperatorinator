# W3 — Tiger Batched Encoder Prefill (Handoff)

**Status:** **SCAFFOLD READY** — CPU unit/smoke green; **GPU reciprocal DEFERRED** until W1 harvest seals  
**Branch:** `codex/w3-batched-encoder-prefill`  
**Base tip:** `55949274` / FP16 **366.11** — **no merge to main**; **no 500 claim**  
**Package:** `notes/500tps-cross-arch-package.md` · W1: `notes/500tps-cross-arch-w1-handoff.md`

## Why W3 (provisional)

A5000 W1 lean: Tiger compiled-decode fp16 recon **~350 ≥** optimized **~316**. Package prefers **W3** (Tiger 16-wide window encoder prefill) over **W2** (bf16) **if Ada confirms** tiger ≥ optimized. Firm call still waits on W1a/W1b harvest.

Prior related STOP: encoder precompute `49903861` **STOP_ENCODER_PRECOMPUTE** (B1 complete-request wall regress). W3 revisits with **B16**, CPU store, and decode-skip credited to complete wall — not a B1-only encoder microbench.

## Tiger source (PR #120)

Upstream / local WT `w1-tiger14n-compiled-decode`:

- `server.precompute_encoder_outputs` — chunked encoder over all windows (`batch_size=16` default via `max_batch_size`)
- Processor path hoists precompute before sequential window decode; each window passes `encoder_outputs=enc_hidden[wi:wi+1]`
- Store on **CPU**; quality-equivalent / **not bit-identical** to one-window-at-a-time (batched kernel drift)
- Opt-in; V32 path unchanged when disabled

## Our port (ownership)

| Piece | Path | Role |
| --- | --- | --- |
| Core scout | `osuT5/.../optimized/scout/batched_encoder_prefill.py` | B16 precompute + opt-in Processor hooks |
| Unit/smoke | `tests/test_batched_encoder_prefill_scaffold.py` | CPU-only; mock encoder B1≡B16 |
| Ceiling harness (existing) | `utils/profile_batched_encoder_precompute_ceiling.py` | Verifier-only B1..B16 timings |

**Not wired into** accepted `OptimizedSingleRuntime` / selectors / server. Install only via:

```python
from osuT5.osuT5.inference.optimized.scout.batched_encoder_prefill import (
    install_batched_encoder_prefill_candidate,
)

with install_batched_encoder_prefill_candidate(Processor, batch_size=16, storage="cpu"):
    ...
```

Defaults: `batch_size=16`, `storage="cpu"`, precisions `{fp16, fp32}`, `result_class=documented-drift`.

## Implemented vs TODO

### Implemented (this scaffold)

- Tiger-aligned `chunk_ranges` / B16 precompute loop via `model.get_encoder()`
- CPU store default + per-window H2D at inject (VRAM-safe)
- Opt-in `install_batched_encoder_prefill_candidate` (no cold-default import)
- Timing→main store reuse; conditioning / source-window identity checks
- Profiler metadata key `optimized_batched_encoder_prefill`
- CPU unit + mock-encoder smoke (no full song, no GPU)

### TODO (after W1 seals — do not fire GPU yet)

1. **Firm W3 go/no-go** from sealed W1 matrix (2080 Ti + Ada + A5000).
2. **DCC reciprocal** vs tip `55949274` / 366.11 on same song / seed policy:
   - baseline: optimized tip sequential encoder-in-loop
   - candidate: tip + `install_batched_encoder_prefill_candidate(..., batch_size=16)`
3. Exactness / drift report (greedy tokens + `.osu` bytes policy for documented-drift).
4. Complete-request wall gate (≥5% realistic headroom; JIT excluded from TPS claim).
5. Optional production wiring under `optimized/single/` **only after** gates pass — still no V32 / server change.
6. Decide whether turbo stack wants encoder prefill under the same request wall (see below).

## Gates (promotion)

Compare **like with like** on tip hardware (prefer 2080 Ti for tip-class TPS; A5000/Ada for cross-arch).

| Gate | Pass rule |
| --- | --- |
| **Ownership** | Changes only under `inference/optimized/`; V32 cold path / tip bytes unchanged when candidate off |
| **Exactness** | Documented-drift preset: greedy token IDs / `.osu` within declared policy vs tip optimized; do **not** require bit-identical encoder hidden states |
| **Wall** | Candidate complete-request wall (first-main-to-last-main + request-to-output) improves ≥**5%** vs tip optimized; precompute time charged to request; no projection-as-production |
| **TPS** | Main-gen model TPS must not regress ≥1% vs tip **366.11** class on same GPU/precision (encoder skip should be ~neutral-to-positive on model TPS; wall is the primary W3 claim) |
| **Memory** | Peak VRAM with CPU store ≤ tip + small H2D scratch; fail loudly on OOM |
| **Stop** | First exactness / negative-wall / ownership / insufficient-gain failure → remove wiring, keep scout + verifiers |

Do **not** treat PR #120 marketing tok/s or isolated encoder ceiling rows as production throughput.

## Stacking with turbo endgame

Turbo (`notes/500tps-turbo-endgame-package.md`, remapped §§53–57) is **decode-bound** (E binding on map_rest / lookback). W3 is **encoder / complete-wall**.

| Interaction | Guidance |
| --- | --- |
| Independent | W3 does not fix §56 E; turbo does not replace W3 encoder hoist |
| Compose later | Only after each track seals its own gate; measure stacked complete wall + main TPS + E separately |
| Claim hygiene | Never fold W3 wall savings into a turbo TPS projection, or turbo E into a W3 wall claim |
| Capacity | One GPU experiment at a time per worker; ≤2 concurrent if idle; W1 harvest + §57 take priority over W3 reciprocal |

## Ready-for-GPU?

**N** — scaffold + CPU tests only. Fire GPU reciprocal **after** W1 harvest seals (next productive wake ≥ **2026-07-18 10:28 EDT** for 2080 Ti; Ada later) and coordinator confirms W3 over W2.

## Smoke (CPU)

```bash
cd /work/projects/Mapperatorinator-worktrees/w3-batched-encoder-prefill
python -m pytest tests/test_batched_encoder_prefill_scaffold.py -q
```

## One-line summary

W3 scaffold ports Tiger PR #120 16-wide encoder prefill under `optimized/scout/` as opt-in documented-drift (CPU store, fp16/fp32); tip `55949274` untouched; GPU reciprocal deferred until W1 seals Ada/2080 lean.
