#!/usr/bin/env python3
"""§58 STEP 2: wire + measure turbo speculative window on tiger.

Runs full-map inference with MAPPERATORINATOR_TURBO=1 and reports:
  - in-loop E (accepted_per_verify)
  - ms_per_map_token / cold_start_seconds / main_tps (metric ruling)

Strict rejection-sampling only. Kill on E collapse / negative wall.
No 500 claim. Tip 55949274 frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()

OVERLAY = r'''
import atexit
import json
import os
import time
from pathlib import Path

import torch

_STATS_PATH = Path(os.environ["MAPPERATORINATOR_S58_STATS_PATH"])
_COLD_START = float(os.environ["MAPPERATORINATOR_S58_COLD_START"])
_STATE = {
    "first_generate_at": None,
    "main_tokens": 0,
    "main_model_seconds": 0.0,
    "timing_tokens": 0,
    "timing_model_seconds": 0.0,
    "calls": [],
    "turbo_windows": [],
}


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _install():
    from osuT5.osuT5.inference import processor as proc_mod
    from osuT5.osuT5.inference import server as server_mod
    try:
        from osuT5.osuT5.inference import compiled_decode as cd_mod
    except Exception:
        cd_mod = None

    orig_mg = server_mod.model_generate

    def mg_wrap(model, tokenizer, model_kwargs, generate_kwargs):
        _cuda_sync()
        t0 = time.perf_counter()
        result, stats = orig_mg(model, tokenizer, model_kwargs, generate_kwargs)
        _cuda_sync()
        elapsed = time.perf_counter() - t0
        if isinstance(stats, dict):
            stats = dict(stats)
            stats["elapsed_seconds"] = elapsed
            toks = int(stats.get("generated_tokens", 0) or 0)
            stats["tokens_per_second"] = toks / elapsed if elapsed > 0 else 0.0
        return result, stats

    server_mod.model_generate = mg_wrap
    proc_mod.model_generate = mg_wrap

    if cd_mod is not None and hasattr(cd_mod, "model_generate_compiled"):
        orig_cd = cd_mod.model_generate_compiled

        def cd_wrap(model, tokenizer, model_kwargs, generate_kwargs):
            _cuda_sync()
            t0 = time.perf_counter()
            result, stats = orig_cd(model, tokenizer, model_kwargs, generate_kwargs)
            _cuda_sync()
            elapsed = time.perf_counter() - t0
            if isinstance(stats, dict):
                stats = dict(stats)
                stats["elapsed_seconds"] = elapsed
                toks = int(stats.get("generated_tokens", 0) or 0)
                stats["tokens_per_second"] = toks / elapsed if elapsed > 0 else 0.0
            return result, stats

        cd_mod.model_generate_compiled = cd_wrap
        proc_mod.model_generate_compiled = cd_wrap

    orig_gen = proc_mod.Processor.generate

    def gen_wrap(self, *args, **kwargs):
        out_context = kwargs.get("out_context")
        label = "unknown"
        if out_context:
            try:
                label = "+".join(getattr(c, "value", str(c)) for c in out_context)
            except Exception:
                label = "unknown"
        if _STATE["first_generate_at"] is None:
            _STATE["first_generate_at"] = time.perf_counter()
        self._reset_generation_stats()
        result = orig_gen(self, *args, **kwargs)
        stats = getattr(self, "last_generation_stats", None) or {}
        toks = int(stats.get("generated_tokens", 0) or 0)
        elapsed = float(stats.get("elapsed_seconds", 0.0) or 0.0)
        _STATE["calls"].append(
            {"label": label, "generated_tokens": toks, "elapsed_seconds": elapsed, "stats": dict(stats)}
        )
        if "MAP" in label.upper():
            _STATE["main_tokens"] += toks
            _STATE["main_model_seconds"] += elapsed
        else:
            _STATE["timing_tokens"] += toks
            _STATE["timing_model_seconds"] += elapsed
        session = getattr(self, "turbo_session", None)
        if session is not None and getattr(session, "window_stats", None):
            _STATE["turbo_windows"] = list(session.window_stats)
        return result

    proc_mod.Processor.generate = gen_wrap

    def flush():
        cold_end = time.perf_counter()
        main_s = float(_STATE["main_model_seconds"])
        main_tok = int(_STATE["main_tokens"])
        first = _STATE.get("first_generate_at")
        include = cold_end - _COLD_START
        exclude = (cold_end - first) if first is not None else None
        e_vals = [
            float(w.get("turbo_E_accepted_per_verify", 0.0) or 0.0)
            for w in _STATE["turbo_windows"]
            if int(w.get("turbo_verify_steps", 0) or 0) >= 1
        ]
        payload = {
            "cold_start": _COLD_START,
            "cold_end": cold_end,
            "first_generate_at": first,
            "e2e_wall_s_include_load_jit": include,
            "e2e_wall_s_exclude_load_jit": exclude,
            "cold_start_seconds": (first - _COLD_START) if first is not None else None,
            "main_tokens": main_tok,
            "main_model_seconds": main_s,
            "main_tps": (main_tok / main_s) if main_s > 0 else 0.0,
            "ms_per_map_token": (
                (exclude * 1000.0 / main_tok)
                if (exclude is not None and main_tok > 0)
                else None
            ),
            "turbo_E_per_window": e_vals,
            "turbo_E_median": (
                float(sorted(e_vals)[len(e_vals) // 2]) if e_vals else None
            ),
            "turbo_E_mean": (sum(e_vals) / len(e_vals) if e_vals else None),
            "turbo_windows": _STATE["turbo_windows"],
            "calls": [
                {k: v for k, v in c.items() if k != "stats"}
                for c in _STATE["calls"]
            ],
            "timing_model": "cuda_synced_perf_counter_around_generate",
        }
        if _STATS_PATH.is_file():
            try:
                prev = json.loads(_STATS_PATH.read_text())
                if len(prev.get("calls") or []) > len(payload["calls"]):
                    return
            except Exception:
                pass
        _STATS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    atexit.register(flush)


import os as _os
import sys as _sys
_cwd = _os.getcwd()
if _cwd not in _sys.path:
    _sys.path.insert(0, _cwd)
_install()
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--draft-ckpt", type=Path, required=True)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--min-e", type=float, default=1.05)
    args = parser.parse_args()

    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / "output"
    output.mkdir(parents=True, exist_ok=True)
    stats_path = run_root / "s58_generation_stats.json"
    overlay = run_root / "overlay"
    overlay.mkdir(parents=True, exist_ok=True)
    _write(overlay / "s58_tiger_measure.py", OVERLAY)
    _write(overlay / "sitecustomize.py", "import s58_tiger_measure  # noqa: F401\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(overlay) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["MAPPERATORINATOR_TURBO"] = "1"
    env["MAPPERATORINATOR_TURBO_DRAFT_CKPT"] = str(args.draft_ckpt)
    env["MAPPERATORINATOR_TURBO_GAMMA"] = str(args.gamma)
    env["MAPPERATORINATOR_TURBO_DRAFT_CHAIN_GRAPH"] = "1"
    env["MAPPERATORINATOR_TURBO_STRUCTURAL_PROCESSORS"] = "0"
    env["MAPPERATORINATOR_S58_STATS_PATH"] = str(stats_path)
    cold_start = time.perf_counter()
    env["MAPPERATORINATOR_S58_COLD_START"] = str(cold_start)

    cmd = [
        str(args.python),
        "inference.py",
        "--config-name",
        "v32",
        f"audio_path={args.audio}",
        f"output_path={output}",
        "device=cuda",
        f"precision={args.precision}",
        "attn_implementation=sdpa",
        "use_server=false",
        "parallel=false",
        "cfg_scale=1.0",
        "num_beams=1",
        f"seed={args.seed}",
        "do_sample=true",
        "temperature=0.9",
        "top_p=0.9",
        "fast_decoder_loop=true",
        "super_timing_fast_loop=true",
        "output_type=[TIMING,MAP,SV]",
    ]
    with (run_root / "stdout.txt").open("w") as out, (
        run_root / "stderr.txt"
    ).open("w") as err:
        proc = subprocess.run(
            cmd, cwd=str(args.repo), env=env, check=False, stdout=out, stderr=err
        )
    process_wall = time.perf_counter() - cold_start
    _write(run_root / "exit_code.txt", f"{proc.returncode}\n")

    summary: dict[str, Any] = {
        "section": 58,
        "step": 2,
        "track": "turbo-on-tiger",
        "rc": proc.returncode,
        "process_wall_seconds": process_wall,
        "gamma": args.gamma,
        "precision": args.precision,
        "draft_ckpt": str(args.draft_ckpt),
        "repo_commit": subprocess.check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "campaign_tip_frozen": "55949274",
    }

    if proc.returncode != 0:
        summary["decision"] = "FAIL_runtime"
        summary["note"] = "inference failed; see stderr"
        _write(run_root / "summary.json", json.dumps(summary, indent=2) + "\n")
        raise SystemExit(proc.returncode)

    if not stats_path.is_file():
        summary["decision"] = "FAIL_no_stats"
        _write(run_root / "summary.json", json.dumps(summary, indent=2) + "\n")
        raise SystemExit(3)

    stats = json.loads(stats_path.read_text())
    e_med = stats.get("turbo_E_median")
    e_mean = stats.get("turbo_E_mean")
    main_tps = stats.get("main_tps")
    ms_tok = stats.get("ms_per_map_token")
    cold = stats.get("cold_start_seconds")
    exclude = stats.get("e2e_wall_s_exclude_load_jit")

    summary.update(
        {
            "turbo_E_median": e_med,
            "turbo_E_mean": e_mean,
            "turbo_E_per_window": stats.get("turbo_E_per_window"),
            "main_tps": main_tps,
            "ms_per_map_token": ms_tok,
            "cold_start_seconds": cold,
            "e2e_wall_s_exclude_load_jit": exclude,
            "main_tokens": stats.get("main_tokens"),
            "n_turbo_windows": len(stats.get("turbo_windows") or []),
        }
    )

    # Kill gates
    decision = "PASS"
    reasons: list[str] = []
    if e_med is None or float(e_med) < float(args.min_e):
        decision = "KILL_E_collapse"
        reasons.append(f"E_median={e_med} < min_e={args.min_e}")
    if exclude is not None and float(exclude) <= 0:
        decision = "KILL_negative_wall"
        reasons.append(f"e2e_exclude={exclude}")
    if ms_tok is not None and float(ms_tok) <= 0:
        decision = "KILL_negative_wall"
        reasons.append(f"ms_per_map_token={ms_tok}")
    if not (stats.get("turbo_windows") or []):
        decision = "KILL_no_turbo_windows"
        reasons.append("no turbo window stats harvested")

    summary["decision"] = decision
    summary["kill_reasons"] = reasons
    summary["note"] = (
        "STEP 2 speculative window on tiger. Strict rejection-sampling. "
        "No fused verify. No 500 claim."
    )
    _write(run_root / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if decision != "PASS":
        raise SystemExit(4)


if __name__ == "__main__":
    main()
