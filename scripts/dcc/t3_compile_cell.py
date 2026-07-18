#!/usr/bin/env python3
"""T3 compile-then-capture cell: tiger-base ms/map-token + main_tps + cold_start.

Variants:
  baseline — plain CUDA graphs (MAPPERATORINATOR_COMPILE_DECODE unset)
  compile  — full decode-step Inductor compile-then-capture + warm-all-buckets
             (eager mono+temp ``_tail``; mode=default)

Gates (reported in summary; promote decided offline under T3 relaxation):
  A5000 main-gen +≥10% vs like-with-like baseline
  2080 Ti no-regression
  coherent maps + T5 KS (greedy byte-match optional / documented drift OK)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TIGER_OVERLAY = r'''
import atexit
import json
import os
import time
from pathlib import Path

import torch

_STATS_PATH = Path(os.environ["MAPPERATORINATOR_W1_STATS_PATH"])
_COLD_START = float(os.environ["MAPPERATORINATOR_W1_COLD_START"])
_STATE = {
    "first_generate_at": None,
    "main_tokens": 0,
    "main_model_seconds": 0.0,
    "timing_tokens": 0,
    "timing_model_seconds": 0.0,
    "calls": [],
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
            {"label": label, "generated_tokens": toks, "elapsed_seconds": elapsed}
        )
        if "MAP" in label.upper():
            _STATE["main_tokens"] += toks
            _STATE["main_model_seconds"] += elapsed
        else:
            _STATE["timing_tokens"] += toks
            _STATE["timing_model_seconds"] += elapsed
        return result

    proc_mod.Processor.generate = gen_wrap

    def flush():
        cold_end = time.perf_counter()
        main_s = float(_STATE["main_model_seconds"])
        main_tok = int(_STATE["main_tokens"])
        first = _STATE.get("first_generate_at")
        include = cold_end - _COLD_START
        exclude = (cold_end - first) if first is not None else None
        payload = {
            "cold_start": _COLD_START,
            "cold_end": cold_end,
            "first_generate_at": first,
            "e2e_wall_s_include_load_jit": include,
            "e2e_wall_s_exclude_load_jit": exclude,
            "load_jit_proxy_seconds": (first - _COLD_START) if first is not None else None,
            "main_tokens": main_tok,
            "main_model_seconds": main_s,
            "main_gen_model_tps": (main_tok / main_s) if main_s > 0 else 0.0,
            "timing_tokens": int(_STATE["timing_tokens"]),
            "timing_model_seconds": float(_STATE["timing_model_seconds"]),
            "calls": _STATE["calls"],
            "timing_model": "cuda_synced_perf_counter_around_generate",
            "compile_decode": os.environ.get("MAPPERATORINATOR_COMPILE_DECODE"),
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32", "bf16"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--variant", required=True, choices=("baseline", "compile"))
    parser.add_argument("--expected-gpu-substr", default="")
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--no-do-sample", action="store_false", dest="do_sample")
    args = parser.parse_args()

    run_root: Path = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / "output"
    output.mkdir(parents=True, exist_ok=True)

    if args.expected_gpu_substr:
        import torch
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        if args.expected_gpu_substr not in name:
            raise SystemExit(f"GPU mismatch: want substring {args.expected_gpu_substr!r}, got {name!r}")

    stats_path = run_root / "tiger_generation_stats.json"
    overlay = run_root / "overlay"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "w1_tiger_measure.py").write_text(TIGER_OVERLAY, encoding="utf-8")
    (overlay / "sitecustomize.py").write_text(
        "import w1_tiger_measure  # noqa: F401\n",
        encoding="utf-8",
    )

    cold_start = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(overlay) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["MAPPERATORINATOR_W1_STATS_PATH"] = str(stats_path)
    env["MAPPERATORINATOR_W1_COLD_START"] = str(cold_start)
    env.pop("MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK", None)

    if args.variant == "compile":
        env["MAPPERATORINATOR_COMPILE_DECODE"] = "1"
        env["MAPPERATORINATOR_WARM_ALL_BUCKETS"] = "1"
        # Full-step is package default; leave harvest-4 sub-ops unset.
        env.pop("MAPPERATORINATOR_COMPILE_SUBOPS", None)
        env.pop("MAPPERATORINATOR_COMPILE_FULL_STEP", None)
    else:
        env.pop("MAPPERATORINATOR_COMPILE_DECODE", None)
        env.pop("MAPPERATORINATOR_WARM_ALL_BUCKETS", None)
        env.pop("MAPPERATORINATOR_COMPILE_SUBOPS", None)
        env.pop("MAPPERATORINATOR_COMPILE_FULL_STEP", None)

    overrides = [
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
        "gamemode=0",
        "difficulty=6.0",
        "year=2015",
        "hitsounded=true",
        "descriptors=[skillset/streams,streams/flow aim,streams/spaced streams,streams/bursts]",
        "generate_positions=false",
        "export_osz=false",
        "temperature=0.9",
        "top_p=0.9",
        "fast_decoder_loop=true",
        "super_timing_fast_loop=true",
        "output_type=[TIMING,MAP,SV]",
        f"do_sample={str(args.do_sample).lower()}",
    ]

    cmd = [
        str(args.python),
        "inference.py",
        "--config-name",
        "v32",
        *overrides,
    ]
    with (run_root / "stdout.txt").open("w", encoding="utf-8") as out, (
        run_root / "stderr.txt"
    ).open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(args.repo), env=env, check=False, stdout=out, stderr=err)
    process_wall = time.perf_counter() - cold_start
    (run_root / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"inference failed rc={proc.returncode}")
    if not stats_path.is_file():
        raise SystemExit("stats dump missing — measurement overlay did not flush")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    map_tokens = 0
    map_model_s = 0.0
    for call in stats.get("calls") or []:
        label = str(call.get("label", "")).upper()
        if "MAP" in label:
            map_tokens += int(call.get("generated_tokens", 0) or 0)
            map_model_s += float(call.get("elapsed_seconds", 0.0) or 0.0)
    if map_tokens <= 0:
        map_tokens = int(stats.get("main_tokens") or 0)
        map_model_s = float(stats.get("main_model_seconds") or 0.0)

    exclude = stats.get("e2e_wall_s_exclude_load_jit")
    include = stats.get("e2e_wall_s_include_load_jit")
    cold = stats.get("load_jit_proxy_seconds")
    ms_per = (float(exclude) * 1000.0 / map_tokens) if exclude and map_tokens > 0 else None
    main_tps = (map_tokens / map_model_s) if map_model_s > 0 else float(stats.get("main_gen_model_tps") or 0.0)

    summary = {
        "status": "OK",
        "variant": args.variant,
        "precision": args.precision,
        "seed": args.seed,
        "do_sample": args.do_sample,
        "compile_decode": args.variant == "compile",
        "compile_kind": "full_step" if args.variant == "compile" else None,
        "compile_mode": os.environ.get("MAPPERATORINATOR_COMPILE_MODE", "default"),
        "repo": str(args.repo),
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(args.repo), text=True
        ).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None,
        "map_tokens": map_tokens,
        "main_model_seconds": map_model_s,
        "main_gen_model_tps": main_tps,
        "e2e_wall_s_exclude_load_jit": exclude,
        "e2e_wall_s_include_load_jit": include,
        "cold_start_seconds": cold,
        "ms_per_map_token": ms_per,
        "process_wall_seconds": process_wall,
        "calls": stats.get("calls"),
        "tiger_stats_path": str(stats_path),
        "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }
    _write_json(run_root / "summary.json", summary)
    print(
        f"OK variant={args.variant} ms/map-token={ms_per:.4f} "
        f"main_tps={main_tps:.2f} cold_start={cold:.2f}s "
        f"exclude={exclude:.2f}s map_tok={map_tokens}"
        if ms_per is not None and cold is not None and exclude is not None
        else json.dumps(summary)
    )


if __name__ == "__main__":
    main()
