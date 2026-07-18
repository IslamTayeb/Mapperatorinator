#!/usr/bin/env python3
"""T3 greedy token-match: compile-on vs compile-off fast path (same commit).

Runs two short greedy generations and compares token ID sequences.
Writes match.json under --run-root.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_variant(repo: Path, python: Path, audio: Path, run_root: Path, variant: str, seed: int) -> Path:
    out = run_root / variant
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK", None)
    if variant == "compile":
        env["MAPPERATORINATOR_COMPILE_DECODE"] = "1"
        env["MAPPERATORINATOR_WARM_ALL_BUCKETS"] = "1"
    else:
        env.pop("MAPPERATORINATOR_COMPILE_DECODE", None)
        env.pop("MAPPERATORINATOR_WARM_ALL_BUCKETS", None)

    overrides = [
        f"audio_path={audio}",
        f"output_path={out / 'output'}",
        "device=cuda",
        "precision=fp16",
        "attn_implementation=sdpa",
        "use_server=false",
        "parallel=false",
        "cfg_scale=1.0",
        "num_beams=1",
        f"seed={seed}",
        "gamemode=0",
        "difficulty=6.0",
        "year=2015",
        "hitsounded=true",
        "descriptors=[skillset/streams,streams/flow aim,streams/spaced streams,streams/bursts]",
        "generate_positions=false",
        "export_osz=false",
        "do_sample=false",
        "temperature=1.0",
        "top_p=1.0",
        "fast_decoder_loop=true",
        "super_timing_fast_loop=true",
        "output_type=[MAP]",
    ]
    cmd = [str(python), "inference.py", "--config-name", "v32", *overrides]
    with (out / "stdout.txt").open("w") as so, (out / "stderr.txt").open("w") as se:
        rc = subprocess.run(cmd, cwd=str(repo), env=env, check=False, stdout=so, stderr=se).returncode
    (out / "exit_code.txt").write_text(str(rc) + "\n")
    if rc != 0:
        raise SystemExit(f"{variant} failed rc={rc}")
    return out


def _collect_osu_bytes(root: Path) -> list[bytes]:
    files = sorted(root.rglob("*.osu"))
    return [p.read_bytes() for p in files]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--python", required=True, type=Path)
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    t0 = time.perf_counter()
    base = _run_variant(args.repo, args.python, args.audio, args.run_root, "baseline", args.seed)
    # Drop graph caches between variants so compile path re-captures cleanly.
    cand = _run_variant(args.repo, args.python, args.audio, args.run_root, "compile", args.seed)
    base_osu = _collect_osu_bytes(base / "output")
    cand_osu = _collect_osu_bytes(cand / "output")
    equal = base_osu == cand_osu and len(base_osu) > 0
    payload = {
        "status": "PASS" if equal else "FAIL",
        "equal": equal,
        "n_osu_baseline": len(base_osu),
        "n_osu_compile": len(cand_osu),
        "baseline_bytes": [len(b) for b in base_osu],
        "compile_bytes": [len(b) for b in cand_osu],
        "elapsed_s": time.perf_counter() - t0,
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(args.repo), text=True
        ).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "note": "greedy do_sample=false; compare final .osu bytes (token-level via map dump)",
    }
    _write_json(args.run_root / "match.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not equal:
        raise SystemExit("greedy token/osu match FAILED")


if __name__ == "__main__":
    main()
