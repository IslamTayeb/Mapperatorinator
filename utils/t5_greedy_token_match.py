#!/usr/bin/env python3
"""T5 greedy token-match gate (CPU-first).

Compares baseline vs candidate token ID streams bit-for-bit.
Sources (any mix):
  - raw JSON list / {"tokens"|"token_ids"|"result_tokens"|"generated_token_ids": [...]}
  - inference profile JSON (generation[*].generated_token_ids)
  - newline-separated integer dump

Used where defined:
  - T3 compile-then-capture vs uncompiled fast path
  - tip regression vs frozen 55949274 reference dumps (when available)
  - turbo TIER1a (when unparked) vs exact engine

Does not invent relaxed acceptance (§34 standing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


def _as_int_list(values: Iterable[Any]) -> list[int]:
    out: list[int] = []
    for v in values:
        out.append(int(v))
    return out


def load_token_ids(source: Path | str | Sequence[int] | dict[str, Any]) -> list[int]:
    """Load a flat token-id sequence from several scout dump shapes."""
    if isinstance(source, (list, tuple)):
        return _as_int_list(source)
    if isinstance(source, dict):
        return _tokens_from_obj(source)
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ".jsonl"} or text.lstrip().startswith(("{", "[")):
        payload = json.loads(text)
        return _tokens_from_obj(payload)
    # Plain whitespace / comma separated ints
    parts = text.replace(",", " ").split()
    return [int(p) for p in parts if p]


def _tokens_from_obj(obj: Any) -> list[int]:
    if isinstance(obj, list):
        if obj and isinstance(obj[0], (int, float)):
            return _as_int_list(obj)
        # list of generation records
        flat: list[int] = []
        for row in obj:
            if isinstance(row, dict) and "generated_token_ids" in row:
                flat.extend(_as_int_list(row["generated_token_ids"]))
            elif isinstance(row, (list, tuple)):
                flat.extend(_as_int_list(row))
        if flat:
            return flat
        raise ValueError("list payload has no token ids")

    if not isinstance(obj, dict):
        raise ValueError(f"unsupported token payload type: {type(obj).__name__}")

    for key in (
        "tokens",
        "token_ids",
        "result_tokens",
        "generated_token_ids",
        "greedy_tokens",
        "candidate_tokens",
        "baseline_tokens",
    ):
        if key in obj and isinstance(obj[key], (list, tuple)):
            return _as_int_list(obj[key])

    # Nested canary / gate shapes
    for nest in ("tier1a_greedy_canary", "greedy_token_match", "t5_quality_gates"):
        inner = obj.get(nest)
        if isinstance(inner, dict):
            for key in ("tokens", "token_ids", "result_tokens", "candidate_tokens"):
                if key in inner and isinstance(inner[key], (list, tuple)):
                    return _as_int_list(inner[key])

    gen = obj.get("generation")
    if isinstance(gen, list):
        flat = []
        for row in gen:
            if isinstance(row, dict) and "generated_token_ids" in row:
                flat.extend(_as_int_list(row["generated_token_ids"]))
        if flat:
            return flat

    stages = obj.get("stages")
    if isinstance(stages, dict):
        for stage in stages.values():
            if isinstance(stage, dict) and "generated_token_ids" in stage:
                return _as_int_list(stage["generated_token_ids"])

    raise ValueError(
        "could not find token ids (expected tokens/token_ids/result_tokens/"
        "generated_token_ids or profile generation[].generated_token_ids)"
    )


def compare_token_ids(
    baseline: Sequence[int],
    candidate: Sequence[int],
    *,
    label: str = "greedy",
    max_context: int = 5,
) -> dict[str, Any]:
    base = _as_int_list(baseline)
    cand = _as_int_list(candidate)
    n = min(len(base), len(cand))
    first_mismatch = next(
        (i for i in range(n) if base[i] != cand[i]),
        None if len(base) == len(cand) else n,
    )
    passed = base == cand
    ctx = None
    if first_mismatch is not None:
        lo = max(0, int(first_mismatch) - 2)
        hi = min(max(len(base), len(cand)), int(first_mismatch) + max_context)
        ctx = {
            "abs_index": int(first_mismatch),
            "baseline_slice": base[lo:hi],
            "candidate_slice": cand[lo:hi],
            "slice_start": lo,
        }
    return {
        "gate": "greedy_token_match",
        "label": label,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "baseline_len": len(base),
        "candidate_len": len(cand),
        "first_mismatch": first_mismatch,
        "context": ctx,
        "note": (
            "Bit-exact greedy token match. Not a 500 claim. "
            "§34: do not fold relaxed acceptance into turbo."
        ),
    }


def compare_sources(
    baseline_source: Path | str | Sequence[int] | dict[str, Any],
    candidate_source: Path | str | Sequence[int] | dict[str, Any],
    *,
    label: str = "greedy",
) -> dict[str, Any]:
    return compare_token_ids(
        load_token_ids(baseline_source),
        load_token_ids(candidate_source),
        label=label,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--label", default="greedy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit 2 on mismatch (strict gate).",
    )
    args = ap.parse_args()

    payload = compare_sources(args.baseline, args.candidate, label=args.label)
    payload["baseline_path"] = str(args.baseline)
    payload["candidate_path"] = str(args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"pass": payload["pass"], "out": str(args.out), "first_mismatch": payload["first_mismatch"]}))
    if args.require_pass and not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
