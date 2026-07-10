"""Replay profile token transcripts through the CPU-only generated-prefix n-gram scout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.speculative import (  # noqa: E402
    CAMPAIGN_SPECULATION_K,
    DEFAULT_ADVANCE_THRESHOLD,
    DEFAULT_MAX_NGRAM_LENGTH,
    run_generated_prefix_ngram_profile_scout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profiles",
        nargs="+",
        type=Path,
        help="One or more profile_inference JSONs containing per-window generated_token_ids.",
    )
    parser.add_argument("--profile-label", default="main_generation")
    parser.add_argument(
        "--speculation-k",
        nargs="+",
        type=int,
        choices=CAMPAIGN_SPECULATION_K,
        default=list(CAMPAIGN_SPECULATION_K),
        help="Bounded K values to replay; defaults to 2 4 8.",
    )
    parser.add_argument("--max-ngram-length", type=int, default=DEFAULT_MAX_NGRAM_LENGTH)
    parser.add_argument(
        "--advance-threshold",
        type=float,
        default=DEFAULT_ADVANCE_THRESHOLD,
        help="Minimum structural target-call reduction needed to justify the GPU gate.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_generated_prefix_ngram_profile_scout(
        args.profiles,
        profile_label=args.profile_label,
        speculation_k_values=args.speculation_k,
        max_ngram_length=args.max_ngram_length,
        advance_threshold=args.advance_threshold,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
