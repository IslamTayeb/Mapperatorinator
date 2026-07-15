from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.scout.exact_k4_multinomial import (  # noqa: E402
    run_exact_k4_multinomial_probe,
)


def _csv_integers(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=_csv_integers,
        default=(7, 12345, 987654, 42),
    )
    parser.add_argument(
        "--eos-positions",
        type=_csv_integers,
        default=(1, 2, 3, 4),
    )
    args = parser.parse_args()

    if args.output.suffix != ".json":
        raise ValueError("probe output must be a JSON file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_exact_k4_multinomial_probe(
        seeds=args.seeds,
        eos_positions=args.eos_positions,
    )
    payload = result.to_dict()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if not result.exact_k4_feasible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
