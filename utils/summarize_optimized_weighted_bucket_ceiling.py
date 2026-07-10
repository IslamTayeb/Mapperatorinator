"""Write the audited model-free B2 queue ceiling and target requirement."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_CONTRACT_PATH = (
    REPO_ROOT / "osuT5/osuT5/inference/optimized/batch/weighted_bucket.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_mapperatorinator_weighted_bucket_ceiling", _CONTRACT_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load weighted bucket contracts from {_CONTRACT_PATH}")
_CONTRACT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONTRACT
_SPEC.loader.exec_module(_CONTRACT)
build_model_free_ceiling_report = _CONTRACT.build_model_free_ceiling_report
validate_model_free_ceiling_report = _CONTRACT.validate_model_free_ceiling_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", type=Path, required=True)
    parser.add_argument("--packed-prefill-report", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    report = build_model_free_ceiling_report(
        cli.profile,
        packed_prefill_report_path=cli.packed_prefill_report,
    )
    validate_model_free_ceiling_report(report)
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
