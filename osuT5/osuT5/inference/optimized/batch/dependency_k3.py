"""CPU-only dependency-aware ceiling gate for a five-song K3 scout.

The parent weighted-ceiling report deliberately treated setup as either free
or fully charged.  This module places the accepted linear setup charge at the
production dependency boundary: the first window of each song may be prepared
before first-main, while the remaining nine windows cannot be prepared until
their predecessor has generated its output.

This is arithmetic and evidence validation only.  It has no CUDA, model,
scheduler, runtime, or server integration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


_WEIGHTED_BUCKET_PATH = Path(__file__).with_name("weighted_bucket.py")
_WEIGHTED_BUCKET_SPEC = importlib.util.spec_from_file_location(
    "_mapperatorinator_dependency_k3_weighted_bucket",
    _WEIGHTED_BUCKET_PATH,
)
if _WEIGHTED_BUCKET_SPEC is None or _WEIGHTED_BUCKET_SPEC.loader is None:
    raise RuntimeError(f"cannot load weighted ceiling contract from {_WEIGHTED_BUCKET_PATH}")
_WEIGHTED_BUCKET = importlib.util.module_from_spec(_WEIGHTED_BUCKET_SPEC)
sys.modules[_WEIGHTED_BUCKET_SPEC.name] = _WEIGHTED_BUCKET
_WEIGHTED_BUCKET_SPEC.loader.exec_module(_WEIGHTED_BUCKET)
validate_model_free_ceiling_report = (
    _WEIGHTED_BUCKET.validate_model_free_ceiling_report
)


DEPENDENCY_K3_SCHEMA_VERSION = 1
DEPENDENCY_K3_GATE = "accepted_five_song_dependency_aware_k3_ceiling"
PARENT_REPORT_FILE_SHA256 = (
    "44a680ab29867e3aea8dde713127bdb154ef42a316fd2834bc75afbbc0927fc9"
)
PARENT_SCHEMA_VERSION = 1
PARENT_GATE = "accepted_five_song_model_free_b2_ceiling"
PARENT_SOURCE_JOB_ID = "49543717"
PARENT_SOURCE_COMMIT = "a709b86c37484c4ef9754d582e4506939273bf67"
PARENT_K3_WALL_SECONDS = 9.293543316309831
PARENT_K3_MAIN_TPS = 640.7674443771291
FIVE_MAIN_TOKENS = 5955
SONG_COUNT = 5
WINDOWS_PER_SONG = 10
TOTAL_WINDOWS = SONG_COUNT * WINDOWS_PER_SONG
INITIAL_WINDOWS_EXCLUDED = SONG_COUNT
DEPENDENT_TRANSITIONS = TOTAL_WINDOWS - INITIAL_WINDOWS_EXCLUDED
SERIAL_B1_PREFILL_BATCH8_SECONDS = 0.34391318541020155
PACKER_BATCH8_SECONDS = 0.04675073456019163
SETUP_SOURCE_BATCH_SIZE = 8
PER_WINDOW_SETUP_SECONDS = (
    SERIAL_B1_PREFILL_BATCH8_SECONDS + PACKER_BATCH8_SECONDS
) / SETUP_SOURCE_BATCH_SIZE
TARGET_TPS = 500.0
STRONG_TARGET_TPS = 525.0
STRONG_MAX_WALL_SECONDS = FIVE_MAIN_TOKENS / STRONG_TARGET_TPS


def canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(left: Any, right: float) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isfinite(float(left))
        and math.isclose(float(left), right, rel_tol=1e-12, abs_tol=1e-12)
    )


def _validate_parent(
        parent: Mapping[str, Any],
        *,
        parent_file_sha256: str,
) -> Sequence[Mapping[str, Any]]:
    if parent_file_sha256 != PARENT_REPORT_FILE_SHA256:
        raise ValueError("dependency-aware K3 parent report file SHA-256 changed.")
    validate_model_free_ceiling_report(parent)
    source = parent["source"]
    k3 = parent["five_request_ideal_k3"]
    profiles = source["profile_inputs"]
    if (
            parent["schema_version"] != PARENT_SCHEMA_VERSION
            or parent["gate"] != PARENT_GATE
            or source["job_id"] != PARENT_SOURCE_JOB_ID
            or source["commit"] != PARENT_SOURCE_COMMIT
            or k3["main_tokens"] != FIVE_MAIN_TOKENS
            or not _close(k3["ideal_wall_seconds"], PARENT_K3_WALL_SECONDS)
            or not _close(
                k3["ideal_scheduler_wall_tokens_per_second"],
                PARENT_K3_MAIN_TPS,
            )
            or not isinstance(profiles, Sequence)
            or isinstance(profiles, (str, bytes))
            or len(profiles) != SONG_COUNT
    ):
        raise ValueError("dependency-aware K3 parent contract changed.")
    setup = source["setup_input"]
    if (
            setup["batch_size"] != SETUP_SOURCE_BATCH_SIZE
            or setup["linearized_window_count"] != 100
            or not _close(
                setup["serial_b1_prefill_wall_seconds_for_batch8"],
                SERIAL_B1_PREFILL_BATCH8_SECONDS,
            )
            or not _close(
                setup["packer_wall_seconds_for_batch8"],
                PACKER_BATCH8_SECONDS,
            )
    ):
        raise ValueError("dependency-aware K3 setup source changed.")
    return profiles


def _transition_rows(
        profiles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for song_index, profile in enumerate(profiles):
        records = profile["main_records"]
        if len(records) != WINDOWS_PER_SONG:
            raise ValueError("dependency-aware K3 song window count changed.")
        for sequence_index in range(1, WINDOWS_PER_SONG):
            if records[sequence_index]["sequence_index"] != sequence_index:
                raise ValueError("dependency-aware K3 sequence order changed.")
            rows.append({
                "transition_index": len(rows),
                "song_index": song_index,
                "song_id": profile["song_id"],
                "source_run_index": profile["run_index"],
                "predecessor_sequence_index": sequence_index - 1,
                "sequence_index": sequence_index,
                "setup_seconds": PER_WINDOW_SETUP_SECONDS,
                "placement": "after_predecessor_output_inside_first_main_to_last_main",
            })
    if len(rows) != DEPENDENT_TRANSITIONS:
        raise ValueError("dependency-aware K3 transition count changed.")
    return rows


def _scenario(
        *,
        policy: str,
        charged_setup_seconds: float,
        excluded_setup_seconds: float,
) -> dict[str, Any]:
    wall = PARENT_K3_WALL_SECONDS + charged_setup_seconds
    tps = FIVE_MAIN_TOKENS / wall
    strong_pass = bool(
        tps > STRONG_TARGET_TPS
        and not math.isclose(
            tps, STRONG_TARGET_TPS, rel_tol=0.0, abs_tol=1e-12
        )
    )
    return {
        "policy": policy,
        "main_tokens": FIVE_MAIN_TOKENS,
        "setup_free_k3_wall_seconds": PARENT_K3_WALL_SECONDS,
        "charged_setup_seconds": charged_setup_seconds,
        "excluded_pre_first_main_setup_seconds": excluded_setup_seconds,
        "scheduler_wall_seconds": wall,
        "scheduler_wall_main_tokens_per_second": tps,
        "headroom_over_500_fraction": tps / TARGET_TPS - 1.0,
        "strong_525_bar_pass": strong_pass,
    }


def build_dependency_aware_k3_report(
        parent: Mapping[str, Any],
        *,
        parent_file_sha256: str,
) -> dict[str, Any]:
    profiles = _validate_parent(
        parent, parent_file_sha256=parent_file_sha256
    )
    transitions = _transition_rows(profiles)
    transition_seconds = math.fsum(
        float(row["setup_seconds"]) for row in transitions
    )
    all_setup_seconds = PER_WINDOW_SETUP_SECONDS * TOTAL_WINDOWS
    initial_setup_seconds = PER_WINDOW_SETUP_SECONDS * INITIAL_WINDOWS_EXCLUDED
    if not math.isclose(
            transition_seconds,
            all_setup_seconds - initial_setup_seconds,
            rel_tol=1e-12,
            abs_tol=1e-12,
    ):
        raise ValueError("dependency-aware K3 setup accounting is inconsistent.")
    setup_free = _scenario(
        policy="fantasy_all_window_setup_excluded",
        charged_setup_seconds=0.0,
        excluded_setup_seconds=all_setup_seconds,
    )
    all_setup = _scenario(
        policy="all_fifty_window_setups_charged",
        charged_setup_seconds=all_setup_seconds,
        excluded_setup_seconds=0.0,
    )
    dependency_aware = _scenario(
        policy="five_initial_windows_excluded_forty_five_transitions_charged",
        charged_setup_seconds=transition_seconds,
        excluded_setup_seconds=initial_setup_seconds,
    )
    gpu_authorized = bool(dependency_aware["strong_525_bar_pass"])
    return {
        "schema_version": DEPENDENCY_K3_SCHEMA_VERSION,
        "gate": DEPENDENCY_K3_GATE,
        "claim_scope": "cpu_only_model_free_dependency_placement_not_gpu_runtime",
        "source_contract": {
            "parent_report_file_sha256": parent_file_sha256,
            "parent_schema_version": parent["schema_version"],
            "parent_gate": parent["gate"],
            "source_job_id": parent["source"]["job_id"],
            "source_commit": parent["source"]["commit"],
            "setup_job_id": parent["source"]["setup_input"]["job_id"],
            "setup_commit": parent["source"]["setup_input"]["commit"],
            "setup_report_sha256": parent["source"]["setup_input"][
                "report_sha256"
            ],
        },
        "workload": {
            "song_count": SONG_COUNT,
            "windows_per_song": WINDOWS_PER_SONG,
            "total_windows": TOTAL_WINDOWS,
            "initial_windows_excluded_before_first_main": (
                INITIAL_WINDOWS_EXCLUDED
            ),
            "dependent_transitions_charged": DEPENDENT_TRANSITIONS,
            "main_tokens": FIVE_MAIN_TOKENS,
            "song_ids": [profile["song_id"] for profile in profiles],
        },
        "setup_contract": {
            "serial_b1_prefill_batch8_seconds": (
                SERIAL_B1_PREFILL_BATCH8_SECONDS
            ),
            "packer_batch8_seconds": PACKER_BATCH8_SECONDS,
            "source_batch_size": SETUP_SOURCE_BATCH_SIZE,
            "per_window_setup_seconds": PER_WINDOW_SETUP_SECONDS,
            "all_fifty_windows_setup_seconds": all_setup_seconds,
            "five_initial_windows_setup_seconds": initial_setup_seconds,
            "forty_five_dependent_transitions_setup_seconds": transition_seconds,
            "provenance_scope": "accepted optimistic linearized synthetic setup charge",
        },
        "transition_ledger": {
            "count": len(transitions),
            "sha256": canonical_json_sha256(transitions),
            "total_setup_seconds": transition_seconds,
            "rows": transitions,
        },
        "scenarios": {
            "setup_free": setup_free,
            "all_setup_charged": all_setup,
            "dependency_aware": dependency_aware,
        },
        "strong_gate": {
            "target_main_tokens_per_second": STRONG_TARGET_TPS,
            "maximum_scheduler_wall_seconds": STRONG_MAX_WALL_SECONDS,
            "dependency_aware_scheduler_wall_seconds": dependency_aware[
                "scheduler_wall_seconds"
            ],
            "dependency_aware_main_tokens_per_second": dependency_aware[
                "scheduler_wall_main_tokens_per_second"
            ],
            "wall_seconds_over_bar": dependency_aware["scheduler_wall_seconds"]
            - STRONG_MAX_WALL_SECONDS,
            "pass": gpu_authorized,
        },
        "decision": {
            "k3_gpu_scout_authorized": gpu_authorized,
            "reason": (
                "dependency-aware fantasy clears 525 main tok/s"
                if gpu_authorized
                else "dependency-aware fantasy is below the 525 tok/s strong bar"
            ),
            "scheduler_or_runtime_wiring_authorized": False,
            "server_work_authorized": False,
            "gpu_work_executed": False,
            "runtime_code_touched": False,
        },
    }


def validate_dependency_aware_k3_report(
        report: Mapping[str, Any],
        *,
        parent_report: Mapping[str, Any],
        parent_file_sha256: str,
) -> None:
    expected = build_dependency_aware_k3_report(
        parent_report, parent_file_sha256=parent_file_sha256
    )
    if report != expected:
        raise ValueError(
            "dependency-aware K3 report differs from source-derived arithmetic."
        )
    ledger = report["transition_ledger"]
    rows = ledger["rows"]
    if (
            ledger["count"] != DEPENDENT_TRANSITIONS
            or ledger["sha256"] != canonical_json_sha256(rows)
            or not _close(
                ledger["total_setup_seconds"],
                math.fsum(float(row["setup_seconds"]) for row in rows),
            )
    ):
        raise ValueError("dependency-aware K3 transition ledger failed validation.")
    scenarios = report["scenarios"]
    if (
            not _close(
                scenarios["setup_free"]["scheduler_wall_main_tokens_per_second"],
                PARENT_K3_MAIN_TPS,
            )
            or scenarios["dependency_aware"]["strong_525_bar_pass"] is not False
            or report["strong_gate"]["pass"] is not False
            or report["decision"]["k3_gpu_scout_authorized"] is not False
    ):
        raise ValueError("dependency-aware K3 rejection decision changed.")


def load_parent_report(path: Path) -> tuple[dict[str, Any], str]:
    observed_sha256 = file_sha256(path)
    if observed_sha256 != PARENT_REPORT_FILE_SHA256:
        raise ValueError("dependency-aware K3 parent report file SHA-256 changed.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dependency-aware K3 parent must be a JSON object.")
    _validate_parent(payload, parent_file_sha256=observed_sha256)
    return payload, observed_sha256
