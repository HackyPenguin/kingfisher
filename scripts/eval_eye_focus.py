#!/usr/bin/env python3
"""Read-only diagnostic evaluator for advisory eye-focus fields.

It compares fields already present in analysis records; it never invokes the
pipeline or writes metadata.  ``combined_score`` is displayed only when it was
already recorded by an external experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATABASE_NAME = "kingfisher_database.csv"
ADVISORY_STATUSES = frozenset(("unknown", "error", "no_eye", "detected"))
ADVISORY_DETECTOR = "heuristic_eye_focus"
ADVISORY_VERSION = 1


@dataclass(frozen=True)
class CaseResult:
    scene: str
    baseline_primary: int
    prototype_primary: int
    baseline_quality: float
    prototype_quality: float
    prototype_combined: float
    changed: bool
    missing_reasons: tuple[str, ...]
    crops: tuple[dict[str, Any], ...]


class EvalInputError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose recorded advisory eye-focus fields without changing analysis output.")
    parser.add_argument("--input", required=True, help="Output folder, kingfisher_database.csv, or JSON analysis record file.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum deterministic filename-ordered cases to print (default: 20).")
    parser.add_argument("--show", choices=("changed", "all", "missing"), default="changed", help="Print changed, all, or missing-metadata cases.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = [evaluate_record(record) for record in load_records(Path(args.input))]
    except (EvalInputError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {escape_terminal_text(str(exc))}", file=sys.stderr)
        return 2
    if not results:
        print("No analysis records found.", file=sys.stderr)
        return 1
    results.sort(key=lambda item: item.scene)
    print_summary(results)
    print_cases(results, show=args.show, limit=max(1, args.limit))
    return 0


def load_records(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise EvalInputError(f"Input path does not exist: {input_path}")
    if input_path.is_dir():
        input_path = input_path / DATABASE_NAME
        if not input_path.exists():
            raise EvalInputError(f"Directory input must contain {DATABASE_NAME}: {input_path.parent}")
    if input_path.suffix.lower() == ".csv":
        return load_csv_records(input_path)
    if input_path.suffix.lower() == ".json":
        return load_json_records(input_path)
    raise EvalInputError("Unsupported input. Expected a folder, .csv, or .json file of analysis outputs.")


def load_csv_records(csv_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            crops_json = (row.get("crops_json") or "").strip()
            crops: list[dict[str, Any]] = []
            if crops_json:
                try:
                    parsed = json.loads(crops_json)
                except json.JSONDecodeError as exc:
                    raise EvalInputError(f"Invalid crops_json in {csv_path} for {row.get('filename', '<unknown>')}: {exc}") from exc
                if not isinstance(parsed, list):
                    raise EvalInputError(f"Expected crops_json list for {row.get('filename', '<unknown>')}")
                crops = [coerce_crop_dict(item) for item in parsed]
            records.append({"scene": row.get("filename") or row.get("scene") or f"row-{len(records)}", "crops": crops})
    return records


def load_json_records(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload["records"] if isinstance(payload.get("records"), list) else [payload]
    if not isinstance(payload, list):
        raise EvalInputError(f"Expected JSON array or object with records in {json_path}")
    records = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise EvalInputError(f"Record {index} in {json_path} is not an object")
        crops = item.get("crops")
        if crops is None and isinstance(item.get("crops_json"), str):
            try:
                crops = json.loads(item["crops_json"])
            except json.JSONDecodeError as exc:
                raise EvalInputError(f"Invalid crops_json in record {index}: {exc}") from exc
        if crops is None:
            crops = []
        if not isinstance(crops, list):
            raise EvalInputError(f"Record {index} crops must be a list")
        records.append({"scene": item.get("filename") or item.get("scene") or item.get("scene_id") or f"record-{index}", "crops": [coerce_crop_dict(crop) for crop in crops]})
    return records


def coerce_crop_dict(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise EvalInputError(f"Crop entry is not an object: {item!r}")
    return dict(item)


def numeric_score(value: Any, default: float) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_unit_interval(value: Any) -> bool:
    return is_finite_number(value) and 0.0 <= float(value) <= 1.0


def evaluate_record(record: dict[str, Any]) -> CaseResult:
    scene, crops = str(record.get("scene") or "<unknown>"), list(record.get("crops") or [])
    missing: list[str] = []
    if not crops:
        missing.append("no_crops")
        return CaseResult(scene, -1, -1, -1.0, -1.0, -1.0, False, tuple(missing), tuple())
    for index, crop in enumerate(crops):
        if "quality" not in crop:
            missing.append(f"crop_{index}_missing_quality")
        elif not is_finite_number(crop["quality"]):
            missing.append(f"crop_{index}_invalid_quality")
        if "combined_score" not in crop:
            missing.append(f"crop_{index}_missing_combined_score")
        elif not is_finite_number(crop["combined_score"]):
            missing.append(f"crop_{index}_invalid_combined_score")
        absent = [field for field in ("eye_detected", "eye_confidence", "eye_focus_score") if field not in crop]
        if absent:
            missing.append(f"crop_{index}_missing_eye_fields:{','.join(absent)}")
        for field in ("eye_confidence", "eye_focus_score"):
            if field in crop and not is_unit_interval(crop[field]):
                missing.append(f"crop_{index}_invalid_{field}")
        missing.extend(advisory_metadata_issues(crop, index))
    baseline_primary = max(range(len(crops)), key=lambda index: numeric_score(crops[index].get("quality"), -1.0))
    prototype_primary = max(range(len(crops)), key=lambda index: numeric_score(crops[index].get("combined_score", crops[index].get("quality")), -1.0))
    baseline, prototype = crops[baseline_primary], crops[prototype_primary]
    return CaseResult(scene, baseline_primary, prototype_primary, numeric_score(baseline.get("quality"), -1.0), numeric_score(prototype.get("quality"), -1.0), numeric_score(prototype.get("combined_score", prototype.get("quality")), -1.0), baseline_primary != prototype_primary, tuple(sorted(set(missing))), tuple(crops))


def advisory_metadata_issues(crop: dict[str, Any], index: int) -> list[str]:
    prefix = f"crop_{index}_"
    status = crop.get("analysis_status")
    issues: list[str] = []
    status_is_valid = isinstance(status, str) and status in ADVISORY_STATUSES
    if not status_is_valid:
        issues.append(prefix + ("missing_analysis_status" if status is None else "invalid_analysis_status"))
    provenance = crop.get("provenance")
    if not isinstance(provenance, dict):
        issues.append(prefix + "missing_provenance")
    else:
        if provenance.get("advisory") is not True:
            issues.append(prefix + "invalid_provenance_advisory")
        if provenance.get("detector") != ADVISORY_DETECTOR:
            issues.append(prefix + "invalid_provenance_detector")
        if provenance.get("version") != ADVISORY_VERSION:
            issues.append(prefix + "invalid_provenance_version")
        if provenance.get("status") != status:
            issues.append(prefix + "invalid_provenance_status")
        reason = provenance.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            issues.append(prefix + "invalid_provenance_reason")
    eye_detected = crop.get("eye_detected")
    if not isinstance(eye_detected, bool):
        issues.append(prefix + "invalid_eye_detected")
    elif status == "detected" and not eye_detected:
        issues.append(prefix + "detected_status_mismatch")
    elif status_is_valid and status != "detected" and eye_detected:
        issues.append(prefix + "non_detected_status_mismatch")
    return issues


def print_summary(results: list[CaseResult]) -> None:
    changed = sum(result.changed for result in results)
    missing = sum(bool(result.missing_reasons) for result in results)
    print("Eye-focus advisory diagnostic")
    print(f"cases={len(results)} changed={changed} unchanged={len(results) - changed} missing_metadata={missing}")
    print()


def should_print(result: CaseResult, show: str) -> bool:
    return show == "all" or (show == "missing" and bool(result.missing_reasons)) or (show == "changed" and result.changed)


def print_cases(results: list[CaseResult], *, show: str, limit: int) -> None:
    selected = [result for result in results if should_print(result, show)]
    if not selected:
        print(f"No cases matched --show={show}.")
        return
    for result in selected[:limit]:
        print_case(result)
    remaining = len(selected) - min(limit, len(selected))
    if remaining:
        print(f"... {remaining} more cases not shown (increase --limit to inspect more).")


def print_case(result: CaseResult) -> None:
    print(f"scene={escape_terminal_text(result.scene)}")
    print(f"  baseline_primary={result.baseline_primary} quality={result.baseline_quality:.3f}")
    print(f"  recorded_primary={result.prototype_primary} quality={result.prototype_quality:.3f} combined_score={result.prototype_combined:.3f}")
    print(f"  changed={'yes' if result.changed else 'no'}")
    print(f"  metadata_status={'missing (' + '; '.join(result.missing_reasons) + ')' if result.missing_reasons else 'ok'}")
    for index, crop in enumerate(result.crops):
        flags = []
        if index == result.baseline_primary:
            flags.append("baseline")
        if index == result.prototype_primary:
            flags.append("recorded")
        print(f"    crop={index}{' [' + ' '.join(flags) + ']' if flags else ''} quality={numeric_score(crop.get('quality'), -1.0):.3f} eye_focus_score={numeric_score(crop.get('eye_focus_score'), -1.0):.3f} combined_score={numeric_score(crop.get('combined_score', crop.get('quality')), -1.0):.3f} analysis_status={crop.get('analysis_status')!r} eye_detected={crop.get('eye_detected')!r} eye_confidence={numeric_score(crop.get('eye_confidence'), -1.0):.3f}")
    print()


def escape_terminal_text(value: str) -> str:
    return "".join(character if character.isprintable() else repr(character)[1:-1] for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
