"""Validate official M5 reference weight CSVs downloaded by authorized users.

This is NOT the full WRMSSE scorer. Standard-library only; works on Windows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

EXPECTED = dict(zip((f"Level{i}" for i in range(1, 13)),
                    (1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(path: Path, compact: bool = False) -> tuple[dict, dict]:
    expected_cols = ("Level_id", "Agg_Level_1", "Agg_Level_2")
    expected_cols += ("Weight",) if compact else ("Dollar_Sales", "weight")
    by_key = {}
    counts = Counter()
    sums = {level: 0.0 for level in EXPECTED}
    dollars = {level: 0.0 for level in EXPECTED}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_cols:
            raise ValueError(f"{path}: bad columns: {reader.fieldnames}")
        for row in reader:
            key = tuple(row[k] for k in ("Level_id", "Agg_Level_1", "Agg_Level_2"))
            level = key[0]
            if level not in EXPECTED or key in by_key:
                raise ValueError(f"unknown level or duplicate reference key: {key}")
            weight = float(row["Weight" if compact else "weight"])
            dollar = None if compact else float(row["Dollar_Sales"])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{key}: invalid weight {weight}")
            if dollar is not None and (not math.isfinite(dollar) or dollar < 0):
                raise ValueError(f"{key}: invalid dollar sales {dollar}")
            by_key[key] = (weight, dollar)
            counts[level] += 1
            sums[level] += weight
            if dollar is not None:
                dollars[level] += dollar

    if any(counts.get(level) != number for level, number in EXPECTED.items()):
        raise ValueError(f"bad level counts: {dict(counts)}")
    if len(by_key) != 42840:
        raise ValueError(f"expected 42,840 keys; got {len(by_key)}")

    if not compact:
        total = dollars["Level1"]
        if total <= 0:
            raise ValueError("total dollar sales must be positive")
        for level in EXPECTED:
            if abs(sums[level] - 1.0) > 1e-7:
                raise ValueError(f"{level}: weights do not sum to one: {sums[level]}")
            if abs(dollars[level] - total) > 0.05:
                raise ValueError(f"{level}: dollar sales do not match total")
        for key, (weight, dollar) in by_key.items():
            if abs(weight - dollar / total) > 1e-9:
                raise ValueError(f"{key}: weight differs from dollar ratio")

    info = {
        "file": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": len(by_key),
        "level_counts": dict(counts),
        "weight_sums": sums,
        "dollar_sums": None if compact else dollars,
        "format": "compact" if compact else "primary",
    }
    return by_key, info


def compare_validation(primary: dict, compact: dict) -> dict:
    if set(primary) != set(compact):
        raise ValueError("validation files contain different aggregation keys")
    diffs = [abs(primary[key][0] - compact[key][0]) for key in primary]
    largest = max(diffs)
    if largest > 1e-6:
        raise ValueError(f"compact validation weights too different: {largest}")
    return {
        "matched_keys": len(primary),
        "max_abs_weight_difference": largest,
        "mean_abs_weight_difference": sum(diffs) / len(diffs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--compact-validation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        val, val_summary = load_reference(args.validation)
        eva, eva_summary = load_reference(args.evaluation)
        if set(val) != set(eva):
            raise ValueError("validation/evaluation key sets differ")
        result = {
            "status": "REFERENCE_FILES_VALIDATED",
            "not_a_full_wrmsse_scorer": True,
            "validation": val_summary,
            "evaluation": eva_summary,
        }
        if args.compact_validation:
            comp, comp_summary = load_reference(args.compact_validation, compact=True)
            result["compact_validation"] = comp_summary
            result["cross_check"] = compare_validation(val, comp)
    except (OSError, ValueError) as exc:
        result = {"status": "BLOCKED", "error": str(exc)}

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["status"] == "REFERENCE_FILES_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
