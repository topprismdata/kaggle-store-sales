"""Pure synthetic test vectors: never store or redistribute official weights."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "05_validate_reference_weights.py"
SPEC = importlib.util.spec_from_file_location("m5_reference_validator", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_synthetic_weights(path: Path, *, compact: bool = False) -> None:
    cols = ["Level_id", "Agg_Level_1", "Agg_Level_2"]
    cols += ["Weight"] if compact else ["Dollar_Sales", "weight"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for level, count in MODULE.EXPECTED.items():
            for index in range(count):
                row = {
                    "Level_id": level,
                    "Agg_Level_1": f"item_{index}",
                    "Agg_Level_2": "X",
                }
                if compact:
                    row["Weight"] = f"{1.0 / count:.9g}"
                else:
                    row["Dollar_Sales"] = 100.0 / count
                    row["weight"] = 1.0 / count
                writer.writerow(row)


def test_official_reference_structure_with_synthetic_fixture(tmp_path):
    primary = tmp_path / "weights_validation.csv"
    evaluation = tmp_path / "weights_evaluation.csv"
    compact = tmp_path / "compact_validation.csv"
    write_synthetic_weights(primary)
    write_synthetic_weights(evaluation)
    write_synthetic_weights(compact, compact=True)

    v, v_info = MODULE.load_reference(primary)
    e, _ = MODULE.load_reference(evaluation)
    c, _ = MODULE.load_reference(compact, compact=True)

    assert len(v) == len(e) == len(c) == 42840
    assert v_info["row_count"] == 42840
    assert MODULE.compare_validation(v, c)["matched_keys"] == 42840


def test_invalid_reference_columns_are_rejected(tmp_path):
    wrong = tmp_path / "weights_validation.csv"
    wrong.write_text("Level_id,weight\nLevel1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad columns"):
        MODULE.load_reference(wrong)
