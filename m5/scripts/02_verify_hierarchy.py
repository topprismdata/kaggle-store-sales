from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5research.hierarchy import EXPECTED_LEVEL_COUNTS, hierarchy_counts, validate_official_hierarchy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify official M5 12-level hierarchy counts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    path = args.data_dir / "sales_train_validation.csv"
    if not path.is_file():
        parser.error(f"missing file: {path}")

    frame = pd.read_csv(path, usecols=["item_id", "dept_id", "cat_id", "store_id", "state_id"])
    counts = hierarchy_counts(frame)
    total = sum(counts.values())
    errors = []
    try:
        validate_official_hierarchy(counts)
    except ValueError as exc:
        errors.append(str(exc))

    report = {
        "status": "PASS" if not errors else "BLOCKED",
        "bottom_rows": int(len(frame)),
        "levels": counts,
        "total_series": total,
        "expected_levels": EXPECTED_LEVEL_COUNTS,
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
