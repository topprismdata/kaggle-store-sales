from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED = {
    "sales_train_validation.csv": (30490, 1913),
    "sales_train_evaluation.csv": (30490, 1941),
}
ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def inspect_csv(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return header, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit M5 Accuracy file presence and dimensions.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.data_dir.resolve()

    required = [
        "sales_train_validation.csv",
        "sales_train_evaluation.csv",
        "calendar.csv",
        "sell_prices.csv",
        "sample_submission.csv",
    ]
    report = {"data_dir": str(root), "files": {}, "errors": []}
    for name in required:
        path = root / name
        if not path.is_file():
            report["errors"].append(f"missing file: {name}")
            continue
        header, rows = inspect_csv(path)
        item = {"rows": rows, "columns": len(header), "header": header[:8]}
        if name in EXPECTED:
            expected_rows, expected_days = EXPECTED[name]
            expected_cols = len(ID_COLUMNS) + expected_days
            item["expected_rows"] = expected_rows
            item["expected_columns"] = expected_cols
            if rows != expected_rows:
                report["errors"].append(f"{name}: expected {expected_rows} rows, got {rows}")
            if len(header) != expected_cols:
                report["errors"].append(f"{name}: expected {expected_cols} columns, got {len(header)}")
            if header[:6] != ID_COLUMNS:
                report["errors"].append(f"{name}: unexpected identifier column order")
            if header[6] != "d_1" or header[-1] != f"d_{expected_days}":
                report["errors"].append(f"{name}: unexpected day-column boundaries")
        report["files"][name] = item

    report["status"] = "PASS" if not report["errors"] else "BLOCKED"
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
