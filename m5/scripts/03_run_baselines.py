from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 Accuracy baseline runner (implementation gate).")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", choices=["seasonal_naive_7", "seasonal_mean_28"], required=True)
    parser.parse_args()
    raise SystemExit(
        "BASELINE_RUNNER_NOT_IMPLEMENTED: implement only after G0-G2 data/hierarchy checks pass "
        "and G3 scorer reference is available. No model result has been generated."
    )


if __name__ == "__main__":
    raise SystemExit(main())
