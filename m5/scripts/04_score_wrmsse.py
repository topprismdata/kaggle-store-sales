from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 official WRMSSE scorer entry point.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.parse_args()
    raise SystemExit(
        "SCORER_NOT_VERIFIED: obtain official weights/scales and reference outputs, "
        "then cross-check all 12 levels before reporting official WRMSSE."
    )


if __name__ == "__main__":
    raise SystemExit(main())
