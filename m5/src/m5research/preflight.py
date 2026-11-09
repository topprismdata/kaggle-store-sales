from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

REQUIRED_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "scipy",
    "lightgbm",
    "yaml",
    "pytest",
)


def _version(distribution: str) -> str | None:
    package = "PyYAML" if distribution == "yaml" else distribution
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_preflight(project_root: Path | None = None, data_dir: Path | None = None) -> dict[str, Any]:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    py_ok = sys.version_info[:2] == (3, 11)
    add("python_version", py_ok, f"expected 3.11.x; actual {platform.python_version()}")

    utf8_env = os.environ.get("PYTHONUTF8")
    stream_encoding = getattr(sys.stdout, "encoding", None) or "unknown"
    utf8_ok = (
        utf8_env == "1"
        or sys.flags.utf8_mode == 1
        or stream_encoding.lower().replace("-", "") == "utf8"
    )
    if os.name == "nt":
        utf8_ok = utf8_env == "1" and utf8_ok
    add(
        "utf8_mode",
        utf8_ok,
        f"PYTHONUTF8={utf8_env!r}; utf8_mode={sys.flags.utf8_mode}; stdout={stream_encoding}",
    )

    ascii_ok = str(root).isascii()
    add("ascii_project_path", ascii_ok, str(root))

    package_versions = {name: _version(name) for name in REQUIRED_DISTRIBUTIONS}
    missing = [name for name, version in package_versions.items() if version is None]
    add("required_packages", not missing, json.dumps(package_versions, ensure_ascii=False))

    config_path = root / "configs" / "accuracy_v1.yaml"
    add("accuracy_config", config_path.is_file(), str(config_path))

    selected_data = data_dir or (
        Path(os.environ["M5_DATA_DIR"]) if os.environ.get("M5_DATA_DIR") else None
    )
    if selected_data is None:
        add("data_directory", False, "set M5_DATA_DIR or pass --data-dir; data not checked")
    else:
        exists = selected_data.is_dir()
        add("data_directory", exists, str(selected_data))

    report = {
        "status": "PASS" if all(item["ok"] for item in checks) else "BLOCKED",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(root),
        "cpu_logical_count": os.cpu_count(),
        "checks": checks,
    }
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check M5 development environment.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report = run_preflight(data_dir=args.data_dir)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
