from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a portable SHA-256 manifest.")
    parser.add_argument("--input", type=Path, required=True, help="File or directory to fingerprint.")
    parser.add_argument("--output", type=Path, required=True, help="JSON manifest output path.")
    args = parser.parse_args()

    source = args.input.resolve()
    if not source.exists():
        parser.error(f"input does not exist: {source}")

    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    entries = []
    for path in files:
        rel = path.name if source.is_file() else str(path.relative_to(source))
        entries.append({
            "path": rel.replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "file_count": len(entries),
        "files": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "file_count": len(entries), "manifest": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
