#!/usr/bin/env python3
"""Verify the self-contained senior source/checkpoint copy without PyTorch."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PROJECT_ROOT / "senior_code"
MANIFEST = BUNDLE_ROOT / "COPY_MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print(f"[FAIL] missing manifest: {MANIFEST}")
        return 1

    failures = []
    checked = 0
    for line_number, raw in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split(None, 1)
        except ValueError:
            failures.append(f"manifest line {line_number}: invalid format")
            continue
        path = BUNDLE_ROOT / relative.strip()
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"hash mismatch: {relative}\n  expected {expected}\n  actual   {actual}")
        checked += 1

    if failures:
        print("[FAIL] portable bundle verification")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"[OK] verified {checked} bundled senior files")
    print(f"[OK] bundle root: {BUNDLE_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
