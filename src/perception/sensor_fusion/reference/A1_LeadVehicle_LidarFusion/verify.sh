#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$ROOT/tools/verify_copies.py"
cmake -S "$ROOT/core" -B "$ROOT/.build/core" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/.build/core" -j 2
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider "$ROOT/reference/tests" "$ROOT/tests"
