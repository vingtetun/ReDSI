#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FLAGS=(--upgrade --editable)
if [[ "${1:-}" == "--force" || "${1:-}" == "-f" ]]; then
  FLAGS+=(--force-reinstall)
fi

exec "${PYTHON_BIN}" -m pip install "${FLAGS[@]}" "${ROOT_DIR}"
