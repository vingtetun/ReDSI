#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-.}"

if [ "$TARGET" = "." ]; then
  echo "▶ pre-commit: all files"
  pre-commit run --all-files
else
  echo "▶ pre-commit: $TARGET"
  pre-commit run --files "$TARGET"
fi
