#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# Fundamental checks. A change is finished only when this passes.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. clean build, warnings as errors"
SKBUILD_CMAKE_DEFINE="CMAKE_COMPILE_WARNING_AS_ERROR=ON" \
  uv sync --locked --reinstall-package tickdes

echo "== 2. format and lint"
uv run --no-sync ruff check
uv run --no-sync ruff format --check
cpp_files=$(find . \( -name .git -o -name .venv -o -name build \) -prune -o -type f \
  \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) -print)
if [ -n "$cpp_files" ]; then
  # shellcheck disable=SC2086
  uv run --no-sync clang-format --dry-run -Werror $cpp_files
fi

echo "== 4. tests"
uv run --no-sync pytest -q

echo "All checks passed."
