#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
data_root="${LIBERO_DATA_ROOT:-$HOME/vla-libero}"
export UV_PROJECT_ENVIRONMENT="${LIBERO_ENVIRONMENT:-$data_root/venv}"
mkdir -p "$data_root"
uv sync --project "$project_root" --frozen --extra dev --python 3.12
echo "LIBERO environment: $UV_PROJECT_ENVIRONMENT"
