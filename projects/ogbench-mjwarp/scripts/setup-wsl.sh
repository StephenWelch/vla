#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
data_root="${OGBENCH_DATA_ROOT:-/mnt/e/vla-ogbench}"
export UV_PROJECT_ENVIRONMENT="${OGBENCH_ENVIRONMENT:-$data_root/venv}"
mkdir -p "$data_root/datasets"
uv sync --project "$project_root" --frozen --extra dataset --extra evaluation --extra dev --python 3.12
MUJOCO_GL=egl "$UV_PROJECT_ENVIRONMENT/bin/ogbench-mjwarp" doctor
