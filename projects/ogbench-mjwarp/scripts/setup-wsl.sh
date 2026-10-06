#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
data_root="${OGBENCH_DATA_ROOT:-/mnt/e/vla-ogbench}"
export UV_PROJECT_ENVIRONMENT="${OGBENCH_ENVIRONMENT:-$data_root/venv}"
mkdir -p "$data_root/datasets"
extras=(--extra dataset --extra evaluation --extra dev)
if [[ "${OGBENCH_CUROBO:-0}" == "1" ]]; then
  extras+=(--extra curobo)
fi
uv sync --project "$project_root" --frozen "${extras[@]}" --python 3.12
MUJOCO_GL=egl "$UV_PROJECT_ENVIRONMENT/bin/ogbench-mjwarp" doctor
