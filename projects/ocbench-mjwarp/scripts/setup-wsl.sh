#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${OCBENCH_ENVIRONMENT:-$HOME/.venvs/vla-ocbench}"
uv sync --project "$root/projects/ocbench-mjwarp" --locked
printf 'Activate: source %s/bin/activate\n' "$UV_PROJECT_ENVIRONMENT"
