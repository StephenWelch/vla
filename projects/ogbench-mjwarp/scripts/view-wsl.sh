#!/usr/bin/env bash
set -euo pipefail
data_root="${OGBENCH_DATA_ROOT:-/mnt/e/vla-ogbench}"
environment="${OGBENCH_ENVIRONMENT:-$data_root/venv}"
if [[ ! -x "$environment/bin/ogbench-mjwarp" ]]; then
    echo "Missing WSL environment at $environment; run bash scripts/setup-wsl.sh first." >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is unset. Run from a WSLg-enabled WSL 2 distribution." >&2
    exit 1
fi
# EGL is for offscreen generation; the native viewer needs GLFW/X11 through WSLg.
export MUJOCO_GL=glfw
# Mesa may otherwise choose llvmpipe despite an available WSL D3D12 GPU.
if [[ -e /usr/lib/wsl/lib/libd3d12.so ]]; then
    export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
    export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
fi
exec "$environment/bin/ogbench-mjwarp" view "$@"
