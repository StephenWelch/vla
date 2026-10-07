#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${OCBENCH_ENVIRONMENT:-$HOME/.venvs/vla-ocbench}"
if ! command -v ffmpeg >/dev/null 2>&1; then
  printf 'Install FFmpeg shared libraries first: sudo apt-get install ffmpeg\n' >&2
  exit 1
fi
uv sync --project "$root/projects/ocbench-mjwarp" --locked --extra gpu-video
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'from torchcodec.decoders import VideoDecoder; print("TorchCodec decoder available")'
printf 'Activate: source %s/bin/activate\n' "$UV_PROJECT_ENVIRONMENT"
