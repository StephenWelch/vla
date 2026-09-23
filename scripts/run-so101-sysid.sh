#!/usr/bin/env bash
# Run the SO-101 sysid fit in a Linux Docker container (mjbatch ships Linux wheels only).
# Usage (from the repo root or anywhere):
#   scripts/run-so101-sysid.sh --data outputs/so101_sysid/<collection> [runner args...]
#   scripts/run-so101-sysid.sh --synthetic
# The repo is bind-mounted, so everything under outputs/ written by the
# container is visible on the Windows side.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-so101-sysid:latest}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building $IMAGE from docker/sysid.Dockerfile..."
  docker build -f docker/sysid.Dockerfile -t "$IMAGE" .
fi

docker run --rm -i \
  -v "$(pwd):/work" \
  -w /work \
  "$IMAGE" \
  python scripts/run-so101-sysid.py "$@"
