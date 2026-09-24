#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
IMAGE=${IMAGE:-qwen-image-2.1-rocm:w7900d-fast}
MODELS="$ROOT/models"
mkdir -p "$MODELS"
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "local image not found: $IMAGE; run scripts/build.sh first" >&2
  exit 1
}
args=(run --rm --pull=never -i --entrypoint python -e QWEN_MODEL_ROOT=/models -e QWEN_MODEL_MANIFEST=/opt/qwen/models/manifest.json -v "$MODELS:/models:rw" "$IMAGE" /opt/qwen/scripts/download_models.py)
if [[ ${1:-} == --verify-only ]]; then
  args+=(--verify-only)
elif [[ $# -ne 0 ]]; then
  echo "usage: scripts/download-models.sh [--verify-only]" >&2
  exit 2
fi
docker "${args[@]}"
