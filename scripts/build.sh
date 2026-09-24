#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
IMAGE=${IMAGE:-qwen-image-2.1-rocm:w7900d-fast}
EXPECTED=0328285d26d04786c0e72a463db5952bf8c9f9aab4544d88d71ce595fc45560d

[[ "$(sha256sum "$ROOT/artifacts/comfy_kitchen_gfx1100.so" | awk '{print $1}')" == "$EXPECTED" ]] || {
  echo "validated gfx1100 extension checksum mismatch" >&2
  exit 1
}
docker build --pull=false --tag "$IMAGE" "$ROOT"
docker image inspect "$IMAGE" --format 'IMAGE_ID={{.Id}}'
