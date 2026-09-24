#!/usr/bin/env bash
set -euo pipefail
CONTAINER=${CONTAINER:-qwen-image-2.1-rocm-fast}
if ! docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
  echo "container not found: $CONTAINER"
  exit 0
fi
docker rm --force "$CONTAINER"
