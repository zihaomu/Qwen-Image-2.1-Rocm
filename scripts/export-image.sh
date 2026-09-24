#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
IMAGE=${IMAGE:-qwen-image-2.1-rocm:w7900d-fast}
OUTPUT=${OUTPUT:-$ROOT/dist/qwen-image-2.1-rocm-w7900d-fast.tar}
mkdir -p "$(dirname -- "$OUTPUT")"
docker image inspect "$IMAGE" >/dev/null
docker save --output "$OUTPUT" "$IMAGE"
output_dir=$(cd -- "$(dirname -- "$OUTPUT")" && pwd -P)
output_name=$(basename -- "$OUTPUT")
( cd "$output_dir" && sha256sum "$output_name" ) | tee "$OUTPUT.sha256"
