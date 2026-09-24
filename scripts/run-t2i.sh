#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
CONTAINER=${CONTAINER:-qwen-image-2.1-rocm-fast}
SEED=${SEED:-44}
RUN_NAME=${RUN_NAME:-t2i-seed${SEED}}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "invalid RUN_NAME" >&2; exit 2; }

CONTAINER="$CONTAINER" "$ROOT/scripts/status.sh" >/dev/null
result_dir=/opt/ComfyUI/user/results
docker exec "$CONTAINER" mkdir -p "$result_dir"
docker exec "$CONTAINER" python /opt/qwen/scripts/benchmark.py \
  --url http://127.0.0.1:8188 \
  --graph-file /opt/qwen/workflows/qwen_image_2_1_t2i_api.json \
  --seed "$SEED" \
  --prefix "qwen_image_2_1_rocm_fast/$RUN_NAME" \
  --output-json "$result_dir/$RUN_NAME.json" \
  --timeout 1200
runtime_user=$(docker inspect "$CONTAINER" --format '{{range .Mounts}}{{if eq .Destination "/opt/ComfyUI/user"}}{{.Source}}{{end}}{{end}}')
[[ -n "$runtime_user" ]] || { echo "cannot locate mounted user directory" >&2; exit 1; }
printf 'RESULT=%s/results/%s.json\nOUTPUT_DIR=%s/output/qwen_image_2_1_rocm_fast\n' "$runtime_user" "$RUN_NAME" "$ROOT"
