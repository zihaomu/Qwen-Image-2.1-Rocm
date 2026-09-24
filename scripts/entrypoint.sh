#!/usr/bin/env bash
set -euo pipefail

mkdir -p /opt/ComfyUI/output /opt/ComfyUI/user/default/workflows /var/log/comfyui
workflow=/opt/ComfyUI/user/default/workflows/qwen_image_2_1_t2i_fast.json
if [[ ! -e "$workflow" ]]; then
  install -m 0644 /opt/qwen/workflows/qwen_image_2_1_t2i.json "$workflow"
fi

exec "$@"
