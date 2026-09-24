#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
IMAGE=${IMAGE:-qwen-image-2.1-rocm:w7900d-fast}
CONTAINER=${CONTAINER:-qwen-image-2.1-rocm-fast}
HOST_PORT=${HOST_PORT:-28188}
GPU_INDEX=${GPU_INDEX:-0}
DRY_RUN=0

usage() { echo "usage: scripts/start.sh [--dry-run]"; }
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi
if [[ $# -eq 1 ]]; then
  case "$1" in --dry-run) DRY_RUN=1 ;; -h|--help) usage; exit 0 ;; *) usage >&2; exit 2 ;; esac
fi
[[ "$CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "invalid container name" >&2; exit 1; }
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || { echo "GPU_INDEX must be numeric" >&2; exit 1; }
[[ "$HOST_PORT" =~ ^[0-9]+$ ]] && (( HOST_PORT >= 1 && HOST_PORT <= 65535 )) || { echo "HOST_PORT must be 1..65535" >&2; exit 1; }
if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then echo "container already exists: $CONTAINER" >&2; exit 1; fi
[[ -z "$(ss -H -ltn "sport = :$HOST_PORT")" ]] || { echo "port is busy: $HOST_PORT" >&2; exit 1; }
IMAGE="$IMAGE" "$ROOT/scripts/download-models.sh" --verify-only

gpu_report=$(rocm-smi -d "$GPU_INDEX" --showmeminfo vram --showuse 2>&1) || { printf '%s\n' "$gpu_report" >&2; exit 1; }
used=$(printf '%s\n' "$gpu_report" | awk '/VRAM Total Used Memory \(B\):/ {print $NF; exit}')
util=$(printf '%s\n' "$gpu_report" | awk '/GPU use \(%\):/ {print $NF; exit}')
[[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || { echo "cannot parse GPU state" >&2; exit 1; }
awk -v used="$used" -v util="$util" 'BEGIN { exit !(used <= 268435456 && util <= 5) }' || {
  echo "GPU $GPU_INDEX is not idle: used_bytes=$used use_percent=$util" >&2
  exit 1
}

runtime="$ROOT/runtime/$CONTAINER"
mkdir -p "$runtime"/{user,logs} "$ROOT/output"
[[ -e /dev/kfd && -d /dev/dri ]] || { echo "ROCm devices /dev/kfd and /dev/dri are required" >&2; exit 1; }
kfd_gid=$(stat -c %g /dev/kfd)
render_node=$(find /dev/dri -maxdepth 1 -type c -name 'renderD*' -print -quit)
[[ -n "$render_node" ]] || { echo "no render node found under /dev/dri" >&2; exit 1; }
render_gid=$(stat -c %g "$render_node")
args=(run -d --name "$CONTAINER" --device /dev/kfd --device /dev/dri --group-add "$kfd_gid" --group-add "$render_gid" --ipc=host --shm-size=32g --security-opt seccomp=unconfined --log-opt max-size=50m --log-opt max-file=3 -e "ROCR_VISIBLE_DEVICES=$GPU_INDEX" -e HIP_VISIBLE_DEVICES=0 -p "127.0.0.1:$HOST_PORT:8188" -v "$ROOT/models:/opt/ComfyUI/models:ro" -v "$ROOT/output:/opt/ComfyUI/output:rw" -v "$runtime/user:/opt/ComfyUI/user:rw" -v "$runtime/logs:/var/log/comfyui:rw" "$IMAGE")
if (( DRY_RUN )); then printf 'DRY_RUN='; printf '%q ' docker "${args[@]}"; printf '\n'; exit 0; fi
id=$(docker "${args[@]}")
ready=0
for _ in $(seq 1 120); do
  if docker exec "$CONTAINER" python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2).read()' >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! docker inspect "$CONTAINER" --format '{{.State.Running}}' | grep -Fxq true; then
    break
  fi
  sleep 1
done
if (( ! ready )); then
  docker logs --tail 100 "$CONTAINER" >&2 || true
  docker rm --force "$CONTAINER" >/dev/null 2>&1 || true
  echo "ComfyUI did not become healthy; failed container removed" >&2
  exit 1
fi
printf 'CONTAINER_ID=%s\nURL=http://127.0.0.1:%s\nGPU_PHYSICAL=%s\nGPU_LOGICAL=0\n' "$id" "$HOST_PORT" "$GPU_INDEX"
