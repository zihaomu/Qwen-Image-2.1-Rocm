#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
CONTAINER=${CONTAINER:-qwen-image-2.1-rocm-fast}

docker inspect "$CONTAINER" >/dev/null 2>&1 || { echo "container not found: $CONTAINER" >&2; exit 1; }
docker restart "$CONTAINER" >/dev/null
ready=0
for _ in $(seq 1 120); do
  if CONTAINER="$CONTAINER" "$ROOT/scripts/status.sh" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! docker inspect "$CONTAINER" --format '{{.State.Running}}' | grep -Fxq true; then
    break
  fi
  sleep 1
done
(( ready )) || { docker logs --tail 100 "$CONTAINER" >&2; echo "ComfyUI did not become healthy after restart" >&2; exit 1; }

CONTAINER="$CONTAINER" "$ROOT/scripts/status.sh" >/dev/null
result_dir=/opt/ComfyUI/user/results
docker exec "$CONTAINER" mkdir -p "$result_dir"
docker exec -i "$CONTAINER" python - <<'PY'
import json
import urllib.request
from pathlib import Path

with urllib.request.urlopen("http://127.0.0.1:8188/qwen_image_rocm_fast/status", timeout=10) as response:
    status = json.load(response)
Path("/opt/ComfyUI/user/results/backend-before.json").write_text(
    json.dumps(status.get("stats", {}), indent=2, sort_keys=True) + "\n"
)
PY

docker exec "$CONTAINER" python /opt/qwen/scripts/benchmark.py \
  --url http://127.0.0.1:8188 \
  --graph-file /opt/qwen/workflows/qwen_image_2_1_t2i_api.json \
  --seed 43 \
  --prefix qwen_image_2_1_rocm_fast/warm-seed43 \
  --output-json "$result_dir/warm-seed43.json" \
  --timeout 1200 \
  --warmup

docker exec "$CONTAINER" python /opt/qwen/scripts/benchmark.py \
  --url http://127.0.0.1:8188 \
  --graph-file /opt/qwen/workflows/qwen_image_2_1_t2i_api.json \
  --seed 44 \
  --prefix qwen_image_2_1_rocm_fast/measure-seed44 \
  --output-json "$result_dir/measure-seed44.json" \
  --timeout 1200

docker exec -i "$CONTAINER" python - <<'PY'
import json
import urllib.request
from pathlib import Path

result = json.loads(Path("/opt/ComfyUI/user/results/measure-seed44.json").read_text())
assert result["error"] is None
assert result["seed"] == 44
expected_cached = {"13", "459:451", "459:452", "459:453", "459:454", "459:456"}
assert set(result["cached_nodes"]) == expected_cached, result["cached_nodes"]
assert result["metadata"]["warmup"] is False
assert result["metadata"]["prompt_suffix_length"] == 0
ksampler = next(item for item in result["node_intervals"] if item["class_type"] == "KSampler")
with urllib.request.urlopen("http://127.0.0.1:8188/qwen_image_rocm_fast/status", timeout=10) as response:
    status = json.load(response)
assert status["mode"] == "candidate"
assert status["extension"]["verified"] is True
backend_before = json.loads(Path("/opt/ComfyUI/user/results/backend-before.json").read_text())
backend_after = status["stats"]
candidate_calls = backend_after.get("candidate:selected", 0) - backend_before.get("candidate:selected", 0)
fallback_calls = sum(
    backend_after.get(key, 0) - backend_before.get(key, 0)
    for key in set(backend_after) | set(backend_before)
    if key != "candidate:selected"
)
assert candidate_calls == 1664, candidate_calls
assert fallback_calls == 0, fallback_calls
summary = {
    "wall_ms": result["wall_ms"],
    "ksampler_ms": ksampler["duration_ms"],
    "candidate_calls": candidate_calls,
    "fallback_calls": fallback_calls,
    "outputs": result["outputs"],
}
Path("/opt/ComfyUI/user/results/latest-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
print("BENCHMARK_VALIDATED=PASS")
PY
runtime_user=$(docker inspect "$CONTAINER" --format '{{range .Mounts}}{{if eq .Destination "/opt/ComfyUI/user"}}{{.Source}}{{end}}{{end}}')
printf 'SUMMARY=%s/results/latest-summary.json\n' "$runtime_user"
