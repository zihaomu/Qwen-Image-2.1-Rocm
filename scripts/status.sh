#!/usr/bin/env bash
set -euo pipefail
CONTAINER=${CONTAINER:-qwen-image-2.1-rocm-fast}

docker exec -i "$CONTAINER" python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=10) as response:
    system = json.load(response)
with urllib.request.urlopen("http://127.0.0.1:8188/qwen_image_rocm_fast/status", timeout=10) as response:
    fast = json.load(response)
assert len(system["devices"]) == 1, system["devices"]
assert system["devices"][0]["index"] == 0, system["devices"]
assert fast["configured"] is True
assert fast["mode"] == "candidate"
assert fast["capabilities"]["available"] is True
assert fast["extension"]["verified"] is True
assert fast["extension"]["sage_sdpa"] is True
assert fast["extension"]["rms_rope"] is True
print(json.dumps({"device": system["devices"][0], "fast": fast}, indent=2, sort_keys=True))
print("RUNTIME_STATUS=PASS")
PY
