#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e}
OUT=${OUT:-$ROOT/dist/comfy_kitchen_gfx1100_source.so}
mkdir -p "$(dirname -- "$OUT")"

docker run --rm \
  -v "$ROOT:/project:ro" \
  -v "$(dirname -- "$OUT"):/out:rw" \
  -w /tmp --entrypoint sh "$BASE" -lc '
    set -eu
    python -m pip install --no-cache-dir \
      /project/vendor/cmake-4.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
      /project/vendor/nanobind-3.1.0-py3-none-any.whl
    mkdir /tmp/comfy-kitchen
    tar -xzf /project/vendor/comfy-kitchen-v0.2.35-b2a2972.tar.gz --strip-components=1 -C /tmp/comfy-kitchen
    install -m 0644 /project/kernels/int8_attn.hip /tmp/comfy-kitchen/comfy_kitchen/backends/hip/sage_attention/int8_attn.hip
    install -m 0644 /project/kernels/rms_rope.hip /tmp/comfy-kitchen/comfy_kitchen/backends/hip/ops/rms_rope.hip
    cd /tmp/comfy-kitchen
    COMFY_KITCHEN_BUILD_HIP=1 COMFY_KITCHEN_BUILD_NO_CUDA=1 COMFY_HIP_ARCHS=gfx1100 \
      CMAKE_GENERATOR=Ninja python setup.py build_ext --inplace --hip-archs=gfx1100
    install -m 0755 comfy_kitchen/backends/hip/_C.abi3.so /out/comfy_kitchen_gfx1100_source.so
  '
sha256sum "$OUT"
echo "Source rebuilds may not be byte-identical because linker/build metadata is not reproducible."
