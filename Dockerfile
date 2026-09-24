FROM rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e

ARG COMFYUI_COMMIT=d158420950922dde49dad1fa87fa549735fd5137
ARG COMFY_KITCHEN_COMMIT=b2a2972ac68c395bbda8ad9030e8ae1089287815

LABEL org.opencontainers.image.title="Qwen-Image-2.1 ComfyUI ROCm fast gfx1100" \
      org.opencontainers.image.description="Fastest validated Qwen-Image-2.1 T2I path for AMD Radeon PRO W7900D" \
      org.opencontainers.image.base.digest="sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e" \
      org.opencontainers.image.revision="${COMFYUI_COMMIT}" \
      io.qwen.rocm.comfy-kitchen.revision="${COMFY_KITCHEN_COMMIT}" \
      io.qwen.rocm.extension.sha256="0328285d26d04786c0e72a463db5952bf8c9f9aab4544d88d71ce595fc45560d" \
      io.qwen.rocm.gpu.arch="gfx1100" \
      io.qwen.rocm.backend="candidate"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COMFYUI_PATH=/opt/ComfyUI

WORKDIR /opt/ComfyUI

COPY vendor/ComfyUI-d158420.tar.gz /tmp/ComfyUI.tar.gz
RUN tar -xzf /tmp/ComfyUI.tar.gz --strip-components=1 -C /opt/ComfyUI \
    && rm /tmp/ComfyUI.tar.gz

COPY docker/requirements.txt docker/constraints-rocm.txt docker/constraints-runtime.txt /tmp/qwen-build/
RUN python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
      --constraint /tmp/qwen-build/constraints-rocm.txt \
      --constraint /tmp/qwen-build/constraints-runtime.txt \
      --requirement /tmp/qwen-build/requirements.txt \
    && python -m pip check

COPY artifacts/comfy_kitchen_gfx1100.so /tmp/comfy_kitchen_gfx1100.so
RUN target="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/comfy_kitchen/backends/hip/_C.abi3.so")')" \
    && install -m 0755 /tmp/comfy_kitchen_gfx1100.so "$target" \
    && echo "0328285d26d04786c0e72a463db5952bf8c9f9aab4544d88d71ce595fc45560d  $target" | sha256sum -c - \
    && ! ldd "$target" | grep -q 'not found' \
    && rm /tmp/comfy_kitchen_gfx1100.so

COPY custom_nodes/qwen_image_rocm_fast /opt/ComfyUI/custom_nodes/qwen_image_rocm_fast
COPY workflows /opt/qwen/workflows
COPY models/manifest.json /opt/qwen/models/manifest.json
COPY scripts/entrypoint.sh scripts/download_models.py scripts/benchmark.py /opt/qwen/scripts/

RUN python - <<'PY'
import hashlib
import importlib.metadata as metadata
from pathlib import Path
import torch
from comfy_kitchen.backends.hip import _C

extension = Path(_C.__file__).resolve()
assert hashlib.sha256(extension.read_bytes()).hexdigest() == "0328285d26d04786c0e72a463db5952bf8c9f9aab4544d88d71ce595fc45560d"
assert hasattr(_C, "sage_sdpa")
assert hasattr(_C, "rms_rope")
assert torch.__version__ == "2.10.0+rocm7.2.4.git3d3aa833"
assert metadata.version("torch") == "2.10.0+rocm7.2.4.lw.git3d3aa833"
assert torch.version.hip == "7.2.53211"
assert metadata.version("comfy-kitchen") == "0.2.35"
assert metadata.version("transformers") == "5.17.0"
assert Path("/opt/ComfyUI/main.py").is_file()
PY

RUN mkdir -p /opt/ComfyUI/models /opt/ComfyUI/input /opt/ComfyUI/output \
      /opt/ComfyUI/user /var/log/comfyui \
    && chmod 0755 /opt/qwen/scripts/entrypoint.sh

EXPOSE 8188
ENTRYPOINT ["/opt/qwen/scripts/entrypoint.sh"]
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-auto-launch"]
