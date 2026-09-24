from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import logging
from pathlib import Path
from typing import Any

from aiohttp import web
from server import PromptServer

from .backend import install


EXPECTED_EXTENSION_SHA256 = "0328285d26d04786c0e72a463db5952bf8c9f9aab4544d88d71ce595fc45560d"
BACKEND = install("candidate")
logging.info("Qwen Image ROCm fast attention dispatcher installed in candidate mode")


def _extension_info() -> dict[str, Any]:
    from comfy_kitchen.backends.hip import _C

    path = Path(_C.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "expected_sha256": EXPECTED_EXTENSION_SHA256,
        "verified": digest == EXPECTED_EXTENSION_SHA256,
        "sage_sdpa": hasattr(_C, "sage_sdpa"),
        "rms_rope": hasattr(_C, "rms_rope"),
    }


@PromptServer.instance.routes.get("/qwen_image_rocm_fast/status")
async def fast_status(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "configured": True,
            "mode": BACKEND.mode,
            "capabilities": BACKEND.capabilities(),
            "stats": BACKEND.stats(),
            "extension": _extension_info(),
            "versions": {
                "comfy-kitchen": metadata.version("comfy-kitchen"),
                "torch": metadata.version("torch"),
            },
        }
    )


NODE_CLASS_MAPPINGS = {}
