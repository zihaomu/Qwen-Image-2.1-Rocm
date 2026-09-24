#!/usr/bin/env python3
"""Download and verify the exact ComfyUI-format Qwen-Image-2.1 weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = Path(os.environ.get("QWEN_MODEL_ROOT", ROOT / "models")).resolve()
MANIFEST_PATH = Path(os.environ.get("QWEN_MODEL_MANIFEST", ROOT / "models" / "manifest.json"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or len(manifest.get("files", [])) != 3:
        raise RuntimeError(f"invalid model manifest: {MANIFEST_PATH}")
    return manifest


def verify(manifest: dict) -> None:
    for item in manifest["files"]:
        path = MODEL_ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["size_bytes"]:
            raise RuntimeError(f"size mismatch: {path}")
        actual = digest(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"sha256 mismatch: {path}: {actual}")
        print(f"OK {item['sha256']} {item['size_bytes']} {path}")


def download(manifest: dict) -> None:
    from modelscope import snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    allow_patterns = [item["path"] for item in manifest["files"]]
    snapshot_download(
        manifest["repository"],
        revision=manifest["revision"],
        local_dir=str(MODEL_ROOT),
        allow_patterns=allow_patterns,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    if not args.verify_only:
        download(manifest)
    verify(manifest)
    print("MODEL_SET_VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
