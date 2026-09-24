from __future__ import annotations

import functools
import math
import os
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import torch
import comfy_kitchen as ck


MIN_AUTO_QUERY_TOKENS = 512
SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
VALID_MODES = {"reference", "candidate", "auto"}


@dataclass
class DispatchResult:
    output: torch.Tensor
    backend: str
    fallback: bool
    reason: str | None


class AttentionBackend:
    def __init__(
        self,
        reference: Callable[..., torch.Tensor],
        candidate: Callable[..., torch.Tensor],
        *,
        mode: str = "auto",
        min_auto_query_tokens: int = MIN_AUTO_QUERY_TOKENS,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"QWEN_P1_BACKEND must be one of {sorted(VALID_MODES)}, got {mode!r}")
        if min_auto_query_tokens < 1:
            raise ValueError("min_auto_query_tokens must be positive")
        self.reference = reference
        self.candidate = candidate
        self.mode = mode
        self.min_auto_query_tokens = min_auto_query_tokens
        self._counts: Counter[tuple[str, str | None]] = Counter()
        self._lock = threading.Lock()

    def capabilities(self) -> dict[str, Any]:
        return {
            "available": bool(ck.int8_attention_is_available()),
            "dtypes": sorted(str(dtype) for dtype in SUPPORTED_DTYPES),
            "max_head_dim": 256,
            "gqa": True,
            "mask": ["bool", "float16", "bfloat16", "float32"],
            "odd_lengths": True,
            "min_auto_query_tokens": self.min_auto_query_tokens,
        }

    def supports(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
        attn_precision: torch.dtype | None = None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
        **kwargs: Any,
    ) -> tuple[bool, str | None]:
        del attn_precision, skip_output_reshape
        enable_gqa = bool(kwargs.get("enable_gqa", False))
        scale = kwargs.get("scale")
        transformer_options = kwargs.get("transformer_options")
        preferred_attention = kwargs.get("preferred_attention")
        if not ck.int8_attention_is_available():
            return False, "int8_attention_unavailable"
        if transformer_options and transformer_options.get("optimized_attention_override") is not None:
            return False, "attention_override_present"
        if getattr(preferred_attention, "function", None) is not None:
            return False, "preferred_attention_present"

        expected_dims = 4 if skip_reshape else 3
        if any(tensor.ndim != expected_dims for tensor in (q, k, v)):
            return False, f"expected_{expected_dims}d_qkv"
        if q.dtype not in SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
            return False, "unsupported_or_mixed_dtype"
        if not q.is_cuda or q.device != k.device or q.device != v.device:
            return False, "qkv_device_mismatch"
        if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
            return False, "last_dimension_not_contiguous"
        if not isinstance(heads, int) or heads <= 0:
            return False, "invalid_query_heads"

        if skip_reshape:
            batch, query_heads, query, head_dim = q.shape
            key_batch, kv_heads, key, key_dim = k.shape
            value_batch, value_heads, value_length, value_dim = v.shape
            if query_heads != heads:
                return False, "heads_argument_mismatch"
        else:
            batch, query, query_width = q.shape
            key_batch, key, key_width = k.shape
            value_batch, value_length, value_width = v.shape
            if query_width % heads:
                return False, "query_width_not_divisible_by_heads"
            query_heads = heads
            head_dim = query_width // heads
            if head_dim <= 0 or key_width % head_dim or value_width % head_dim:
                return False, "kv_width_not_divisible_by_head_dim"
            kv_heads = key_width // head_dim
            value_heads = value_width // head_dim
            key_dim = value_dim = head_dim

        if min(batch, query_heads, kv_heads, query, key, head_dim) <= 0:
            return False, "empty_dimension"
        if batch != key_batch or batch != value_batch:
            return False, "batch_mismatch"
        if key != value_length or kv_heads != value_heads:
            return False, "kv_shape_mismatch"
        if head_dim != key_dim or head_dim != value_dim:
            return False, "head_dim_mismatch"
        if query_heads % kv_heads:
            return False, "query_heads_not_divisible_by_kv_heads"
        if not enable_gqa and query_heads != kv_heads:
            return False, "gqa_not_enabled"
        if head_dim > 256:
            return False, "head_dim_above_256"
        if scale is not None and not math.isfinite(float(scale)):
            return False, "non_finite_scale"

        if mask is not None:
            if mask.device != q.device:
                return False, "mask_device_mismatch"
            if mask.dtype not in SUPPORTED_DTYPES | {torch.bool}:
                return False, "unsupported_mask_dtype"
            mask_shape = tuple(mask.shape)
            if mask.ndim == 2:
                mask_shape = (1, 1, *mask_shape)
            elif mask.ndim == 3:
                mask_shape = (mask_shape[0], 1, *mask_shape[1:])
            target_shape = (batch, query_heads, query, key)
            try:
                broadcast_shape = torch.broadcast_shapes(mask_shape, target_shape)
            except RuntimeError:
                return False, "mask_not_broadcastable"
            if broadcast_shape != target_shape:
                return False, "mask_not_broadcastable"
        return True, None

    @staticmethod
    def query_length(q: torch.Tensor, skip_reshape: bool = False, **_kwargs: Any) -> int:
        return int(q.shape[-2] if skip_reshape else q.shape[1])

    def select(self, *args: Any, **kwargs: Any) -> tuple[str, str | None]:
        if self.mode == "reference":
            return "reference", None
        supported, reason = self.supports(*args, **kwargs)
        if not supported:
            return "reference", reason
        if self.mode == "auto":
            skip_reshape = bool(args[6]) if len(args) > 6 else bool(kwargs.get("skip_reshape", False))
            query = self.query_length(args[0], skip_reshape=skip_reshape)
            if query < self.min_auto_query_tokens:
                return "reference", f"auto_query_below_{self.min_auto_query_tokens}"
        return "candidate", None

    def run(self, *args: Any, **kwargs: Any) -> DispatchResult:
        selected, reason = self.select(*args, **kwargs)
        if selected == "reference":
            result = DispatchResult(
                self.reference(*args, **kwargs),
                "reference",
                reason is not None,
                reason,
            )
        else:
            try:
                result = DispatchResult(
                    self.candidate(*args, **kwargs), "candidate", False, None
                )
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                result = DispatchResult(
                    self.reference(*args, **kwargs), "reference", True, reason
                )
        with self._lock:
            self._counts[(result.backend, result.reason)] += 1
        return result

    def attention(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.run(*args, **kwargs).output

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                f"{backend}:{reason or 'selected'}": count
                for (backend, reason), count in sorted(
                    self._counts.items(),
                    key=lambda item: (item[0][0], item[0][1] or ""),
                )
            }


def install(
    mode: str | None = None,
    *,
    min_auto_query_tokens: int = MIN_AUTO_QUERY_TOKENS,
) -> AttentionBackend:
    import comfy.ldm.modules.attention as attention_module
    import comfy.ldm.qwen_image21.model as qwen_module

    existing = getattr(qwen_module.optimized_attention, "_qwen_p1_backend", None)
    if isinstance(existing, AttentionBackend):
        return existing
    selected_mode = mode or os.getenv("QWEN_P1_BACKEND", "auto")
    backend = AttentionBackend(
        qwen_module.optimized_attention,
        attention_module.attention_comfy_kitchen_int8,
        mode=selected_mode,
        min_auto_query_tokens=min_auto_query_tokens,
    )

    @functools.wraps(qwen_module.optimized_attention)
    def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        return backend.attention(*args, **kwargs)

    wrapped._qwen_p1_backend = backend
    qwen_module.optimized_attention = wrapped
    return backend
