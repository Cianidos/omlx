# SPDX-License-Identifier: Apache-2.0
"""Coarse-to-exact lm_head readout for greedy MTP drafts."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_COARSE_BITS = 2
_COARSE_GROUP_SIZE = 64
_SHORTLIST_SIZE = 32
_REQUANTIZE_CHUNK_ROWS = 65536
_MIN_FREE_MEMORY_BYTES = 2 * 1024**3
_RERANK_ENV = "OMLX_MTP_DRAFT_RERANK"
_TOP32_TILES = 64
_TOP32_THREADGROUP = 256
_TOP32_CANDIDATES = _TOP32_TILES * _SHORTLIST_SIZE

_TOP32_HEADER = r"""
inline uint omlx_mtp_top32_ordinal(float v) {
    if (isnan(v))  { return 0xFFFFFFFFu; }
    if (v == 0.0f) { return 0x80000000u; }
    uint u = as_type<uint>(v);
    return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
}
"""

_TOP32_PARTIAL_SOURCE = r"""
constexpr uint REAL_COUNT = uint(RC);
constexpr uint TG_SIZE = 256;
constexpr uint STRIDE = 64u * TG_SIZE;
constexpr uint PER_THREAD = (REAL_COUNT + STRIDE - 1u) / STRIDE;
constexpr uint TOPK = 32;
constexpr uint SIMD_SIZE = 32;
constexpr uint NSIMD = TG_SIZE / SIMD_SIZE;
constexpr uint PB = (NSIMD * TOPK) / SIMD_SIZE;

uint tile = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;

uint ord[PER_THREAD];
uint idx[PER_THREAD];
for (uint t = 0; t < PER_THREAD; ++t) { ord[t] = 0u; idx[t] = 0u; }
uint n = 0;
for (uint i = tile * TG_SIZE + tid; i < REAL_COUNT; i += STRIDE) {
    ord[n] = omlx_mtp_top32_ordinal(float(logits[i]));
    idx[n] = i;
    n++;
}

threadgroup uint shared_ord[NSIMD * TOPK];
threadgroup uint shared_idx[NSIMD * TOPK];
uint taken = 0u;
for (uint r = 0; r < TOPK; ++r) {
    uint best_ord = 0u, best_idx = 0u, best_slot = 0xFFFFFFFFu;
    for (uint t = 0; t < PER_THREAD; ++t) {
        if ((taken & (1u << t)) != 0u) continue;
        if (ord[t] > best_ord || (ord[t] == best_ord && idx[t] > best_idx)) {
            best_ord = ord[t]; best_idx = idx[t]; best_slot = t;
        }
    }
    uint max_ord = simd_max(best_ord);
    uint max_idx = simd_max((best_ord == max_ord) ? best_idx : 0u);
    if (best_slot != 0xFFFFFFFFu && best_ord == max_ord && best_idx == max_idx) {
        taken |= (1u << best_slot);
    }
    if (lane == 0) {
        shared_ord[sg * TOPK + r] = max_ord;
        shared_idx[sg * TOPK + r] = max_idx;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (sg == 0) {
    uint ord2[PB];
    uint idx2[PB];
    for (uint t = 0; t < PB; ++t) {
        uint p = t * SIMD_SIZE + lane;
        ord2[t] = shared_ord[p];
        idx2[t] = shared_idx[p];
    }
    uint taken2 = 0u;
    for (uint r = 0; r < TOPK; ++r) {
        uint best_ord = 0u, best_idx = 0u, best_slot = 0xFFFFFFFFu;
        for (uint t = 0; t < PB; ++t) {
            if ((taken2 & (1u << t)) != 0u) continue;
            if (ord2[t] > best_ord || (ord2[t] == best_ord && idx2[t] > best_idx)) {
                best_ord = ord2[t]; best_idx = idx2[t]; best_slot = t;
            }
        }
        uint max_ord = simd_max(best_ord);
        uint max_idx = simd_max((best_ord == max_ord) ? best_idx : 0u);
        if (best_slot != 0xFFFFFFFFu && best_ord == max_ord && best_idx == max_idx) {
            taken2 |= (1u << best_slot);
        }
        if (lane == 0) {
            candidate_ordinals[tile * TOPK + r] = max_ord;
            candidate_indices[tile * TOPK + r] = max_idx;
        }
    }
}
"""

_TOP32_FINAL_SOURCE = r"""
constexpr uint TG_SIZE = 256;
constexpr uint PER_THREAD = 8;
constexpr uint TOPK = 32;
constexpr uint SIMD_SIZE = 32;
constexpr uint NSIMD = TG_SIZE / SIMD_SIZE;
constexpr uint PB = (NSIMD * TOPK) / SIMD_SIZE;

uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint ord[PER_THREAD];
uint idx[PER_THREAD];
for (uint t = 0; t < PER_THREAD; ++t) {
    uint p = t * TG_SIZE + tid;
    ord[t] = candidate_ordinals[p];
    idx[t] = candidate_indices[p];
}

threadgroup uint shared_ord[NSIMD * TOPK];
threadgroup uint shared_idx[NSIMD * TOPK];
uint taken = 0u;
for (uint r = 0; r < TOPK; ++r) {
    uint best_ord = 0u, best_idx = 0u, best_slot = 0xFFFFFFFFu;
    for (uint t = 0; t < PER_THREAD; ++t) {
        if ((taken & (1u << t)) != 0u) continue;
        if (ord[t] > best_ord || (ord[t] == best_ord && idx[t] > best_idx)) {
            best_ord = ord[t]; best_idx = idx[t]; best_slot = t;
        }
    }
    uint max_ord = simd_max(best_ord);
    uint max_idx = simd_max((best_ord == max_ord) ? best_idx : 0u);
    if (best_slot != 0xFFFFFFFFu && best_ord == max_ord && best_idx == max_idx) {
        taken |= (1u << best_slot);
    }
    if (lane == 0) {
        shared_ord[sg * TOPK + r] = max_ord;
        shared_idx[sg * TOPK + r] = max_idx;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (sg == 0) {
    uint ord2[PB];
    uint idx2[PB];
    for (uint t = 0; t < PB; ++t) {
        uint p = t * SIMD_SIZE + lane;
        ord2[t] = shared_ord[p];
        idx2[t] = shared_idx[p];
    }
    uint taken2 = 0u;
    for (uint r = 0; r < TOPK; ++r) {
        uint best_ord = 0u, best_idx = 0u, best_slot = 0xFFFFFFFFu;
        for (uint t = 0; t < PB; ++t) {
            if ((taken2 & (1u << t)) != 0u) continue;
            if (ord2[t] > best_ord || (ord2[t] == best_ord && idx2[t] > best_idx)) {
                best_ord = ord2[t]; best_idx = idx2[t]; best_slot = t;
            }
        }
        uint max_ord = simd_max(best_ord);
        uint max_idx = simd_max((best_ord == max_ord) ? best_idx : 0u);
        if (best_slot != 0xFFFFFFFFu && best_ord == max_ord && best_idx == max_idx) {
            taken2 |= (1u << best_slot);
        }
        if (lane == 0) token_ids[TOPK - 1u - r] = max_idx;
    }
}
"""

_top32_partial = None
_top32_final = None


def _inner_model(model: Any) -> Any:
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            return inner
    return model


def _trunk_head(model: Any) -> Any:
    inner = _inner_model(model)
    if bool(getattr(getattr(inner, "args", None), "tie_word_embeddings", False)):
        return getattr(getattr(inner, "model", None), "embed_tokens", None)
    return getattr(inner, "lm_head", None)


def _rerank_requested() -> bool | None:
    value = os.environ.get(_RERANK_ENV)
    if value is None:
        return None
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _head_parameters(head: Any) -> tuple[Any, Any, Any, int, int, str] | None:
    if not isinstance(head, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        return None
    if getattr(head, "mode", None) != "affine":
        return None
    bits = int(getattr(head, "bits", 0) or 0)
    group_size = int(getattr(head, "group_size", 0) or 0)
    weight = getattr(head, "weight", None)
    scales = getattr(head, "scales", None)
    biases = getattr(head, "biases", None)
    if (
        not isinstance(weight, mx.array)
        or not isinstance(scales, mx.array)
        or not isinstance(biases, mx.array)
        or weight.ndim != 2
        or weight.shape[0] < _SHORTLIST_SIZE
        or bits <= _COARSE_BITS
        or group_size <= 0
    ):
        return None
    return weight, scales, biases, group_size, bits, head.mode


def _coarse_head_bytes(rows: int, hidden_size: int) -> int:
    packed = rows * hidden_size * _COARSE_BITS // 8
    tables = 2 * rows * (hidden_size // _COARSE_GROUP_SIZE) * 2
    return packed + tables


def _top32(logits: Any) -> Any:
    global _top32_partial, _top32_final

    rows = int(logits.size)
    per_thread = (rows + _TOP32_TILES * _TOP32_THREADGROUP - 1) // (
        _TOP32_TILES * _TOP32_THREADGROUP
    )
    if rows < _TOP32_TILES * _TOP32_THREADGROUP or per_thread > 32:
        raise ValueError(f"unsupported MTP top-32 width: {rows}")
    if _top32_partial is None:
        _top32_partial = mx.fast.metal_kernel(
            name="omlx_mtp_top32_partial",
            input_names=["logits"],
            output_names=["candidate_ordinals", "candidate_indices"],
            source=_TOP32_PARTIAL_SOURCE,
            header=_TOP32_HEADER,
        )
    if _top32_final is None:
        _top32_final = mx.fast.metal_kernel(
            name="omlx_mtp_top32_final",
            input_names=["candidate_ordinals", "candidate_indices"],
            output_names=["token_ids"],
            source=_TOP32_FINAL_SOURCE,
        )
    candidate_ordinals, candidate_indices = _top32_partial(
        inputs=[logits.reshape(-1)],
        template=[("RC", rows)],
        grid=(_TOP32_TILES * _TOP32_THREADGROUP, 1, 1),
        threadgroup=(_TOP32_THREADGROUP, 1, 1),
        output_shapes=[(_TOP32_CANDIDATES,), (_TOP32_CANDIDATES,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    return _top32_final(
        inputs=[candidate_ordinals, candidate_indices],
        grid=(_TOP32_THREADGROUP, 1, 1),
        threadgroup=(_TOP32_THREADGROUP, 1, 1),
        output_shapes=[(_SHORTLIST_SIZE,)],
        output_dtypes=[mx.uint32],
    )[0]


def _fits_memory(coarse_bytes: int, *, forced: bool) -> bool:
    if forced:
        return True
    try:
        from omlx.process_memory_enforcer import get_effective_metal_cap_bytes

        cap = int(get_effective_metal_cap_bytes() or 0)
        active = int(mx.get_active_memory())
    except Exception:
        return False
    return cap > 0 and cap - active >= coarse_bytes + _MIN_FREE_MEMORY_BYTES


def _array_bytes(array: Any) -> int:
    return int(array.size) * int(array.itemsize)


def _requantize_rows(
    weight: Any,
    scales: Any,
    biases: Any,
    *,
    source_group_size: int,
    source_bits: int,
    source_mode: str,
    chunk_rows: int = _REQUANTIZE_CHUNK_ROWS,
) -> tuple[Any, Any, Any]:
    weights = []
    scale_parts = []
    bias_parts = []
    for start in range(0, weight.shape[0], chunk_rows):
        stop = min(weight.shape[0], start + chunk_rows)
        dense = mx.dequantize(
            weight[start:stop],
            scales[start:stop],
            biases[start:stop],
            group_size=source_group_size,
            bits=source_bits,
            mode=source_mode,
            dtype=mx.bfloat16,
        )
        quantized = mx.quantize(
            dense,
            group_size=_COARSE_GROUP_SIZE,
            bits=_COARSE_BITS,
            mode="affine",
        )
        mx.eval(*quantized)
        weights.append(quantized[0])
        scale_parts.append(quantized[1])
        bias_parts.append(quantized[2])
    return (
        mx.concatenate(weights, axis=0),
        mx.concatenate(scale_parts, axis=0),
        mx.concatenate(bias_parts, axis=0),
    )


def build(model: Any) -> bool:
    """Build one load-time 2-bit copy of a quantized Qwen trunk lm_head."""
    inner = _inner_model(model)
    if getattr(inner, "_omlx_mtp_draft_rerank_tried", False):
        return getattr(inner, "_omlx_mtp_draft_rerank", None) is not None
    inner._omlx_mtp_draft_rerank_tried = True

    requested = _rerank_requested()
    if requested is False:
        return False
    if not bool(getattr(inner, "_omlx_mtp_decode_enabled", False)):
        return False
    model_type = str(getattr(getattr(inner, "args", None), "model_type", ""))
    if not model_type.startswith(("qwen3_5", "qwen3_6")):
        return False

    head = _trunk_head(inner)
    parameters = _head_parameters(head)
    if parameters is None:
        return False
    weight, scales, biases, group_size, bits, mode = parameters
    hidden_size = group_size * scales.shape[1]
    coarse_bytes = _coarse_head_bytes(weight.shape[0], hidden_size)
    if not _fits_memory(coarse_bytes, forced=requested is True):
        logger.info(
            "MTP draft rerank skipped: coarse lm_head needs %.0f MiB plus 2 GiB headroom",
            coarse_bytes / 1024**2,
        )
        return False

    start = time.perf_counter()
    try:
        coarse = _requantize_rows(
            weight,
            scales,
            biases,
            source_group_size=group_size,
            source_bits=bits,
            source_mode=mode,
        )
        mx.eval(*coarse)
        resident_bytes = sum(_array_bytes(array) for array in coarse)
        if resident_bytes != coarse_bytes:
            raise ValueError(
                f"coarse lm_head size mismatch: {resident_bytes} != {coarse_bytes}"
            )
        mx.eval(_top32(mx.zeros((weight.shape[0],), dtype=mx.float32)))
    except Exception:
        logger.warning(
            "MTP draft rerank build failed; using full lm_head", exc_info=True
        )
        return False

    inner._omlx_mtp_draft_rerank = coarse
    inner._omlx_mtp_draft_rerank_head = head
    inner._omlx_mtp_draft_rerank_logged = False
    logger.info(
        "MTP draft rerank ready: 2-bit/gs64 coarse lm_head %.0f MiB, top-%d exact rescore (%.0f ms)",
        coarse_bytes / 1024**2,
        _SHORTLIST_SIZE,
        (time.perf_counter() - start) * 1000,
    )
    return True


def available(model: Any) -> bool:
    return getattr(_inner_model(model), "_omlx_mtp_draft_rerank", None) is not None


def select(
    model: Any, hidden: Any, processors: Any, prev_tokens: Any
) -> tuple[Any, Any]:
    """Return greedy draft id and shortlisted logprobs."""
    inner = _inner_model(model)
    coarse = getattr(inner, "_omlx_mtp_draft_rerank", None)
    head = getattr(inner, "_omlx_mtp_draft_rerank_head", None)
    parameters = _head_parameters(head)
    if coarse is None or parameters is None:
        raise RuntimeError("MTP draft rerank is unavailable")

    hidden = hidden.reshape(-1, hidden.shape[-1])
    coarse_logits = mx.quantized_matmul(
        hidden,
        coarse[0],
        scales=coarse[1],
        biases=coarse[2],
        transpose=True,
        group_size=_COARSE_GROUP_SIZE,
        bits=_COARSE_BITS,
        mode="affine",
    )
    if "bias" in head:
        coarse_logits = coarse_logits + head.bias
    processed_coarse = coarse_logits
    processor_states = []
    if processors:
        processed_coarse = coarse_logits + 0
        processor_states = [
            (processor, processor.snapshot_state())
            for processor in processors
            if hasattr(processor, "snapshot_state")
        ]
        for processor in processors:
            processed_coarse = processor(prev_tokens, processed_coarse)
    candidates = mx.sort(_top32(processed_coarse))
    for processor, state in processor_states:
        processor.restore_state(state)

    weight, scales, biases, group_size, bits, mode = parameters
    flat_candidates = candidates.reshape(-1)
    exact = mx.quantized_matmul(
        hidden,
        weight[flat_candidates],
        scales=scales[flat_candidates],
        biases=biases[flat_candidates],
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    if "bias" in head:
        exact = exact + head.bias[flat_candidates]
    if processors:
        exact = coarse_logits.at[:, flat_candidates].add(
            exact - coarse_logits[:, flat_candidates]
        )
        for processor in processors:
            exact = processor(prev_tokens, exact)
        exact = exact[:, flat_candidates]
    shortlist_lp = exact - mx.logsumexp(exact, axis=-1, keepdims=True)
    picked = mx.take(
        candidates,
        mx.argmax(exact, axis=-1).reshape(-1),
        axis=0,
    )
    if not getattr(inner, "_omlx_mtp_draft_rerank_logged", False):
        logger.info("MTP draft rerank engaged: 2-bit coarse, top-32 exact rescore")
        inner._omlx_mtp_draft_rerank_logged = True
    return picked.reshape(-1).astype(mx.uint32), shortlist_lp.reshape(-1)
