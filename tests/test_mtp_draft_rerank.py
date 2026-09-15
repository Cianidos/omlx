# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.mlx_lm_mtp import draft_rerank


def _quantized_head(rows=96, hidden=128, bits=4):
    dense = mx.random.normal((rows, hidden), dtype=mx.bfloat16)
    head = nn.Linear(hidden, rows, bias=False).to_quantized(group_size=64, bits=bits)
    head.weight, head.scales, head.biases = mx.quantize(dense, group_size=64, bits=bits)
    model = SimpleNamespace(
        args=SimpleNamespace(
            tie_word_embeddings=False,
            model_type="qwen3_5_text",
        ),
        lm_head=head,
        _omlx_mtp_decode_enabled=True,
    )
    return model, head


def test_requantize_rows_matches_one_shot():
    _model, head = _quantized_head()

    chunked = draft_rerank._requantize_rows(
        head.weight,
        head.scales,
        head.biases,
        source_group_size=64,
        source_bits=4,
        source_mode="affine",
        chunk_rows=24,
    )
    dense = mx.dequantize(
        head.weight,
        head.scales,
        head.biases,
        group_size=64,
        bits=4,
        mode="affine",
        dtype=mx.bfloat16,
    )
    expected = mx.quantize(dense, group_size=64, bits=2, mode="affine")

    for actual, reference in zip(chunked, expected):
        assert mx.array_equal(actual, reference).item()


def _install_coarse(model, head):
    coarse = draft_rerank._requantize_rows(
        head.weight,
        head.scales,
        head.biases,
        source_group_size=64,
        source_bits=4,
        source_mode="affine",
        chunk_rows=24,
    )
    model._omlx_mtp_draft_rerank = coarse
    model._omlx_mtp_draft_rerank_head = head
    model._omlx_mtp_draft_rerank_logged = False
    return coarse


def _test_top32(logits):
    return mx.argpartition(-logits.reshape(-1), kth=31)[:32]


def test_exact_rescore_returns_full_head_argmax_when_shortlisted(monkeypatch):
    model, head = _quantized_head()
    coarse = _install_coarse(model, head)

    monkeypatch.setattr(draft_rerank, "_top32", _test_top32)
    for _ in range(16):
        hidden = mx.random.normal((1, 1, 128), dtype=mx.bfloat16)
        token, shortlist_lp = draft_rerank.select(model, hidden, None, None)
        full_argmax = int(mx.argmax(head(hidden), axis=-1).item())
        candidates = _test_top32(
            mx.quantized_matmul(
                hidden,
                *coarse,
                transpose=True,
                group_size=64,
                bits=2,
            )
        )
        assert mx.any(candidates == full_argmax).item()
        assert int(token.item()) == full_argmax
        assert shortlist_lp.shape == (32,)


def test_exact_rescore_applies_logits_processors(monkeypatch):
    from mlx_lm.sample_utils import make_repetition_penalty

    model, head = _quantized_head()
    coarse = _install_coarse(model, head)
    monkeypatch.setattr(draft_rerank, "_top32", _test_top32)
    processor = make_repetition_penalty(1.1, 20)
    previous = mx.arange(20, dtype=mx.int32)

    for _ in range(16):
        hidden = mx.random.normal((1, 1, 128), dtype=mx.bfloat16)
        token, _ = draft_rerank.select(model, hidden, [processor], previous)
        exact = processor(previous, head(hidden).reshape(1, -1))
        coarse_logits = processor(
            previous,
            mx.quantized_matmul(
                hidden.reshape(1, -1),
                *coarse,
                transpose=True,
                group_size=64,
                bits=2,
            ),
        )
        candidates = _test_top32(coarse_logits)
        full_argmax = int(mx.argmax(exact, axis=-1).item())
        if mx.any(candidates == full_argmax).item():
            selected = int(token.item())
            assert float(exact[0, selected]) == float(exact[0, full_argmax])


def test_processors_see_unmodified_exact_logits(monkeypatch):
    model, head = _quantized_head()
    _install_coarse(model, head)
    monkeypatch.setattr(draft_rerank, "_top32", _test_top32)
    seen = []

    def processor(_tokens, logits):
        seen.append(logits + 0)
        logits[:, 0] = -float("inf")
        return logits

    hidden = mx.random.normal((1, 1, 128), dtype=mx.bfloat16)
    draft_rerank.select(model, hidden, [processor], mx.array([], dtype=mx.int32))

    assert len(seen) == 2
    assert mx.isfinite(seen[1]).all().item()


def test_stateful_processors_advance_once(monkeypatch):
    model, head = _quantized_head()
    _install_coarse(model, head)
    monkeypatch.setattr(draft_rerank, "_top32", _test_top32)

    class Processor:
        def __init__(self):
            self.calls = 0

        def __call__(self, _tokens, logits):
            self.calls += 1
            return logits

        def snapshot_state(self):
            return self.calls

        def restore_state(self, state):
            self.calls = state

    processor = Processor()
    hidden = mx.random.normal((1, 1, 128), dtype=mx.bfloat16)
    draft_rerank.select(model, hidden, [processor], mx.array([], dtype=mx.int32))

    assert processor.calls == 1


def test_build_gates_dense_and_low_memory_heads(monkeypatch):
    dense = SimpleNamespace(
        args=SimpleNamespace(
            tie_word_embeddings=False,
            model_type="qwen3_5_text",
        ),
        lm_head=nn.Linear(128, 96, bias=False),
        _omlx_mtp_decode_enabled=True,
    )
    assert draft_rerank.build(dense) is False

    quantized, _head = _quantized_head()
    monkeypatch.delenv(draft_rerank._RERANK_ENV, raising=False)
    monkeypatch.setattr(draft_rerank, "_fits_memory", lambda *_args, **_kwargs: False)
    assert draft_rerank.build(quantized) is False
    assert getattr(quantized, "_omlx_mtp_draft_rerank", None) is None


def test_build_rejects_non_qwen_model(monkeypatch):
    model, _head = _quantized_head()
    model.args.model_type = "deepseek_v4"
    monkeypatch.setenv(draft_rerank._RERANK_ENV, "1")
    assert draft_rerank.build(model) is False


def test_forced_build_bypasses_memory_gate(monkeypatch):
    model, _head = _quantized_head()
    monkeypatch.setenv(draft_rerank._RERANK_ENV, "1")
    monkeypatch.setattr(
        draft_rerank,
        "_top32",
        lambda logits: mx.argpartition(-logits.reshape(-1), kth=31)[:32],
    )
    seen = {}

    def fits(_bytes, *, forced):
        seen["forced"] = forced
        return True

    monkeypatch.setattr(draft_rerank, "_fits_memory", fits)
    assert draft_rerank.build(model) is True
    assert seen == {"forced": True}
    assert draft_rerank.available(model) is True


def test_top32_metal_matches_reference_values():
    values = mx.random.normal((248320,), dtype=mx.float32)
    actual = draft_rerank._top32(values)
    expected = mx.argpartition(-values, kth=31)[:32]

    assert mx.array_equal(
        mx.sort(values[actual]),
        mx.sort(values[expected]),
    ).item()
