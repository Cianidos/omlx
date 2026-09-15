# Greedy MTP Draft Reranking

This branch reduces Qwen native-MTP draft cost by replacing each greedy full-vocabulary trunk-head projection with a coarse shortlist followed by exact rescoring.

## Scope

Supported model families:

- Qwen3.5, Qwen3.6, and Qwen3.8 text backbones
- Dense and MoE variants
- Text and VLM engine paths
- Separate affine-quantized `lm_head` modules and tied affine-quantized embeddings

Other MTP families remain on their existing readout paths. Dense-bfloat16 heads, non-affine heads, and heads already at 2 bits are also unchanged.

## Load-time setup

When Lightning MTP is enabled, model loading:

1. Resolves the trunk output head and its actual quantization geometry.
2. Checks that a 2-bit/group-size-64 side copy plus 2 GiB of headroom fits under the effective Metal working-set cap.
3. Requantizes the head in 65,536-row chunks using `mx.dequantize` and `mx.quantize`.
4. Materializes a 2-bit affine copy and warms the two-stage Metal top-32 selector.

The side copy stays outside the MLX module parameter tree. It is not serialized as a model weight and is released with the model.

Set `OMLX_MTP_DRAFT_RERANK=0` to disable the path. Set it to `1` to force the build past the automatic memory-headroom gate; unsupported head formats still fall back.

## Greedy draft path

Each greedy draft step performs:

1. A 2-bit full-vocabulary quantized matrix multiplication.
2. An exact two-dispatch Metal top-32 selection.
3. Row gathers from the original trunk head's packed weights, scales, biases, and optional output bias.
4. A 32-row quantized matrix multiplication using the trunk head's original precision.
5. Argmax over the exact shortlist scores.

The MTP head returns its post-norm hidden state without projecting full logits while this path is active. The first history fold and every chained draft use the same hidden-only contract.

Logits processors run on both shortlist selection and exact rescoring. Stateful processors are snapshotted around the coarse pass, then advanced once by the exact pass.

Stochastic drafting remains unchanged and keeps the full-vocabulary readout plus its sharper temperature/top-p/top-k sampler.

## Correctness

The coarse head only chooses candidates. Final ordering within the shortlist comes from original trunk-head rows. When the target argmax appears in the shortlist, the proposed draft matches the full trunk-head argmax, modulo its existing tie rule. A shortlist miss can lower draft acceptance but cannot change verified output.

Live Qwen3.8-27B checks found:

- 548/548 greedy proposals matched full trunk-head argmax across code and prose workloads.
- Greedy generated text was byte-identical between full-readout and rerank runs.
- Stochastic runs never engaged the rerank path.
- Model unload returned MLX active memory to baseline.

## Performance

Measured on Apple M5 Pro, 48 GiB, with `Jundot/Qwen3.8-27B-oQ4e-mtp`:

- Full trunk-head readout: 2.68 ms median
- Coarse top-32 plus exact rescore: 1.59 ms median
- Draft-readout reduction: about 40%
- PP1024/TG256 static-prompt ABBA decode throughput:
  - Full readout: 33.78 tok/s mean
  - Rerank: 34.33 tok/s mean
  - Gain: 1.63%
- Adaptive acceptance over the same ABBA runs:
  - Full readout: 655/859, 76.25%
  - Rerank: 657/859, 76.48%
- Resident side-copy cost: 379 MiB

End-to-end gain is smaller than readout reduction because target verification dominates each MTP cycle. This change only reduces draft output-head work.

## Validation

Coverage includes chunked requantization, exact rescoring, logits processors, stateful processor replay, memory and architecture gates, forced mode, Metal top-32 selection, sidecar ownership, tied embeddings, VLM plumbing, and fallback compatibility.

Latest complete worktree run:

- 13,236 passed
- 342 skipped
- 82 deselected

Focused MTP and engine suites, Ruff, Black, and `git diff --check` also pass.
