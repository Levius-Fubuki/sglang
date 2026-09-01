"""Regression tests for destination-backed CUDA-graph LoRA-B expansion."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_source_module(name: str, relative_path: str):
    source = _REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from sglang.srt.lora.torch_ops.graph_lora_ops import sgemm_lora_b_graph_fwd
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError as exc:
    if exc.name != "sglang":
        raise
    register_cpu_ci = _load_source_module(
        "_graph_lora_ci_register", "python/sglang/test/ci/ci_register.py"
    ).register_cpu_ci
    _graph_lora_ops_module = _load_source_module(
        "_graph_lora_ops", "python/sglang/srt/lora/torch_ops/graph_lora_ops.py"
    )
    sgemm_lora_b_graph_fwd = _graph_lora_ops_module.sgemm_lora_b_graph_fwd
else:
    _graph_lora_ops_module = None

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _case(dtype: torch.dtype = torch.float32, num_loras: int = 2):
    torch.manual_seed(7)
    num_tokens = 4
    rank = 2
    slice_widths = (3, 4)
    slice_offsets = torch.tensor((0, 3, 7), dtype=torch.int32)
    inputs = torch.randn(num_tokens, len(slice_widths) * rank, dtype=dtype)
    weights = torch.randn(num_loras, sum(slice_widths), rank, dtype=dtype)
    weight_indices = torch.tensor((0, 1, -1, 0), dtype=torch.int32)
    seg_lens = torch.ones(num_tokens, dtype=torch.int32)

    backing = torch.randn(num_tokens, sum(slice_widths) + 2, dtype=dtype)
    base_output = backing[:, 1:-1]
    return (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        backing,
        base_output,
    )


def _reference(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: torch.Tensor,
) -> torch.Tensor:
    expected = base_output.clone()
    num_slices = len(slice_offsets) - 1
    rank = inputs.shape[-1] // num_slices
    for lora_idx in range(weights.shape[0]):
        rows = (weight_indices == lora_idx).unsqueeze(1)
        masked_inputs = torch.where(rows, inputs, 0)
        for slice_idx in range(num_slices):
            input_start = slice_idx * rank
            input_end = input_start + rank
            output_start = int(slice_offsets[slice_idx])
            output_end = int(slice_offsets[slice_idx + 1])
            expected[:, output_start:output_end].add_(
                masked_inputs[:, input_start:input_end]
                @ weights[lora_idx, output_start:output_end].t()
            )
    return expected


def test_inference_accumulates_directly_into_packed_output(monkeypatch):
    (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        backing,
        base_output,
    ) = _case(num_loras=4)
    expected = _reference(inputs, weights, weight_indices, slice_offsets, base_output)
    padding_before = backing[:, (0, -1)].clone()

    real_addmm = torch.addmm
    calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def recording_addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
        calls.append((input, out))
        return real_addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)

    monkeypatch.setattr(torch, "addmm", recording_addmm)
    with torch.inference_mode():
        actual = sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            base_output,
        )

    assert len(calls) == weights.shape[0] * (len(slice_offsets) - 1)
    assert all(destination is out for destination, out in calls)
    assert actual.data_ptr() == base_output.data_ptr()
    assert not actual.is_contiguous()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(backing[:, (0, -1)], padding_before)


def _forbid_addmm(*args, **kwargs):
    raise AssertionError("the direct-accumulation fast path must not run")


def test_small_lora_pool_uses_existing_fallback(monkeypatch):
    (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        _,
        base_output,
    ) = _case(num_loras=2)
    expected = _reference(inputs, weights, weight_indices, slice_offsets, base_output)

    monkeypatch.setattr(torch, "addmm", _forbid_addmm)
    with torch.inference_mode():
        actual = sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            base_output,
        )

    torch.testing.assert_close(actual, expected)


def test_autograd_uses_existing_fallback_and_preserves_gradients(monkeypatch):
    inputs, weights, weight_indices, seg_lens, slice_offsets, _, _ = _case(
        torch.float64, num_loras=4
    )
    actual_inputs = inputs.clone().requires_grad_()
    actual_weights = weights.clone().requires_grad_()
    expected_inputs = inputs.clone().requires_grad_()
    expected_weights = weights.clone().requires_grad_()

    monkeypatch.setattr(torch, "addmm", _forbid_addmm)
    actual = sgemm_lora_b_graph_fwd(
        actual_inputs,
        actual_weights,
        weight_indices,
        seg_lens,
        slice_offsets,
    )
    expected = _reference(
        expected_inputs,
        expected_weights,
        weight_indices,
        slice_offsets,
        torch.zeros_like(actual),
    )

    actual.square().sum().backward()
    expected.square().sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_inputs.grad, expected_inputs.grad)
    torch.testing.assert_close(actual_weights.grad, expected_weights.grad)


def test_mixed_output_dtype_uses_existing_fallback(monkeypatch):
    (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        _,
        low_precision_base,
    ) = _case(torch.float16, num_loras=4)
    backing = torch.randn(low_precision_base.shape[0], low_precision_base.shape[1] + 2)
    base_output = backing[:, 1:-1]
    expected = _reference(inputs, weights, weight_indices, slice_offsets, base_output)

    monkeypatch.setattr(torch, "addmm", _forbid_addmm)
    with torch.inference_mode():
        actual = sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            base_output,
        )

    assert actual.dtype == torch.float32
    assert actual.data_ptr() == base_output.data_ptr()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_low_precision_whole_output_with_empty_adapter_group(dtype):
    torch.manual_seed(11)
    inputs = torch.randn(3, 4, dtype=dtype)
    weights = torch.randn(4, 5, 4, dtype=dtype)
    weight_indices = torch.tensor((0, -1, 0), dtype=torch.int32)
    seg_lens = torch.ones(3, dtype=torch.int32)
    slice_offsets = torch.tensor((0, 5), dtype=torch.int32)
    base_output = torch.zeros(3, 5, dtype=dtype)
    expected = _reference(inputs, weights, weight_indices, slice_offsets, base_output)

    with torch.inference_mode():
        actual = sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            weight_indices,
            seg_lens,
            slice_offsets,
        )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_torch_compile_fullgraph_uses_compatible_fallback():
    (
        inputs,
        weights,
        weight_indices,
        seg_lens,
        slice_offsets,
        _,
        base_output,
    ) = _case(num_loras=4)
    expected = _reference(inputs, weights, weight_indices, slice_offsets, base_output)

    def run_compiled(inputs, weights, weight_indices, base_output):
        return sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            weight_indices,
            seg_lens,
            slice_offsets,
            base_output,
        )

    compiled = torch.compile(run_compiled, backend="eager", fullgraph=True)
    module_context = (
        patch.dict(
            sys.modules,
            {sgemm_lora_b_graph_fwd.__module__: _graph_lora_ops_module},
        )
        if _graph_lora_ops_module is not None
        else nullcontext()
    )
    with module_context, torch.inference_mode():
        actual = compiled(inputs, weights, weight_indices, base_output)

    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
