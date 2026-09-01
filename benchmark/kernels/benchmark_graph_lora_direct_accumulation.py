"""Benchmark direct accumulation for the torch-native graph LoRA-B path.

Examples:
    python3 benchmark/kernels/benchmark_graph_lora_direct_accumulation.py
    python3 benchmark/kernels/benchmark_graph_lora_direct_accumulation.py \
        --representative-matrix --repetitions 100 --output results.csv
    python3 benchmark/kernels/benchmark_graph_lora_direct_accumulation.py \
        --full-matrix --repetitions 100 --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch


@dataclass(frozen=True)
class Config:
    tokens: int
    rank: int
    output_width: int
    adapters: int
    dtype: str
    layout: str
    execution: str


@dataclass(frozen=True)
class Result:
    tokens: int
    rank: int
    output_width: int
    adapters: int
    dtype: str
    layout: str
    execution: str
    baseline_p10_us: float
    baseline_p50_us: float
    baseline_p90_us: float
    optimized_p10_us: float
    optimized_p50_us: float
    optimized_p90_us: float
    speedup: float
    baseline_extra_peak_bytes: int
    optimized_extra_peak_bytes: int


@dataclass
class Operands:
    inputs: torch.Tensor
    weights: torch.Tensor
    weight_indices: torch.Tensor
    base_backing: torch.Tensor
    baseline_backing: torch.Tensor
    optimized_backing: torch.Tensor
    output_slice: slice


def baseline(
    output_slice: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
) -> None:
    for adapter_idx in range(weights.shape[0]):
        masked_inputs = torch.where(
            (weight_indices == adapter_idx).unsqueeze(1), inputs, 0
        )
        output_slice.add_(torch.mm(masked_inputs, weights[adapter_idx].t()))


def optimized(
    output_slice: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
) -> None:
    if weights.shape[0] < 4:
        baseline(output_slice, inputs, weights, weight_indices)
        return
    for adapter_idx in range(weights.shape[0]):
        masked_inputs = torch.where(
            (weight_indices == adapter_idx).unsqueeze(1), inputs, 0
        )
        torch.addmm(
            output_slice,
            masked_inputs,
            weights[adapter_idx].t(),
            out=output_slice,
        )


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _make_operands(config: Config) -> Operands:
    device = torch.device("cuda")
    dtype = _dtype(config.dtype)
    padding = 0 if config.layout == "whole" else 64
    backing_width = config.output_width + 2 * padding
    output_slice = slice(padding, padding + config.output_width)

    generator = torch.Generator(device=device).manual_seed(20260901)
    inputs = torch.randn(
        config.tokens,
        config.rank,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    weights = torch.randn(
        config.adapters,
        config.output_width,
        config.rank,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    weight_indices = torch.arange(
        config.tokens, dtype=torch.int32, device=device
    ).remainder(config.adapters + 1)
    weight_indices[weight_indices == config.adapters] = -1
    base_backing = torch.randn(
        config.tokens,
        backing_width,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return Operands(
        inputs=inputs,
        weights=weights,
        weight_indices=weight_indices,
        base_backing=base_backing,
        baseline_backing=base_backing.clone(),
        optimized_backing=base_backing.clone(),
        output_slice=output_slice,
    )


def _check_correctness(operands: Operands) -> None:
    operands.baseline_backing.copy_(operands.base_backing)
    operands.optimized_backing.copy_(operands.base_backing)
    baseline(
        operands.baseline_backing[:, operands.output_slice],
        operands.inputs,
        operands.weights,
        operands.weight_indices,
    )
    optimized(
        operands.optimized_backing[:, operands.output_slice],
        operands.inputs,
        operands.weights,
        operands.weight_indices,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        operands.optimized_backing,
        operands.baseline_backing,
        rtol=3e-2,
        atol=3e-2,
    )


def _percentiles(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)

    def nearest(percentile: float) -> float:
        index = round(percentile * (len(ordered) - 1))
        return ordered[index]

    return nearest(0.10), nearest(0.50), nearest(0.90)


def _time_eager(
    operation: Callable,
    output_backing: torch.Tensor,
    operands: Operands,
    warmups: int,
    repetitions: int,
) -> list[float]:
    output = output_backing[:, operands.output_slice]
    for _ in range(warmups):
        output_backing.copy_(operands.base_backing)
        operation(output, operands.inputs, operands.weights, operands.weight_indices)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repetitions):
        output_backing.copy_(operands.base_backing)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation(output, operands.inputs, operands.weights, operands.weight_indices)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def _capture_graph(
    operation: Callable,
    output_backing: torch.Tensor,
    operands: Operands,
) -> torch.cuda.CUDAGraph:
    output = output_backing[:, operands.output_slice]
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        output_backing.copy_(operands.base_backing)
        operation(output, operands.inputs, operands.weights, operands.weight_indices)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output_backing.copy_(operands.base_backing)
        operation(output, operands.inputs, operands.weights, operands.weight_indices)
    return graph


def _time_graph(
    operation: Callable,
    output_backing: torch.Tensor,
    operands: Operands,
    warmups: int,
    repetitions: int,
) -> list[float]:
    graph = _capture_graph(operation, output_backing, operands)
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def _extra_peak_bytes(
    operation: Callable,
    output_backing: torch.Tensor,
    operands: Operands,
) -> int:
    output = output_backing[:, operands.output_slice]
    output_backing.copy_(operands.base_backing)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    operation(output, operands.inputs, operands.weights, operands.weight_indices)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before


def _profile_once(operation: Callable, operands: Operands) -> str:
    operands.optimized_backing.copy_(operands.base_backing)
    output = operands.optimized_backing[:, operands.output_slice]
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        operation(output, operands.inputs, operands.weights, operands.weight_indices)
    torch.cuda.synchronize()
    return profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)


def run_config(
    config: Config,
    warmups: int,
    repetitions: int,
    profile: bool,
) -> Result:
    operands = _make_operands(config)
    with torch.inference_mode():
        _check_correctness(operands)
        timer = _time_eager if config.execution == "eager" else _time_graph
        baseline_samples = timer(
            baseline,
            operands.baseline_backing,
            operands,
            warmups,
            repetitions,
        )
        baseline_peak = _extra_peak_bytes(baseline, operands.baseline_backing, operands)
        if config.adapters < 4:
            optimized_samples = baseline_samples.copy()
            optimized_peak = baseline_peak
        else:
            optimized_samples = timer(
                optimized,
                operands.optimized_backing,
                operands,
                warmups,
                repetitions,
            )
            optimized_peak = _extra_peak_bytes(
                optimized, operands.optimized_backing, operands
            )
        if profile:
            print("\nBaseline operators:\n" + _profile_once(baseline, operands))
            print("\nOptimized operators:\n" + _profile_once(optimized, operands))

    baseline_p10, baseline_p50, baseline_p90 = _percentiles(baseline_samples)
    optimized_p10, optimized_p50, optimized_p90 = _percentiles(optimized_samples)
    return Result(
        **asdict(config),
        baseline_p10_us=baseline_p10,
        baseline_p50_us=baseline_p50,
        baseline_p90_us=baseline_p90,
        optimized_p10_us=optimized_p10,
        optimized_p50_us=optimized_p50,
        optimized_p90_us=optimized_p90,
        speedup=baseline_p50 / optimized_p50,
        baseline_extra_peak_bytes=baseline_peak,
        optimized_extra_peak_bytes=optimized_peak,
    )


def _cartesian_configs(args: argparse.Namespace) -> Iterable[Config]:
    for tokens in args.tokens:
        for rank in args.rank:
            for output_width in args.output_width:
                for adapters in args.adapters:
                    for dtype in args.dtype:
                        for layout in args.layout:
                            for execution in args.execution:
                                yield Config(
                                    tokens,
                                    rank,
                                    output_width,
                                    adapters,
                                    dtype,
                                    layout,
                                    execution,
                                )


def _representative_configs() -> Iterable[Config]:
    shapes = (
        (1, 8, 4096, 1),
        (8, 16, 4096, 2),
        (32, 16, 12288, 4),
        (128, 32, 4096, 4),
        (512, 32, 11008, 8),
        (2048, 64, 4096, 8),
    )
    for tokens, rank, width, adapters in shapes:
        for dtype in ("float16", "bfloat16"):
            for layout in ("whole", "packed"):
                for execution in ("eager", "cudagraph"):
                    yield Config(
                        tokens, rank, width, adapters, dtype, layout, execution
                    )


def _full_configs() -> Iterable[Config]:
    for tokens in (1, 8, 32, 128, 512, 2048):
        for rank in (8, 16, 32, 64):
            for width in (4096, 11008, 12288):
                for adapters in (1, 2, 4, 8):
                    for dtype in ("float16", "bfloat16"):
                        for layout in ("whole", "packed"):
                            for execution in ("eager", "cudagraph"):
                                yield Config(
                                    tokens,
                                    rank,
                                    width,
                                    adapters,
                                    dtype,
                                    layout,
                                    execution,
                                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32])
    parser.add_argument("--rank", type=int, nargs="+", default=[16])
    parser.add_argument("--output-width", type=int, nargs="+", default=[4096])
    parser.add_argument("--adapters", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--dtype", nargs="+", choices=("float16", "bfloat16"), default=["float16"]
    )
    parser.add_argument(
        "--layout", nargs="+", choices=("whole", "packed"), default=["packed"]
    )
    parser.add_argument(
        "--execution",
        nargs="+",
        choices=("eager", "cudagraph"),
        default=["eager"],
    )
    matrix = parser.add_mutually_exclusive_group()
    matrix.add_argument("--representative-matrix", action="store_true")
    matrix.add_argument("--full-matrix", action="store_true")
    parser.add_argument("--warmups", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmups < 1 or args.repetitions < 1:
        raise ValueError("warmups and repetitions must be positive")

    if args.full_matrix:
        configs = _full_configs()
    elif args.representative_matrix:
        configs = _representative_configs()
    else:
        configs = _cartesian_configs(args)

    results = []
    for index, config in enumerate(configs):
        result = run_config(
            config,
            warmups=args.warmups,
            repetitions=args.repetitions,
            profile=args.profile and index == 0,
        )
        results.append(result)
        print(
            f"{config} p50={result.optimized_p50_us:.3f}us "
            f"speedup={result.speedup:.3f}x "
            f"peak={result.baseline_extra_peak_bytes}->"
            f"{result.optimized_extra_peak_bytes} bytes"
        )

    geometric_mean = math.exp(statistics.fmean(math.log(r.speedup) for r in results))
    print(f"geometric mean p50 speedup: {geometric_mean:.3f}x")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)


if __name__ == "__main__":
    main()
