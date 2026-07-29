#!/usr/bin/env python3

import importlib.util
import math
import platform
import sys
from pathlib import Path


def load_cpu_benchmark_module():
    benchmark_path = Path(__file__).resolve().with_name("210CpuBackendBenchmark.py")
    spec = importlib.util.spec_from_file_location("cpu_backend_benchmark", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_mps_config_text(
    cpu_benchmark,
    root_path: Path,
    output_requests: list[str] | None = None,
    materialize_output_requests: list[str] | None = None,
) -> str:
    config_text = cpu_benchmark.build_config_text(root_path, output_requests, materialize_output_requests)
    if output_requests is not None:
        config_text = config_text.replace("type: CPU", "type: MPS")
    return config_text


def make_parameter_points(nb_points: int) -> list[tuple[float, float]]:
    points = []
    for i_point in range(nb_points):
        positive = 1.35 + 1.85 * ((i_point * 17) % nb_points) / max(nb_points - 1, 1)
        negative = 1.75 + 2.15 * ((i_point * 29 + 7) % nb_points) / max(nb_points - 1, 1)
        points.append((positive, negative))
    return points


def assert_mps_close_list(label: str, left: list[float], right: list[float]) -> None:
    if len(left) != len(right):
        raise RuntimeError(f"{label}: length mismatch {len(left)} != {len(right)}")

    for i_bin, (left_value, right_value) in enumerate(zip(left, right)):
        if not math.isclose(left_value, right_value, rel_tol=5e-5, abs_tol=5e-3):
            raise RuntimeError(f"{label}: bin {i_bin} mismatch {left_value} != {right_value}")


def compare_mps_results(case_name: str, requested_outputs: list[str], standard_results, backend_results) -> None:
    if len(standard_results) != len(backend_results):
        raise RuntimeError(f"{case_name}: benchmark result count mismatch.")

    non_zero_llh_count = 0
    check_likelihood = "Likelihood" in requested_outputs
    check_histograms = (
        "Histograms" in requested_outputs
        or "EventWeights" in requested_outputs
        or not check_likelihood
    )
    for i_point, (standard, backend) in enumerate(zip(standard_results, backend_results)):
        standard_llh, standard_sums = standard
        backend_llh, backend_sums = backend
        if check_histograms:
            assert_mps_close_list(f"{case_name}: histogram sums at point {i_point}", backend_sums, standard_sums)

        if not math.isclose(standard_llh, 0.0, rel_tol=0.0, abs_tol=1e-8):
            non_zero_llh_count += 1

        if check_likelihood and not math.isclose(backend_llh, standard_llh, rel_tol=5e-5, abs_tol=5e-3):
            raise RuntimeError(f"{case_name}: LLH mismatch at point {i_point}: {backend_llh} != {standard_llh}")

    if non_zero_llh_count == 0:
        raise RuntimeError(f"{case_name}: all benchmark LLH values are zero.")


def main() -> int:
    if platform.system() != "Darwin":
        print("SKIP: MPS backend benchmark requires macOS/Metal.")
        return 0

    cpu_benchmark = load_cpu_benchmark_module()

    work_dir = Path.cwd()
    root_path = work_dir / "211MpsBackendBenchmark.root"
    nb_events = 120000
    nb_parameter_points = 100

    cpu_benchmark.flush_python_outputs()
    cpu_benchmark.write_input_root_file(root_path, nb_events=nb_events)
    parameter_points = make_parameter_points(nb_parameter_points)

    benchmark_cases = [
        cpu_benchmark.BenchmarkCase("mps histograms", ["Histograms"]),
        cpu_benchmark.BenchmarkCase("mps event weights", ["EventWeights"]),
        cpu_benchmark.BenchmarkCase("mps weights device-only", ["EventWeights", "Histograms"], ["Histograms"]),
        cpu_benchmark.BenchmarkCase("mps likelihood only", ["Likelihood"]),
        cpu_benchmark.BenchmarkCase("mps weights+hist", ["EventWeights", "Histograms"]),
        cpu_benchmark.BenchmarkCase("mps weights+hist no weights copy", ["EventWeights", "Histograms"], ["Histograms"]),
        cpu_benchmark.BenchmarkCase("mps hist+llh", ["Histograms", "Likelihood"]),
    ]

    standard_interface = cpu_benchmark.build_likelihood_interface(
        build_mps_config_text(cpu_benchmark, root_path),
        work_dir,
    )
    standard_elapsed, standard_results = cpu_benchmark.benchmark(standard_interface, parameter_points)

    timings = [
        cpu_benchmark.BenchmarkTiming(
            name="standard",
            output_requests=[],
            elapsed=standard_elapsed,
            speedup=1.0,
        )
    ]

    for benchmark_case in benchmark_cases:
        backend_interface = cpu_benchmark.build_likelihood_interface(
            build_mps_config_text(
                cpu_benchmark,
                root_path,
                benchmark_case.output_requests,
                benchmark_case.materialize_output_requests,
            ),
            work_dir,
        )
        backend_elapsed, backend_results = cpu_benchmark.benchmark(backend_interface, parameter_points)
        compare_mps_results(benchmark_case.name, benchmark_case.output_requests, standard_results, backend_results)
        timings.append(
            cpu_benchmark.BenchmarkTiming(
                name=benchmark_case.name,
                output_requests=benchmark_case.output_requests,
                elapsed=backend_elapsed,
                speedup=standard_elapsed / backend_elapsed if backend_elapsed > 0 else float("inf"),
            )
        )

    print(f"Benchmark events: {nb_events}")
    print(f"Benchmark parameter points: {len(parameter_points)}")
    cpu_benchmark.print_timing_table(timings)
    print("SUCCESS: MPS backend benchmark cases match standard propagation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
