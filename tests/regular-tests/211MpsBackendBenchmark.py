#!/usr/bin/env python3

import importlib.util
import platform
import sys
from pathlib import Path


def load_cpu_benchmark_module():
    benchmark_path = Path(__file__).resolve().with_name("210CpuBackendBenchmark.py")
    spec = importlib.util.spec_from_file_location("cpu_backend_benchmark", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    parameter_points = cpu_benchmark.make_parameter_points(nb_parameter_points)

    standard_context = cpu_benchmark.build_likelihood_interface(
        cpu_benchmark.build_config_text(root_path),
        work_dir,
    )
    backend_context = cpu_benchmark.build_likelihood_interface(
        cpu_benchmark.build_config_text(root_path, backend_type="MPS"),
        work_dir,
    )

    standard_elapsed, standard_results = cpu_benchmark.benchmark(standard_context, parameter_points)
    backend_elapsed, backend_results = cpu_benchmark.benchmark(backend_context, parameter_points)

    cpu_benchmark.compare_results(
        standard_results,
        backend_results,
        rel_tol=5e-5,
        abs_tol=5e-3,
    )

    print(f"Benchmark events: {nb_events}")
    print(f"Benchmark parameter points: {len(parameter_points)}")
    print(f"Standard propagation time: {standard_elapsed:.6f} s")
    print(f"MPS backend propagation time: {backend_elapsed:.6f} s")
    print(f"MPS backend speedup: {standard_elapsed / backend_elapsed:.3f}x")
    print("SUCCESS: MPS backend propagation matches the standard propagation path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
