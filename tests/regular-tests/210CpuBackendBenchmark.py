#!/usr/bin/env python3

import math
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def flush_python_outputs() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


def write_input_root_file(root_path: Path, nb_events: int) -> None:
    import uproot

    x_values = array("d")
    c_values = array("d")
    for i_event in range(nb_events):
        x_values.append(-0.95 + 1.9 * ((i_event * 37) % nb_events) / max(nb_events - 1, 1))
        c_values.append(1.0 if i_event % 2 == 0 else -1.0)

    with uproot.recreate(root_path) as root_file:
        tree = root_file.mktree(
            "tree_mc",
            {
                "X": "float64",
                "C": "float64",
            },
        )
        tree.extend({"X": x_values, "C": c_values})


def load_gundam_module():
    repo_root = Path(__file__).resolve().parents[2]
    for build_dir in ("cmake-build-debug", "cmake-build-release"):
        python_module_dir = repo_root / build_dir / "src" / "PythonInterface"
        if python_module_dir.exists():
            sys.path.insert(0, str(python_module_dir))
            break

    import GUNDAM

    return GUNDAM


@dataclass
class BenchmarkContext:
    engine: object
    likelihood_interface: object


@dataclass
class BenchmarkCase:
    name: str
    output_requests: list[str]
    materialize_output_requests: Optional[list[str]] = None


@dataclass
class BenchmarkTiming:
    name: str
    output_requests: list[str]
    elapsed: float
    speedup: float


def build_config_text(
    root_path: Path,
    output_requests: Optional[list[str]] = None,
    materialize_output_requests: Optional[list[str]] = None,
) -> str:
    backend_config = ""
    if output_requests is not None:
        materialize_config = ""
        if materialize_output_requests is not None:
            materialize_config = f"""
      materializeOutputRequests: [{", ".join(materialize_output_requests)}]"""
        backend_config = f"""
    backendConfig:
      isEnabled: true
      type: CPU
      outputRequests: [{", ".join(output_requests)}]{materialize_config}
"""

    return f"""
fitterEngineConfig:

  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: LeastSquares
      lsqPoissonianApproximation: true

    dataSetList:
      - name: "BackendBenchmarkSample"
        isEnabled: true
        model:
          tree: tree_mc
          selectionCutFormula: "(1)"
          nominalWeightFormula: "(1.0)"
          filePathList:
            - "{root_path}"
{backend_config}
    propagatorConfig:
      sampleSetConfig:
        sampleList:
          - name: X
            isEnabled: true
            binning: {{ binningDefinition: [{{ name: "X", edges: [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0] }}] }}
            dataSets: [ "BackendBenchmarkSample" ]

      parametersManagerConfig:
        parameterSetList:
          - name: Normalizations
            isEnabled: true
            nominalStepSize: 0.1

            parameterDefinitions:
              - parameterName: "Positive_C"
                isEnabled: true
                priorValue: 2.0
                priorType: Flat
              - parameterName: "Negative_C"
                isEnabled: true
                priorValue: 3.0
                priorType: Flat

            dialSetDefinitions:
              - dialType: Normalization
                applyCondition: "[C] > 0"
                dialInputList:
                  - name: "Positive_C"
              - dialType: Normalization
                applyCondition: "[C] <= 0"
                dialInputList:
                  - name: "Negative_C"
"""


def build_likelihood_interface(config_text: str, work_dir: Path):
    GUNDAM = load_gundam_module()

    GUNDAM.setRuntimeWorkingDirectory(str(work_dir))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)

    config_builder = GUNDAM.ConfigUtils.ConfigBuilder()
    config_builder.setConfigFromYamlString(config_text)
    config_reader = GUNDAM.ConfigUtils.ConfigReader(config_builder.getConfig())
    config_reader.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))
    fitter_engine_config = config_reader.fetchValueConfigReader("fitterEngineConfig")

    engine = GUNDAM.FitterEngine()
    engine.setConfig(fitter_engine_config)
    engine.configure()
    likelihood_interface = engine.getLikelihoodInterface()
    likelihood_interface.initialize()
    GUNDAM.flushOutput()

    return BenchmarkContext(engine=engine, likelihood_interface=likelihood_interface)


def set_normalization_parameters(likelihood_interface, positive_value: float, negative_value: float) -> None:
    parameter_set = (
        likelihood_interface
        .getModelPropagator()
        .getParametersManager()
        .getParameterSetsList()[0]
    )
    parameter_list = parameter_set.getParameterList()
    parameter_list[0].setParameterValue(positive_value)
    parameter_list[1].setParameterValue(negative_value)


def get_histogram_sums(likelihood_interface) -> list[float]:
    sample = likelihood_interface.getModelPropagator().getSampleSet().getSampleList()[0]
    return [bin_content.sumWeights for bin_content in sample.getHistogram().getBinContentList()]


def benchmark(context: BenchmarkContext, parameter_points: list[tuple[float, float]]) -> tuple[float, list[tuple[float, list[float]]]]:
    likelihood_interface = context.likelihood_interface
    results = []
    start = time.perf_counter()
    for point in parameter_points:
        set_normalization_parameters(likelihood_interface, *point)
        likelihood_interface.propagateAndEvalLikelihood()
        results.append((likelihood_interface.getLastLikelihood(), get_histogram_sums(likelihood_interface)))
    elapsed = time.perf_counter() - start
    return elapsed, results


def assert_close_list(label: str, left: list[float], right: list[float]) -> None:
    if len(left) != len(right):
        raise RuntimeError(f"{label}: length mismatch {len(left)} != {len(right)}")

    for i_bin, (left_value, right_value) in enumerate(zip(left, right)):
        if not math.isclose(left_value, right_value, rel_tol=1e-10, abs_tol=1e-8):
            raise RuntimeError(f"{label}: bin {i_bin} mismatch {left_value} != {right_value}")


def compare_results(case_name: str, requested_outputs: list[str], standard_results, backend_results) -> None:
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
        if check_likelihood and not math.isclose(backend_llh, standard_llh, rel_tol=1e-10, abs_tol=1e-8):
            raise RuntimeError(f"{case_name}: LLH mismatch at point {i_point}: {backend_llh} != {standard_llh}")
        if not math.isclose(standard_llh, 0.0, rel_tol=0.0, abs_tol=1e-8):
            non_zero_llh_count += 1
        if check_histograms:
            assert_close_list(f"{case_name}: histogram sums at point {i_point}", backend_sums, standard_sums)

    if non_zero_llh_count == 0:
        raise RuntimeError(f"{case_name}: all benchmark LLH values are zero.")


def print_timing_table(timings: list[BenchmarkTiming]) -> None:
    rows = [
        (
            timing.name,
            ", ".join(timing.output_requests) if timing.output_requests else "standard",
            f"{timing.elapsed:.6f}",
            "1.000" if timing.speedup == 1.0 else f"{timing.speedup:.3f}",
        )
        for timing in timings
    ]
    headers = ("Case", "Requested outputs", "Time [s]", "Speedup")
    widths = [
        max(len(headers[i_col]), *(len(row[i_col]) for row in rows))
        for i_col in range(len(headers))
    ]

    def format_row(row) -> str:
        return " | ".join(str(value).ljust(widths[i_col]) for i_col, value in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def main() -> int:
    script_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    work_dir = Path.cwd()
    root_path = work_dir / "210CpuBackendBenchmark.root"

    flush_python_outputs()
    write_input_root_file(root_path, nb_events=120000)

    parameter_points = [
        (2.0, 3.0),
        (2.3, 2.6),
        (1.7, 3.5),
        (2.8, 2.1),
        (1.4, 3.8),
        (3.1, 1.9),
        (2.5, 2.4),
        (1.8, 3.2),
        (2.9, 2.0),
        (1.6, 3.6),
    ]

    benchmark_cases = [
        BenchmarkCase("backend histograms", ["Histograms"]),
        BenchmarkCase("backend event weights", ["EventWeights"]),
        BenchmarkCase("backend likelihood", ["Likelihood"]),
        BenchmarkCase("backend weights+hist", ["EventWeights", "Histograms"]),
        BenchmarkCase("backend hist+llh", ["Histograms", "Likelihood"]),
    ]

    standard_interface = build_likelihood_interface(build_config_text(root_path), work_dir)
    standard_elapsed, standard_results = benchmark(standard_interface, parameter_points)

    timings = [
        BenchmarkTiming(
            name="standard",
            output_requests=[],
            elapsed=standard_elapsed,
            speedup=1.0,
        )
    ]

    for benchmark_case in benchmark_cases:
        backend_interface = build_likelihood_interface(
            build_config_text(root_path, benchmark_case.output_requests),
            work_dir,
        )
        backend_elapsed, backend_results = benchmark(backend_interface, parameter_points)
        compare_results(benchmark_case.name, benchmark_case.output_requests, standard_results, backend_results)
        timings.append(
            BenchmarkTiming(
                name=benchmark_case.name,
                output_requests=benchmark_case.output_requests,
                elapsed=backend_elapsed,
                speedup=standard_elapsed / backend_elapsed if backend_elapsed > 0 else float("inf"),
            )
        )

    print(f"Benchmark events: 120000")
    print(f"Benchmark parameter points: {len(parameter_points)}")
    print_timing_table(timings)
    print("SUCCESS: CPU backend benchmark cases match standard propagation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
