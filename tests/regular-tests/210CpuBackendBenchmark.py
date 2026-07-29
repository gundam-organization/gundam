#!/usr/bin/env python3

import math
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path


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


def build_config_text(root_path: Path, enable_backend: bool) -> str:
    backend_config = ""
    if enable_backend:
        backend_config = """
    backendConfig:
      isEnabled: true
      type: CPU
      outputRequests: [EventWeights, Histograms]
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


def compare_results(standard_results, backend_results) -> None:
    if len(standard_results) != len(backend_results):
        raise RuntimeError("Benchmark result count mismatch.")

    non_zero_llh_count = 0
    for i_point, (standard, backend) in enumerate(zip(standard_results, backend_results)):
        standard_llh, standard_sums = standard
        backend_llh, backend_sums = backend
        if not math.isclose(backend_llh, standard_llh, rel_tol=1e-10, abs_tol=1e-8):
            raise RuntimeError(f"LLH mismatch at point {i_point}: {backend_llh} != {standard_llh}")
        if not math.isclose(standard_llh, 0.0, rel_tol=0.0, abs_tol=1e-8):
            non_zero_llh_count += 1
        assert_close_list(f"histogram sums at point {i_point}", backend_sums, standard_sums)

    if non_zero_llh_count == 0:
        raise RuntimeError("All benchmark LLH values are zero.")


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

    standard_interface = build_likelihood_interface(build_config_text(root_path, False), work_dir)
    backend_interface = build_likelihood_interface(build_config_text(root_path, True), work_dir)

    standard_elapsed, standard_results = benchmark(standard_interface, parameter_points)
    backend_elapsed, backend_results = benchmark(backend_interface, parameter_points)
    compare_results(standard_results, backend_results)

    speedup = standard_elapsed / backend_elapsed if backend_elapsed > 0 else float("inf")
    print(f"Benchmark events: 120000")
    print(f"Benchmark parameter points: {len(parameter_points)}")
    print(f"Standard propagation time: {standard_elapsed:.6f} s")
    print(f"CPU backend propagation time: {backend_elapsed:.6f} s")
    print(f"CPU backend speedup: {speedup:.3f}x")
    print("SUCCESS: CPU backend benchmark matches standard propagation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
