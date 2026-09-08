#!/usr/bin/env python3
"""Validate ParameterSet correlated throws against a known covariance matrix.

The test deliberately writes the covariance to a ROOT file, so GUNDAM follows
the same file-loading path used by production configurations. It compares the
empirical moments of GUNDAM throws with the true covariance moments.
"""

import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


N_THROWS = 6000
RANDOM_SEED = 12345
PRIORS = (1.20, -0.70, 3.40)
COVARIANCE = (
    (0.1600, 0.2420, -0.0250),
    (0.2420, 1.2100, 0.0825),
    (-0.0250, 0.0825, 0.0625),
)


@contextmanager
def suppress_cpp_output():
    """Silence the per-throw GUNDAM table while retaining normal test output."""
    saved_stdout = os.dup(sys.stdout.fileno())
    saved_stderr = os.dup(sys.stderr.fileno())
    with open(os.devnull, "w") as null_output:
        os.dup2(null_output.fileno(), sys.stdout.fileno())
        os.dup2(null_output.fileno(), sys.stderr.fileno())
        try:
            yield
        finally:
            os.dup2(saved_stdout, sys.stdout.fileno())
            os.dup2(saved_stderr, sys.stderr.fileno())
            os.close(saved_stdout)
            os.close(saved_stderr)


def write_covariance_file(path: Path) -> None:
    """Create a TMatrixD using ROOT batch mode (safe on headless CI workers)."""
    macro_path = path.with_name("make_covariance.C")
    matrix_rows = ",\n".join(
        "  {" + ", ".join(f"{value:.17g}" for value in row) + "}"
        for row in COVARIANCE
    )
    macro_path.write_text(
        f"""#include <TFile.h>
#include <TMatrixD.h>
#include <TTree.h>

void make_covariance() {{
  const double values[{len(COVARIANCE)}][{len(COVARIANCE)}] = {{
{matrix_rows}
  }};
  TFile output(\"{path}\", \"RECREATE\");
  TMatrixD covariance({len(COVARIANCE)}, {len(COVARIANCE)});
  for (int i = 0; i < {len(COVARIANCE)}; ++i) {{
    for (int j = 0; j < {len(COVARIANCE)}; ++j) covariance[i][j] = values[i][j];
  }}
  covariance.Write(\"covariance\");
  double x = 0.;
  TTree tree(\"tree_mc\", \"tree_mc\");
  tree.Branch(\"X\", &x);
  tree.Fill();
  tree.Write();
}}
"""
    )
    subprocess.run(["root", "-l", "-b", "-q", str(macro_path)], check=True)
    if not path.is_file():
        raise RuntimeError(f"ROOT did not create covariance file: {path}")


def build_config(covariance_path: Path) -> str:
    parameter_definitions = "\n".join(
        f"""              - parameterIndex: {index}
                name: par_{index}
                isEnabled: true
                isThrown: true
                priorValue: {prior}
                dialSetDefinitions:
                  - dialType: Normalization
                    applyOnDataSets: [ TestSample ]"""
        for index, prior in enumerate(PRIORS)
    )
    return f"""
fitterEngineConfig:
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: PoissonLLH
    dataSetList:
      - name: TestSample
        isEnabled: true
        model:
          tree: tree_mc
          filePathList: [ "{covariance_path}" ]
    propagatorConfig:
      sampleSetConfig:
        sampleList:
          - name: X
            isEnabled: true
            binning: {{ binningDefinition: [ {{ name: X, edges: [ -1, 1 ] }} ] }}
            dataSets: [ TestSample ]
      parametersManagerConfig:
        throwToyParametersWithGlobalCov: false
        parameterSetList:
          - name: covariance_throw_test
            isEnabled: true
            enableThrowToyParameters: true
            parameterDefinitionFilePath: "{covariance_path}"
            covarianceMatrix: covariance
            parameterDefinitions:
{parameter_definitions}
"""


def build_gundam_parameters(config_text: str, work_dir: Path):
    import GUNDAM

    GUNDAM.setRuntimeWorkingDirectory(str(work_dir))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)
    GUNDAM.FitterEngine.setRandomSeed(RANDOM_SEED)

    config_builder = GUNDAM.ConfigUtils.ConfigBuilder()
    config_builder.setConfigFromYamlString(config_text)
    config_reader = GUNDAM.ConfigUtils.ConfigReader(config_builder.getConfig())
    config_reader.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))

    engine = GUNDAM.FitterEngine()
    engine.setConfig(config_reader.fetchValueConfigReader("fitterEngineConfig"))
    engine.configure()
    likelihood_interface = engine.getLikelihoodInterface()
    likelihood_interface.initialize()
    propagator = likelihood_interface.getModelPropagator()
    parameter_manager = propagator.getParametersManager()
    parameter_sets = parameter_manager.getParameterSetsList()
    if len(parameter_sets) != 1:
        raise RuntimeError(f"Expected one parameter set, got {len(parameter_sets)}")
    # Keep the ParameterSet Python wrapper alive: ParameterList is a reference
    # internal to that wrapper in the pybind interface.
    parameter_set = parameter_sets[0]
    parameters = parameter_set.getParameterList()
    if len(parameters) != len(PRIORS):
        raise RuntimeError(f"Expected {len(PRIORS)} parameters, got {len(parameters)}")
    # The pybind API exposes these nested objects by reference.  Retain every
    # owner for the duration of the sampling loop.
    return engine, likelihood_interface, propagator, parameter_manager, parameter_set, parameters


def calculate_statistics(throws):
    size = len(PRIORS)
    count = len(throws)
    means = [sum(values[i] for values in throws) / count for i in range(size)]
    covariance = [
        [
            sum((values[i] - means[i]) * (values[j] - means[j]) for values in throws)
            / (count - 1)
            for j in range(size)
        ]
        for i in range(size)
    ]
    standard_deviations = [math.sqrt(covariance[i][i]) for i in range(size)]
    correlations = [
        [covariance[i][j] / (standard_deviations[i] * standard_deviations[j]) for j in range(size)]
        for i in range(size)
    ]
    return means, standard_deviations, correlations


def assert_close(label: str, actual: float, expected: float, tolerance: float) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{label}: got {actual:.6g}, expected {expected:.6g}, tolerance {tolerance:.6g}"
        )


def verify_statistics(label: str, statistics) -> None:
    means, standard_deviations, correlations = statistics
    expected_std_deviations = [math.sqrt(COVARIANCE[i][i]) for i in range(len(PRIORS))]
    expected_correlations = [
        [COVARIANCE[i][j] / (expected_std_deviations[i] * expected_std_deviations[j])
         for j in range(len(PRIORS))]
        for i in range(len(PRIORS))
    ]
    for i, prior in enumerate(PRIORS):
        assert_close(f"{label} mean[{i}]", means[i], prior, 0.08 * expected_std_deviations[i])
        assert_close(f"{label} std-dev[{i}]", standard_deviations[i], expected_std_deviations[i],
                     0.04 * expected_std_deviations[i])
        for j in range(len(PRIORS)):
            assert_close(f"{label} correlation[{i},{j}]", correlations[i][j],
                         expected_correlations[i][j], 0.06)


def print_summary_table(gundam_statistics) -> None:
    gundam_means, gundam_std_deviations, _ = gundam_statistics
    print()
    print(f"{'Quantity':<16} {'true':>12} {'GUNDAM':>12} {'diff':>12}")
    print("-" * 55)
    for index, prior in enumerate(PRIORS):
        true_std_deviation = math.sqrt(COVARIANCE[index][index])
        mean_relative_difference = 100.0 * abs(gundam_means[index] - prior) / abs(prior)
        std_relative_difference = 100.0 * abs(gundam_std_deviations[index] - true_std_deviation) / true_std_deviation
        print(f"{f'mean[{index}]':<16} {prior:>12.6f} {gundam_means[index]:>12.6f} {mean_relative_difference:>11.4f}%")
        print(f"{f'std-dev[{index}]':<16} {true_std_deviation:>12.6f} {gundam_std_deviations[index]:>12.6f} {std_relative_difference:>11.4f}%")


def main() -> int:
    work_dir = Path.cwd()
    covariance_path = work_dir / "213ParameterSetCovarianceThrows.root"
    write_covariance_file(covariance_path)

    (engine, likelihood_interface, propagator, parameter_manager, parameter_set, parameters) = \
        build_gundam_parameters(build_config(covariance_path), work_dir)
    for index, parameter in enumerate(parameters):
        assert_close(f"loaded prior[{index}]", parameter.getPriorValue(), PRIORS[index], 1e-12)
    loaded_covariance = parameter_set.getPriorCovarianceMatrix()
    for i, row in enumerate(COVARIANCE):
        for j, expected_value in enumerate(row):
            assert_close(f"loaded covariance[{i},{j}]", loaded_covariance[i, j], expected_value, 1e-12)

    gundam_throws = []
    import GUNDAM
    GUNDAM.flushOutput()
    # Initialization may consume random numbers; reset immediately before the
    # throw loop so the sampled sequence is reproducible.
    GUNDAM.FitterEngine.setRandomSeed(RANDOM_SEED)
    with suppress_cpp_output():
        for _ in range(N_THROWS):
            parameter_manager.throwParameters()
            throw = [parameter.getParameterValue() for parameter in parameters]
            if len(throw) != len(PRIORS):
                raise RuntimeError(f"GUNDAM returned {len(throw)} parameters during a throw")
            gundam_throws.append(throw)
        GUNDAM.flushOutput()

    gundam_statistics = calculate_statistics(gundam_throws)
    verify_statistics("GUNDAM", gundam_statistics)
    print(f"Number of throws: {len(gundam_throws)}")
    print_summary_table(gundam_statistics)

    print("SUCCESS: loaded covariance and GUNDAM throws agree on means, std-devs, and correlations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
