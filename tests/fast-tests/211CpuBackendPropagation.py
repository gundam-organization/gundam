#!/usr/bin/env python3

import math
import sys
from array import array
from pathlib import Path
from typing import Optional


def flush_python_outputs() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


def write_input_root_file(root_path: Path) -> None:
    import uproot

    with uproot.recreate(root_path) as root_file:
        tree = root_file.mktree(
            "tree_mc",
            {
                "X": "float64",
                "C": "float64",
            },
        )
        tree.extend(
            {
                "X": array("d", [-0.5, -0.4, -0.3, 0.2, 0.3, 0.4]),
                "C": array("d", [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
            }
        )


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
      - name: "BackendSample"
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
            binning: {{ binningDefinition: [{{ name: "X", edges: [-1, 0, 1] }}] }}
            dataSets: [ "BackendSample" ]

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


def evaluate_config(
    config_text: str,
    work_dir: Path,
    systematic_point: Optional[tuple[float, float]] = None,
) -> tuple[float, list[float], list[float]]:
    repo_root = Path(__file__).resolve().parents[2]
    for build_dir in ("cmake-build-debug", "cmake-build-release"):
        python_module_dir = repo_root / build_dir / "src" / "PythonInterface"
        if python_module_dir.exists():
            sys.path.insert(0, str(python_module_dir))
            break

    import GUNDAM

    flush_python_outputs()

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

    if systematic_point is not None:
        set_normalization_parameters(likelihood_interface, *systematic_point)

    likelihood_interface.propagateAndEvalLikelihood()

    likelihood = likelihood_interface.getLastLikelihood()
    sample = likelihood_interface.getModelPropagator().getSampleSet().getSampleList()[0]
    bin_content_list = sample.getHistogram().getBinContentList()

    sums = [bin_content.sumWeights for bin_content in bin_content_list]
    errors = [bin_content.sqrtSumSqWeights for bin_content in bin_content_list]
    return likelihood, sums, errors


def assert_close_list(label: str, left: list[float], right: list[float]) -> None:
    if len(left) != len(right):
        raise RuntimeError(f"{label}: length mismatch {len(left)} != {len(right)}")

    for i, (left_value, right_value) in enumerate(zip(left, right)):
        if not math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"{label}: bin {i} mismatch {left_value} != {right_value}")


def main() -> int:
    script_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    work_dir = Path.cwd()
    root_path = work_dir / "211CpuBackendPropagation.root"

    write_input_root_file(root_path)

    prior_standard_llh, prior_standard_sums, prior_standard_errors = evaluate_config(
        build_config_text(root_path, False),
        work_dir,
    )
    prior_backend_llh, prior_backend_sums, prior_backend_errors = evaluate_config(
        build_config_text(root_path, True),
        work_dir,
    )

    systematic_point = (4.0, 1.0)
    standard_llh, standard_sums, standard_errors = evaluate_config(
        build_config_text(root_path, False),
        work_dir,
        systematic_point,
    )
    backend_llh, backend_sums, backend_errors = evaluate_config(
        build_config_text(root_path, True),
        work_dir,
        systematic_point,
    )

    expected_prior_sums = [7.0, 8.0]
    expected_shifted_sums = [9.0, 6.0]
    assert_close_list("standard prior sums", prior_standard_sums, expected_prior_sums)
    assert_close_list("backend prior sums", prior_backend_sums, prior_standard_sums)
    assert_close_list("backend prior errors", prior_backend_errors, prior_standard_errors)

    assert_close_list("standard shifted sums", standard_sums, expected_shifted_sums)
    assert_close_list("backend shifted sums", backend_sums, standard_sums)
    assert_close_list("backend shifted errors", backend_errors, standard_errors)

    if not math.isclose(prior_backend_llh, prior_standard_llh, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"Prior LLH mismatch {prior_backend_llh} != {prior_standard_llh}")

    if not math.isclose(backend_llh, standard_llh, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"Shifted LLH mismatch {backend_llh} != {standard_llh}")

    if math.isclose(standard_llh, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Shifted LLH is unexpectedly zero.")

    print("Prior standard bin sums:", prior_standard_sums)
    print("Prior backend CPU bin sums:", prior_backend_sums)
    print("Shifted standard bin sums:", standard_sums)
    print("Shifted backend CPU bin sums:", backend_sums)
    print("Prior standard LLH:", prior_standard_llh)
    print("Prior backend CPU LLH:", prior_backend_llh)
    print("Shifted standard LLH:", standard_llh)
    print("Shifted backend CPU LLH:", backend_llh)
    print("SUCCESS: CPU backend propagation matches the standard propagation path at prior and shifted parameters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
