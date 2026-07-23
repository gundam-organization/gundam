#!/usr/bin/env python3

import math
import sys
from pathlib import Path


def add_python_module_paths(repo_root: Path) -> None:
    candidates = [
        repo_root / "cmake-build-debug" / "src" / "PythonInterface",
        repo_root / "cmake-build-release" / "src" / "PythonInterface",
    ]
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

def write_input_root_file(root_path: Path) -> None:
    import uproot
    from array import array

    with uproot.recreate(root_path) as root_file:
        tree = root_file.mktree("tree_mc", {"X": "float64"})
        tree.extend({"X": array("d", [-0.5] * 10)})


def write_binning_file(binning_path: Path) -> None:
    binning_path.write_text(
        "variables: X X\n"
        "-1 0\n"
        "0 1\n",
        encoding="ascii",
    )


def build_config_text(root_path: Path, binning_path: Path, ignore_zero_bins: bool) -> str:
    bool_str = "true" if ignore_zero_bins else "false"
    return f"""
fitterEngineConfig:
  minimizerConfig:
    type: RootMinimizer
    minimizer: "Minuit2"
    algorithm: "Migrad"
    errors: "Hesse"
    print_level: 0
    tolerance: 1E-6

  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: PoissonLLH
      ignoreBinsWithZeroPredictionAtPrior: {bool_str}

    dataSetList:
      - name: "TestSample"
        isEnabled: true
        model:
          tree: tree_mc
          selectionCutFormula: "(1)"
          nominalTreeWeightFormula: "(1.0)"
          filePathList:
            - "{root_path}"

    propagatorConfig:
      sampleSetConfig:
        sampleList:
          - name: X
            isEnabled: true
            binning: "{binning_path}"
            dataSets: [ "TestSample" ]
"""


def evaluate_config(config_path: Path) -> float:
    import GUNDAM

    GUNDAM.setRuntimeWorkingDirectory(str(config_path.parent))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)

    config_builder = GUNDAM.ConfigUtils.ConfigBuilder(str(config_path))
    config_reader = GUNDAM.ConfigUtils.ConfigReader(config_builder.getConfig())
    config_reader.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))
    fitter_engine_config = config_reader.fetchValueConfigReader("fitterEngineConfig")

    engine = GUNDAM.FitterEngine()
    engine.setConfig(fitter_engine_config)
    engine.configure()
    likelihood_interface = engine.getLikelihoodInterface()
    likelihood_interface.initialize()
    likelihood_interface.propagateAndEvalLikelihood()

    likelihood = likelihood_interface.getLastLikelihood()

    sample = likelihood_interface.getModelPropagator().getSampleSet().getSampleList()[0]
    bin_content_list = sample.getHistogram().getBinContentList()

    if len(bin_content_list) != 2:
        raise RuntimeError(f"Expected 2 bins, got {len(bin_content_list)}")
    if bin_content_list[0].sumWeights <= 0.0:
        raise RuntimeError("First bin should be populated.")
    if bin_content_list[1].sumWeights != 0.0:
        raise RuntimeError(f"Second bin should be empty, got {bin_content_list[1].sumWeights}")

    return likelihood


def main() -> int:
    script_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    add_python_module_paths(repo_root)

    work_dir = Path.cwd()
    root_path = work_dir / "210IgnoreZeroPredictionAtPrior.root"
    binning_path = work_dir / "210IgnoreZeroPredictionAtPrior-binning.txt"
    config_false_path = work_dir / "210IgnoreZeroPredictionAtPrior-false.yaml"
    config_true_path = work_dir / "210IgnoreZeroPredictionAtPrior-true.yaml"

    write_input_root_file(root_path)
    write_binning_file(binning_path)
    config_false_path.write_text(build_config_text(root_path, binning_path, False), encoding="ascii")
    config_true_path.write_text(build_config_text(root_path, binning_path, True), encoding="ascii")

    llh_without_ignore = evaluate_config(config_false_path)
    llh_with_ignore = evaluate_config(config_true_path)

    if not math.isinf(llh_without_ignore):
        print(f"FAIL: expected infinite LLH without ignore flag, got {llh_without_ignore}")
        return 1

    if not math.isfinite(llh_with_ignore):
        print(f"FAIL: expected finite LLH with ignore flag, got {llh_with_ignore}")
        return 1

    print(f"LLH without ignore flag: {llh_without_ignore}")
    print(f"LLH with ignore flag: {llh_with_ignore}")
    print("SUCCESS: ignoreBinsWithZeroPredictionAtPrior keeps the Poisson LLH finite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
