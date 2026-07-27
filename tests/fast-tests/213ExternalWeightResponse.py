#!/usr/bin/env python3

import json
import math
import sys
from array import array
from pathlib import Path


def flush_python_outputs() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


def write_input_root_file(root_path: Path) -> None:
    import uproot

    with uproot.recreate(root_path) as root_file:
        tree = root_file.mktree("tree_mc", {"X": "float64"})
        tree.extend({"X": array("d", [0.5] * 10)})


def write_external_weight_script(script_path: Path) -> None:
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "",
                "with open(sys.argv[1], 'r', encoding='utf-8') as payload_file:",
                "    payload = json.load(payload_file)",
                "",
                "parameters = {entry['name']: entry['value'] for entry in payload['parameters']}",
                "delta_msq = parameters['deltaMsq']",
                "sin_sq_theta = parameters['sinSqTheta']",
                "x_values = payload['inputs']['X']",
                "weights = [1.0 + sin_sq_theta * delta_msq * x for x in x_values]",
                "print(json.dumps({'weights': weights}))",
                "",
            ]
        ),
        encoding="ascii",
    )


def build_config_text(root_path: Path, eval_script_path: Path, python_executable: str) -> str:
    return """
fitterEngineConfig:
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: PoissonLLH
      ignoreBinsWithZeroPredictionAtPrior: true

    dataSetList:
      - name: "TestSample"
        isEnabled: true
        model:
          tree: tree_mc
          filePathList:
            - "{root_path}"

    propagatorConfig:
      sampleSetConfig:
        sampleList:
          - name: X
            isEnabled: true
            binning:
              binningDefinition:
                - name: "X"
                  edges: [0, 1]
            dataSets: [ "TestSample" ]

      parametersManagerConfig:
        parameterSetList:
          - name: "ExternalWeightParameters"
            isEnabled: true
            nominalStepSize: 0.1
            dialSetDefinitions:
              - dialType: ExternalWeight
                applyOnDataSets: [ "TestSample" ]
                dialInputList:
                  - name: "deltaMsq"
                  - name: "sinSqTheta"
                externalWeight:
                  pythonExecutable: "{python_executable}"
                  evalScript: "{eval_script_path}"
                  inputList: [ "[X]" ]
            parameterDefinitions:
              - name: "deltaMsq"
                isEnabled: true
                priorValue: 2.0
                priorType: Flat
                parameterLimits: [0.0, 10.0]
              - name: "sinSqTheta"
                isEnabled: true
                priorValue: 0.5
                priorType: Flat
                parameterLimits: [0.0, 1.0]
""".format(
        root_path=root_path,
        eval_script_path=eval_script_path,
        python_executable=python_executable,
    )


def evaluate_config(config_text: str, work_dir: Path) -> float:
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

    likelihood_interface.propagateAndEvalLikelihood()
    GUNDAM.flushOutput()

    sample = likelihood_interface.getModelPropagator().getSampleSet().getSampleList()[0]
    bin_content_list = sample.getHistogram().getBinContentList()
    bin_context_list = sample.getHistogram().getBinContextList()

    if len(bin_content_list) != 1:
      raise RuntimeError("Expected 1 bin, got {0}".format(len(bin_content_list)))

    print("Bin contents (ExternalWeight):")
    for bin_context, bin_content in zip(bin_context_list, bin_content_list):
        print("  {0} -> sumWeights={1}".format(bin_context.bin.getSummary(False), bin_content.sumWeights))

    return float(bin_content_list[0].sumWeights)


def main() -> int:
    script_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    work_dir = Path.cwd()
    root_path = work_dir / "213ExternalWeightResponse.root"
    eval_script_path = work_dir / "213ExternalWeightResponse_eval.py"

    write_input_root_file(root_path)
    write_external_weight_script(eval_script_path)

    config_text = build_config_text(root_path, eval_script_path, sys.executable)
    sum_weights = evaluate_config(config_text, work_dir)

    expected_sum_weights = 10.0 * (1.0 + 0.5 * 2.0 * 0.5)
    if not math.isclose(sum_weights, expected_sum_weights, rel_tol=0.0, abs_tol=1.0e-9):
        print(
            "FAIL: expected external weights to give sumWeights={0}, got {1}".format(
                expected_sum_weights,
                sum_weights,
            )
        )
        return 1

    print("SUCCESS: ExternalWeight evaluates event weights through the test Python venv.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print("FAIL: {0}".format(exception))
        sys.exit(1)
