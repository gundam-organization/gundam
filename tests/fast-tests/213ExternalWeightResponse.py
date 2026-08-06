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

    event_data = {
        "Enu": [0.6, 0.8, 1.2, 1.5, 2.0, 2.5],
        "FlavorEmit": [14, 14, 14, 14, 14, 14],
        "FlavorDetect": [12, 14, 12, 14, 12, 14],
    }

    with uproot.recreate(root_path) as root_file:
        tree = root_file.mktree(
            "tree_mc",
            {
                "Enu": "float64",
                "FlavorEmit": "int32",
                "FlavorDetect": "int32",
            },
        )
        tree.extend(
            {
                "Enu": array("d", event_data["Enu"]),
                "FlavorEmit": array("i", event_data["FlavorEmit"]),
                "FlavorDetect": array("i", event_data["FlavorDetect"]),
            }
        )


def write_external_weight_script(script_path: Path) -> None:
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import argparse",
                "",
                "import numpy as np",
                "",
                "",
                "def configure(arguments_):",
                "    parser = argparse.ArgumentParser()",
                "    parser.add_argument('--baseline-km', type=float, required=True)",
                "    configure.baseline_km = parser.parse_args(arguments_).baseline_km",
                "",
                "",
                "def run(command_):",
                "    command_name = command_['command']",
                "    if command_name == 'initialize':",
                "        if len(command_['parameterInfo']) != 2:",
                "            return {'status': 'error', 'message': 'unexpected parameter count'}",
                "        return {'status': 'ok'}",
                "    if command_name == 'evaluate':",
                "        parameters = command_['parameters']",
                "        enu = command_['inputs']['Enu']",
                "        flavor_emit = np.rint(command_['inputs']['FlavorEmit']).astype(np.int64)",
                "        flavor_detect = np.rint(command_['inputs']['FlavorDetect']).astype(np.int64)",
                "        phase = 1.267 * parameters['deltaMsq'] * configure.baseline_km / enu",
                "        transition_prob = parameters['sinSqTheta'] * np.sin(phase) ** 2",
                "        command_['weights'][:] = np.where(",
                "            flavor_emit == flavor_detect,",
                "            1.0 - transition_prob,",
                "            transition_prob,",
                "        )",
                "        return {'status': 'ok'}",
                "    if command_name == 'shutdown':",
                "        return {'status': 'ok'}",
                "    return {'status': 'error', 'message': 'unknown command: {0}'.format(command_name)}",
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
          - name: Enu
            isEnabled: true
            binning:
              binningDefinition:
                - name: "Enu"
                  edges: [0, 5]
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
                options:
                  type: PythonWorker
                  inputEventVarList: [ "[Enu]", "[FlavorEmit]", "[FlavorDetect]" ]
                  workerConfig:
                    pythonExecutable: "{python_executable}"
                    evalScript: "{eval_script_path}"
                    scriptArgs: ["--baseline-km", "295.0"]
            parameterDefinitions:
              - name: "deltaMsq"
                isEnabled: true
                priorValue: 2.5e-3
                priorType: Flat
                parameterLimits: [0.0, 1.0]
              - name: "sinSqTheta"
                isEnabled: true
                priorValue: 0.2
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
    work_dir = Path.cwd()
    root_path = work_dir / "213ExternalWeightResponse.root"
    eval_script_path = work_dir / "213ExternalWeightResponse_eval.py"

    write_input_root_file(root_path)
    write_external_weight_script(eval_script_path)

    config_text = build_config_text(root_path, eval_script_path, sys.executable)
    sum_weights = evaluate_config(config_text, work_dir)

    baseline_km = 295.0
    delta_msq = 2.5e-3
    sin_sq_theta = 0.2
    event_data = [
        {"Enu": 0.6, "FlavorEmit": 14, "FlavorDetect": 12},
        {"Enu": 0.8, "FlavorEmit": 14, "FlavorDetect": 14},
        {"Enu": 1.2, "FlavorEmit": 14, "FlavorDetect": 12},
        {"Enu": 1.5, "FlavorEmit": 14, "FlavorDetect": 14},
        {"Enu": 2.0, "FlavorEmit": 14, "FlavorDetect": 12},
        {"Enu": 2.5, "FlavorEmit": 14, "FlavorDetect": 14},
    ]
    expected_sum_weights = 0.0
    for event in event_data:
        phase = 1.267 * delta_msq * baseline_km / event["Enu"]
        transition_probability = sin_sq_theta * math.sin(phase) ** 2
        if event["FlavorEmit"] == event["FlavorDetect"]:
            expected_sum_weights += 1.0 - transition_probability
        else:
            expected_sum_weights += transition_probability

    if not math.isclose(sum_weights, expected_sum_weights, rel_tol=0.0, abs_tol=1.0e-9):
        print(
            "FAIL: expected external weights to give sumWeights={0}, got {1}".format(
                expected_sum_weights,
                sum_weights,
            )
        )
        return 1

    print("SUCCESS: ExternalWeight evaluates a two-flavor oscillation probability with mixed input semantics.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print("FAIL: {0}".format(exception))
        sys.exit(1)
