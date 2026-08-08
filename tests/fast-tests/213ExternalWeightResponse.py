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


def build_config_text(
    root_path: Path,
    eval_script_path: Path,
    python_executable: str,
    use_binned_weights: bool,
) -> str:
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
          additionalLeavesStorage: ["Enu", "FlavorEmit", "FlavorDetect"]
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
                  useBinnedWeights: {use_binned_weights}
                  binning:
                    binningDefinition:
                      - name: Enu
                        # Bin centers are exactly the six test event energies.
                        edges: [0.55, 0.65, 0.95, 1.45, 1.55, 2.45, 2.55]
                      - name: FlavorEmit
                        values: [14]
                      - name: FlavorDetect
                        values: [12, 14]
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
        use_binned_weights=str(use_binned_weights).lower(),
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

    event_weight_list = []
    print("Event weights (ExternalWeight):")
    for event in sample.getEventList():
        enu = event.getVariables().fetchVariable("Enu").getVarAsDouble()
        flavor_emit = event.getVariables().fetchVariable("FlavorEmit").getVarAsDouble()
        flavor_detect = event.getVariables().fetchVariable("FlavorDetect").getVarAsDouble()
        event_weight = float(event.getEventWeight())
        event_weight_list.append((event.getIndices().treeEntry, event_weight))
        print(
            "  treeEntry={0}: Enu={1}, FlavorEmit={2}, FlavorDetect={3}, weight={4}".format(
                event.getIndices().treeEntry,
                enu,
                flavor_emit,
                flavor_detect,
                event_weight,
            )
        )

    print("Bin contents (ExternalWeight):")
    for bin_context, bin_content in zip(bin_context_list, bin_content_list):
        print("  {0} -> sumWeights={1}".format(bin_context.bin.getSummary(False), bin_content.sumWeights))

    return float(bin_content_list[0].sumWeights), event_weight_list


def main() -> int:
    script_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    work_dir = Path.cwd()
    root_path = work_dir / "213ExternalWeightResponse.root"
    eval_script_path = work_dir / "213ExternalWeightResponse_eval.py"

    write_input_root_file(root_path)
    write_external_weight_script(eval_script_path)

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
    expected_event_weights = {}
    for event in event_data:
        phase = 1.267 * delta_msq * baseline_km / event["Enu"]
        transition_probability = sin_sq_theta * math.sin(phase) ** 2
        event_weight = (
            1.0 - transition_probability
            if event["FlavorEmit"] == event["FlavorDetect"]
            else transition_probability
        )
        expected_sum_weights += event_weight
        expected_event_weights[event["Enu"]] = event_weight

    for use_binned_weights in (False, True):
        config_text = build_config_text(
            root_path,
            eval_script_path,
            sys.executable,
            use_binned_weights,
        )
        sum_weights, event_weight_list = evaluate_config(config_text, work_dir)
        mode_name = "binned" if use_binned_weights else "event-by-event"
        for tree_entry, event_weight in event_weight_list:
            expected_weight = expected_event_weights[event_data[tree_entry]["Enu"]]
            if not math.isclose(event_weight, expected_weight, rel_tol=0.0, abs_tol=1.0e-9):
                print(
                    "FAIL ({0}): treeEntry={1} expected weight={2}, got {3}".format(
                        mode_name, tree_entry, expected_weight, event_weight
                    )
                )
                return 1
        if not math.isclose(sum_weights, expected_sum_weights, rel_tol=0.0, abs_tol=1.0e-9):
            print(
                "FAIL ({0}): expected external weights to give sumWeights={1}, got {2}".format(
                    mode_name,
                    expected_sum_weights,
                    sum_weights,
                )
            )
            return 1
        print("SUCCESS ({0}): sumWeights={1}".format(mode_name, sum_weights))

    print("SUCCESS: ExternalWeight matches in event-by-event and binned modes.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print("FAIL: {0}".format(exception))
        sys.exit(1)
