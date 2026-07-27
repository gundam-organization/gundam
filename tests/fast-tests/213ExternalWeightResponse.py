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
                "import json",
                "import sys",
                "from multiprocessing import shared_memory",
                "",
                "import numpy as np",
                "",
                "",
                "def attach_array(array_description):",
                "    try:",
                "        shm = shared_memory.SharedMemory(name=array_description['shmName'], track=False)",
                "    except TypeError:",
                "        shm = shared_memory.SharedMemory(name=array_description['shmName'])",
                "        try:",
                "            from multiprocessing import resource_tracker",
                "            resource_tracker.unregister(shm._name, 'shared_memory')",
                "        except Exception:",
                "            pass",
                "    array = np.ndarray(tuple(array_description['shape']), dtype=np.float64, buffer=shm.buf)",
                "    return shm, array",
                "",
                "",
                "def respond(response):",
                "    print(json.dumps(response), flush=True)",
                "",
                "",
                "def run_worker():",
                "    state = {'shared_memory': []}",
                "    for line in sys.stdin:",
                "        command = json.loads(line)",
                "        if command['command'] == 'initialize':",
                "            state['inputs'] = {}",
                "            for input_name, input_description in command['inputs'].items():",
                "                shm, array = attach_array(input_description)",
                "                state['shared_memory'].append(shm)",
                "                state['inputs'][input_name] = array",
                "            shm, array = attach_array(command['parameterBuffer'])",
                "            state['shared_memory'].append(shm)",
                "            state['parameters'] = array",
                "            state['parameter_names'] = [entry['name'] for entry in command['parameters']]",
                "            shm, array = attach_array(command['weights'])",
                "            state['shared_memory'].append(shm)",
                "            state['weights'] = array",
                "            respond({'status': 'ok'})",
                "        elif command['command'] == 'evaluate':",
                "            parameter_values = dict(zip(state['parameter_names'], state['parameters']))",
                "            delta_msq = parameter_values['deltaMsq']",
                "            sin_sq_theta = parameter_values['sinSqTheta']",
                "            baseline_km = 295.0",
                "            enu = state['inputs']['Enu']",
                "            flavor_emit = np.rint(state['inputs']['FlavorEmit']).astype(np.int64)",
                "            flavor_detect = np.rint(state['inputs']['FlavorDetect']).astype(np.int64)",
                "            phase = 1.267 * delta_msq * baseline_km / enu",
                "            transition_prob = sin_sq_theta * np.sin(phase) ** 2",
                "            survival_prob = 1.0 - transition_prob",
                "            same_flavor = flavor_emit == flavor_detect",
                "            state['weights'][:] = np.where(same_flavor, survival_prob, transition_prob)",
                "            respond({'status': 'ok'})",
                "        elif command['command'] == 'shutdown':",
                "            respond({'status': 'ok'})",
                "            for shm in state.get('shared_memory', []):",
                "                shm.close()",
                "            return 0",
                "        else:",
                "            respond({'status': 'error', 'message': 'unknown command: {0}'.format(command['command'])})",
                "            return 1",
                "    return 0",
                "",
                "",
                "if __name__ == '__main__':",
                "    if len(sys.argv) >= 2 and sys.argv[1] == '--worker':",
                "        sys.exit(run_worker())",
                "    print(json.dumps({'status': 'error', 'message': 'expected --worker'}), flush=True)",
                "    sys.exit(1)",
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
                externalWeight:
                  pythonExecutable: "{python_executable}"
                  evalScript: "{eval_script_path}"
                  inputList: [ "[Enu]", "[FlavorEmit]", "[FlavorDetect]" ]
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
