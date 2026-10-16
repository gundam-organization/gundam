#!/usr/bin/env python3
"""Unrelated eigen scans keep a PythonWorker alive without reevaluating it."""
import json
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import_module("005PythonSetup-utils").requireVenv(__file__)


def run_scan(executable, root, threads, scan_migration):
    work = Path.cwd()
    tag = f"{threads}-{'migration' if scan_migration else 'flux'}"
    commands_path = work / f"220ExternalWeightScan-{tag}-commands.jsonl"
    worker_path = work / f"220ExternalWeightScan-{tag}-worker.py"
    worker_path.write_text(f'''import json


def run(command_):
    with open({str(commands_path)!r}, "a") as output:
        output.write(json.dumps({{
            "command": command_["command"],
            "parameters": command_["parameters"],
        }}) + "\\n")
    if command_["command"] == "evaluate":
        command_["weights"][:] = command_["inputs"]["X"] + command_["parameters"]["shift"]
''')
    if commands_path.exists():
        commands_path.unlink()
    points = 100
    config = {"fitterEngineConfig": {
        "parameterScannerConfig": {"nbPoints": points},
        "likelihoodInterfaceConfig": {
            "jointProbabilityConfig": {"type": "LeastSquares", "lsqPoissonianApproximation": True},
            "dataSetList": [{"name": "inputs", "model": {
                "tree": "events", "filePathList": [str(root)]}}],
            "propagatorConfig": {
                "sampleSetConfig": {"sampleList": [{"name": "sample", "binning": {
                    "binningDefinition": [{"name": "X", "edges": [0, 1, 2]}]}}]},
                "parametersManagerConfig": {"parameterSetList": [
                    {"name": "Flux", "isEnabled": True, "enableEigenDecomp": True,
                     "parameterDefinitionFilePath": str(work / "220ExternalWeightScan-covariance.root"),
                     "covarianceMatrix": "flux",
                     "nominalStepSize": .01, "parameterDefinitions": [
                         {"name": "flux1", "parameterIndex": 0, "priorValue": 1., "stdDev": .05,
                          "dialSetDefinitions": [{"dialType": "Normalization"}]},
                         {"name": "flux2", "parameterIndex": 1, "priorValue": 1., "stdDev": .1,
                          "dialSetDefinitions": [{"dialType": "Normalization"}]},
                     ]},
                    {"name": "Migration", "isEnabled": True, "isScanEnabled": scan_migration,
                     "nominalStepSize": .01,
                     "parameterDefinitions": [{"name": "shift", "priorValue": .05, "stdDev": .1}],
                     "dialSetDefinitions": [{
                         "dialType": "ExternalWeight", "applyToVariable": "X", "variableResponseMode": "Override",
                         "dialInputList": [{"name": "shift"}], "options": {
                             "type": "PythonWorker", "inputEventVarList": ["[X]"],
                             "workerConfig": {"pythonExecutable": sys.executable, "evalScript": str(worker_path)},
                         },
                     }]},
                ]},
            },
        },
    }}
    config_path = work / f"220ExternalWeightScan-{tag}.json"
    config_path.write_text(json.dumps(config))
    log_path = work / f"220ExternalWeightScan-{tag}.log"
    with log_path.open("w") as output:
        result = subprocess.run([
            executable, "-c", str(config_path), "-a", "-d", "--scan", str(points),
            "--light-mode", "--cache-manager", "off", "-t", str(threads),
            "-o", str(work / f"220ExternalWeightScan-{tag}.root"),
        ], stdout=output, stderr=subprocess.STDOUT, timeout=120)
    # Concurrent progress output may interleave UTF-8 characters. The scan
    # markers checked below are ASCII; keep the log usable for diagnostics.
    log = log_path.read_text(errors="replace")
    assert result.returncode == 0, log
    assert "Scanning: Flux/#0_eigen / 100 steps" in log, log
    assert "Scanning: Flux/#1_eigen / 100 steps" in log, log
    commands = [json.loads(line) for line in commands_path.read_text().splitlines()]
    assert sum(command["command"] == "initialize" for command in commands) == 1, commands
    assert sum(command["command"] == "shutdown" for command in commands) == 1, commands
    evaluations = [command["parameters"] for command in commands if command["command"] == "evaluate"]
    assert evaluations[0] == {"shift": .05}, evaluations
    if scan_migration:
        assert "Scanning: Migration/#0_shift / 100 steps" in log, log
        # First evaluation, nbPoints+1 scan points, and restoration to the initial value.
        assert len(evaluations) == points + 3, len(evaluations)
        assert len({state["shift"] for state in evaluations}) == points + 1, evaluations
        assert evaluations[-1] == {"shift": .05}, evaluations[-1]
    else:
        assert "Scanning: Migration/" not in log, log
        assert evaluations == [{"shift": .05}], evaluations
    print(f"threads={threads}, scan_migration={scan_migration}: {len(evaluations)} evaluate commands", flush=True)


def main():
    import numpy as np
    import uproot

    executable = shutil.which("gundamFitter")
    assert executable, "gundamFitter must be on PATH"
    print("Testing executable:", executable, flush=True)
    # Eigen decomposition requires an explicit covariance on supported branches.
    root_executable = shutil.which("root")
    assert root_executable, "ROOT must be on PATH to create the covariance fixture"
    macro = Path.cwd() / "ExternalWeightScanCovariance.C"
    macro.write_text('''void ExternalWeightScanCovariance(){
    TFile output("220ExternalWeightScan-covariance.root", "RECREATE");
    TMatrixDSym covariance(2);
    covariance(0, 0) = 0.0025;
    covariance(1, 1) = 0.01;
    covariance.Write("flux");
}
''')
    result = subprocess.run([root_executable, "-n", "-l", "-b", "-q", str(macro)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, result.stdout.decode(errors="replace")
    root = Path.cwd() / "220ExternalWeightScan-input.root"
    with uproot.recreate(root) as output:
        tree = output.mktree("events", {"X": "float64"})
        tree.extend({"X": np.array([.2, .8, 1.2, 1.8])})
    for threads in [1, 4]:
        run_scan(executable, root, threads, scan_migration=False)
        run_scan(executable, root, threads, scan_migration=True)
    print("SUCCESS: unrelated eigen scans reuse Python responses; related scans evaluate and restore them.")


if __name__ == "__main__":
    main()
