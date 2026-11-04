#!/usr/bin/env python3
"""Unrelated eigen scans keep a PythonWorker alive without reevaluating it."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def run_scan(root, threads, scan_migration):
    import GUNDAM
    import uproot

    GUNDAM.setNumberOfThreads(threads)
    work = Path.cwd()
    tag = f"{threads}-{'migration' if scan_migration else 'flux'}"
    commands_path = work / f"610RuntimeExternalWeightScan-{tag}-commands.jsonl"
    worker_path = work / f"610RuntimeExternalWeightScan-{tag}-worker.py"
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
    config = {
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
                     "parameterDefinitionFilePath": str(work / "610RuntimeExternalWeightScan-covariance.root"),
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
    }
    output_path = work / f"610RuntimeExternalWeightScan-{tag}.root"
    app = GUNDAM.GundamApp("610RuntimeExternalWeightScan")
    app.openOutputFile(str(output_path))
    fitter = GUNDAM.FitterEngine()
    fitter.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    fitter.getLikelihoodInterface().setForceAsimovData(True)
    fitter.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    fitter.initialize()
    fitter.getMinimizer().scanParameters(app, "scan")
    # Destroy the engine before inspecting shutdown, then close the ROOT file.
    del fitter
    del app

    # Check the scan output rather than matching application log messages.
    expected_graphs = {"Flux_0_eigen_TGraph", "Flux_1_eigen_TGraph",
                       "Flux_0_flux1_TGraph", "Flux_1_flux2_TGraph"}
    if scan_migration:
        expected_graphs.add("Migration_0_shift_TGraph")
    with uproot.open(output_path) as output:
        graphs = output["scan/llh"]
        assert set(graphs.keys(cycle=False)) == expected_graphs, graphs.keys()
        for name in expected_graphs:
            assert graphs[name].member("fNpoints") == points + 1, name
    commands = [json.loads(line) for line in commands_path.read_text().splitlines()]
    assert sum(command["command"] == "initialize" for command in commands) == 1, commands
    assert sum(command["command"] == "shutdown" for command in commands) == 1, commands
    evaluations = [command["parameters"] for command in commands if command["command"] == "evaluate"]
    assert evaluations[0] == {"shift": .05}, evaluations
    if scan_migration:
        # Initial evaluation, nbPoints+1 scan points, and restoration.
        assert len(evaluations) == points + 3, len(evaluations)
        assert len({state["shift"] for state in evaluations}) == points + 1, evaluations
        assert evaluations[-1] == {"shift": .05}, evaluations[-1]
    else:
        assert evaluations == [{"shift": .05}], evaluations
    print(f"threads={threads}, scan_migration={scan_migration}: {len(evaluations)} evaluations", flush=True)


def main():
    import numpy as np
    import uproot

    import GUNDAM
    print("Testing module:", GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    # Eigen decomposition requires an explicit covariance on supported branches.
    root_executable = shutil.which("root")
    assert root_executable, "ROOT must be on PATH to create the covariance fixture"
    macro = Path.cwd() / "ExternalWeightScanCovariance.C"
    macro.write_text('''void ExternalWeightScanCovariance(){
    TFile output("610RuntimeExternalWeightScan-covariance.root", "RECREATE");
    TMatrixDSym covariance(2);
    covariance(0, 0) = 0.0025;
    covariance(1, 1) = 0.01;
    covariance.Write("flux");
}
''')
    result = subprocess.run([root_executable, "-n", "-l", "-b", "-q", str(macro)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, result.stdout.decode(errors="replace")
    root = Path.cwd() / "610RuntimeExternalWeightScan-input.root"
    with uproot.recreate(root) as output:
        tree = output.mktree("events", {"X": "float64"})
        tree.extend({"X": np.array([.2, .8, 1.2, 1.8])})
    for threads in [1, 4]:
        run_scan(root, threads, scan_migration=False)
        run_scan(root, threads, scan_migration=True)
    print("SUCCESS: unrelated eigen scans reuse Python responses; related scans evaluate and restore them.")


if __name__ == "__main__":
    main()
