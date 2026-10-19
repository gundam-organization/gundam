#!/usr/bin/env python3
"""Compare TArrayF/TArrayD spline inputs with equivalent TGraph branches."""
import json
import math
import sys
import subprocess
from pathlib import Path
from importlib import import_module

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module("005PythonSetup-utils").requireVenv(__file__)


def write_input(path, offset, variant="regular"):
    # uproot cannot write these TArray/TGraph object branches.
    macro = Path.cwd() / f"215ArrayDialBranchData-{offset}-input.C"
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TArrayF.h>
#include <TArrayD.h>
#include <TSystem.h>
#include <string>
#include <vector>
{
 {
  TFile output(OUTPUT_PATH, "RECREATE");
  TTree tree("events", "events");
  double coordinate = .5;
  TArrayF xs(4);
  TArrayD ys(4);
  TArrayF* parameters = &xs;
  TArrayD* responses = &ys;
  TGraph graphObject(4);
  TGraph* graph = &graphObject;
  tree.Branch("X", &coordinate, "X/D");
  tree.Branch("parameters", "TArrayF", &parameters);
  tree.Branch("responses", "TArrayD", &responses);
  tree.Branch("graph", "TGraph", &graph, 32000, 0);
  const std::string variant = VARIANT;
  for (int event = OFFSET; event < OFFSET + 3; ++event) {
    std::vector<double> knots{-1., 0., 1., 2. + event};
    if (variant == "edge") knots.resize(event % 4);
    xs.Set(knots.size()); ys.Set(knots.size()); graph->Set(knots.size());
    for (int point = 0; point < int(knots.size()); ++point) {
      double x = knots[point];
      double y = 1. + (event + 1)*.125*x*x;
      if (variant == "edge" && event == 1) y = 1.25;
      xs[point] = x; ys[point] = y;
      graph->SetPoint(point, x, y);
    }
    if (variant == "mismatch") ys.Set(knots.size() + 1);
    tree.Fill();
  }
  tree.Write();
 }
 gSystem->Exit(0);
}
'''.replace("OUTPUT_PATH", json.dumps(str(path))).replace("VARIANT", json.dumps(variant))
        .replace("OFFSET", str(offset)))
    subprocess.run(["root", "-n", "-l", "-b", "-q", str(macro)], check=True)


def evaluate(paths, branch_data, backend):
    import GUNDAM
    print("Testing module:", GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(2)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(f"""
likelihoodInterfaceConfig:
  jointProbabilityConfig:
    type: LeastSquares
    lsqPoissonianApproximation: true
  dataSetList:
    - name: arrays
      model:
        tree: events
        filePathList: [{', '.join(repr(str(p)) for p in paths)}]
  backendManagerConfig:
    isEnabled: {str(backend).lower()}
    type: CPU
  propagatorConfig:
    sampleSetConfig:
      sampleList:
        - name: sample
          binning: {{binningDefinition: [{{name: X, edges: [0, 1]}}]}}
          dataSets: [arrays]
    parametersManagerConfig:
      parameterSetList:
        - name: spline
          isEnabled: true
          nominalStepSize: 0.1
          parameterDefinitions:
            - parameterName: p
              priorValue: 0.0
              priorType: Flat
          dialSetDefinitions:
            - dialType: Spline
              {branch_data}
              dialInputList: [{{name: p}}]
""")
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(builder.getConfig()))
    engine.configure()
    likelihood = engine.getLikelihoodInterface()
    engine.initialize()
    propagator = likelihood.getModelPropagator()
    parameter = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    results = []
    for value in [0.0, -0.5, 0.5, 1.0, 1.5]:
        parameter.setParameterValue(value)
        likelihood.propagateAndEvalLikelihood()
        content = propagator.getSampleSet().getSampleList()[0].getHistogram().getBinContentList()[0]
        results.append((content.sumWeights, content.sqrtSumSqWeights))
    return results


def expect_failure(paths, config, message, index):
    # Loader errors may terminate a worker thread. Isolate invalid inputs so
    # both thrown exceptions and fatal exits can be checked by the parent.
    payload = json.dumps({"paths": [str(path) for path in paths], "config": config})
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--check-invalid", payload],
                            capture_output=True, text=True)
    log = result.stdout + result.stderr
    Path(f"215ArrayDialBranchData-invalid-{index}.log").write_text(log)
    assert result.returncode != 0 and message in log, (config, result.returncode, log[-4000:])


def main():
    paths = [Path.cwd() / f"215ArrayDialBranchData-{i}.root" for i in range(2)]
    for i, path in enumerate(paths):
        write_input(path, i * 3)
    reference = evaluate(paths, "dialLeafName: graph", False)
    for backend in [False, True]:
        for config in ["treeExpression: graph", "dialBranchData: graph",
                       "dialBranchData: {parameterValuesTArray: parameters, weightValuesTArray: responses}"]:
            result = evaluate(paths, config, backend)
            for actual, expected in zip(result, reference):
                for a, b in zip(actual, expected):
                    if not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7):
                        raise RuntimeError(f"Mismatch ({config}, CPU={backend}): {result} != {reference}")
    if not math.isclose(reference[0][0], 6.0):
        raise RuntimeError(f"Unexpected prior sum: {reference}")
    if math.isclose(reference[2][0], reference[0][0]):
        raise RuntimeError("Spline response did not change")
    for i, path in enumerate(paths):
        write_input(path, i * 3, "edge")
    reference = evaluate(paths, "dialBranchData: graph", False)
    for backend in [False, True]:
        actual = evaluate(paths, "dialBranchData: {parameterValuesTArray: parameters, weightValuesTArray: responses}", backend)
        for a, b in zip(actual, reference):
            if not all(math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-7) for x, y in zip(a, b)):
                raise RuntimeError(f"Empty/constant/small arrays: {actual} != {reference}")
    for index, (config, message) in enumerate([
        ("dialBranchData: {parameterValuesTArray: X, weightValuesTArray: responses}", "must contain a TArray"),
        ("dialBranchData: {parameterValuesTArray: missing, weightValuesTArray: responses}", "Missing dialBranchData branch"),
        ("dialBranchData: {parameterValuesTArray: '', weightValuesTArray: responses}", "non-empty"),
    ]):
        expect_failure(paths, config, message, index)
    write_input(paths[0], 0, "mismatch")
    expect_failure(paths[:1], "dialBranchData: {parameterValuesTArray: parameters, weightValuesTArray: responses}",
                   "array size mismatch", 3)
    print("SUCCESS: TArray spline inputs match TGraph, including aliases and CPU propagation.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--write-input":
        write_input(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "regular")
    elif len(sys.argv) > 1 and sys.argv[1] == "--check-invalid":
        payload = json.loads(sys.argv[2])
        evaluate([Path(path) for path in payload["paths"]], payload["config"], False)
    else:
        sys.exit(main())
