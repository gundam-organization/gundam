#!/usr/bin/env python3
"""Compare TArrayF/TArrayD spline inputs with equivalent TGraph branches."""
import math
import sys
import subprocess
from array import array
from pathlib import Path


def write_input(path, offset, variant="regular"):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    with ROOT.TFile(str(path), "RECREATE") as output:
        tree = ROOT.TTree("events", "events")
        coordinate = array("d", [0.5])
        xs = ROOT.TArrayF(4)
        ys = ROOT.TArrayD(4)
        graph = ROOT.TGraph(4)
        tree.Branch("X", coordinate, "X/D")
        tree.Branch("parameters", xs)
        tree.Branch("responses", ys)
        tree.Branch("graph", graph, 32000, 0)
        for event in range(offset, offset + 3):
            # Vary both knot positions and responses between entries/files.
            knots = [-1.0, 0.0, 1.0, 2.0 + event]
            if variant == "edge":
                knots = knots[:event % 4]
            xs.Set(len(knots))
            ys.Set(len(knots))
            graph.Set(len(knots))
            for point, x in enumerate(knots):
                y = 1.0 + (event + 1) * 0.125 * x * x
                if variant == "edge" and event == 1:
                    y = 1.25
                xs[point], ys[point] = x, y
                graph.SetPoint(point, x, y)
            if variant == "mismatch":
                ys.Set(len(knots) + 1)
            tree.Fill()
        tree.Write()


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
        engine.evaluateLikelihood()
        content = propagator.getSampleSet().getSampleList()[0].getHistogram().getBinContentList()[0]
        results.append((content.sumWeights, content.sqrtSumSqWeights))
    return results


def main():
    paths = [Path.cwd() / f"215ArrayDialBranchData-{i}.root" for i in range(2)]
    for i, path in enumerate(paths):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--write-input", str(path), str(i * 3)], check=True)
    reference = evaluate(paths, "dialLeafName: graph", False)
    for backend in [False, True]:
        for config in ["treeExpression: graph", "dialBranchData: graph",
                       "dialBranchData: {parameterValues: parameters, responses: responses}"]:
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
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--write-input", str(path), str(i * 3), "edge"], check=True)
    reference = evaluate(paths, "dialBranchData: graph", False)
    for backend in [False, True]:
        actual = evaluate(paths, "dialBranchData: {parameterValues: parameters, responses: responses}", backend)
        for a, b in zip(actual, reference):
            if not all(math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-7) for x, y in zip(a, b)):
                raise RuntimeError(f"Empty/constant/small arrays: {actual} != {reference}")
    for config, message in [
        ("dialBranchData: {parameterValues: X, responses: responses}", "must contain a TArray"),
        ("dialBranchData: {parameterValues: missing, responses: responses}", "Missing dialBranchData branch"),
        ("dialBranchData: {parameterValues: '', responses: responses}", "non-empty"),
    ]:
        try:
            evaluate(paths, config, False)
        except RuntimeError as error:
            if message not in str(error):
                raise
        else:
            raise RuntimeError(f"Invalid input accepted: {config}")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--write-input", str(paths[0]), "0", "mismatch"], check=True)
    try:
        evaluate(paths[:1], "dialBranchData: {parameterValues: parameters, responses: responses}", False)
    except RuntimeError as error:
        if "array size mismatch" not in str(error):
            raise
    else:
        raise RuntimeError("Mismatched array lengths accepted")
    print("SUCCESS: TArray spline inputs match TGraph, including aliases and CPU propagation.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--write-input":
        write_input(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "regular")
    else:
        sys.exit(main())
