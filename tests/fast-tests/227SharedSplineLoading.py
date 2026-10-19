#!/usr/bin/env python3
"""Check shared branch dials across events, samples, datasets, and threads."""
import json
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path


def write_input(path):
    # ROOT object branches/containers are written by a separate C++ process.
    macro = Path.cwd() / "227SharedSplineLoading-input.C"
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TSystem.h>
#include <TArrayD.h>
{
 {
  TFile output(OUTPUT_PATH, "RECREATE");
  TTree tree("events", "events");
  double coordinate = .5, enabled = 1.;
  TArrayD xs(4), ys(4);
  TArrayD *parameters = &xs, *responses = &ys;
  TGraph graphObject(4);
  TGraph* graph = &graphObject;
  tree.Branch("X", &coordinate, "X/D");
  tree.Branch("enabled", &enabled, "enabled/D");
  tree.Branch("parameters", "TArrayD", &parameters);
  tree.Branch("responses", "TArrayD", &responses);
  tree.Branch("graph", "TGraph", &graph, 32000, 0);
  for (int event = 0; event < 12; ++event) {
    enabled = (event % 4 != 0);
    for (int knot = 0; knot < 4; ++knot) {
      double x = knot - 1.;
      double y = event % 3 == 0 ? 1. + .125*x*x : (event % 3 == 1 ? 1. + .25*x : x);
      xs[knot] = x; ys[knot] = y;
      graph->SetPoint(knot, x, y);
    }
    tree.Fill();
  }
  tree.Write();
 }
 gSystem->Exit(0);
}
'''.replace("OUTPUT_PATH", json.dumps(str(path))))
    subprocess.run(["root", "-n", "-l", "-b", "-q", str(macro)], check=True)

def exercise(paths, branch_data, threads):
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(threads)
    config = {'enablePca': False, 'scaleParStepWithChi2Response': False,
              'generateSamplePlots': False, 'generateOneSigmaPlots': False,
              'likelihoodInterfaceConfig': {
        'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
        'dataSetList': [{'name': f'input{i}', 'model': {
            'tree': 'events', 'filePathList': [str(path)], 'allowMultipleSamplesPerEntry': True}} for i, path in enumerate(paths)],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [{'name': name, 'binning': {
                'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}
                for name in ('first', 'overlap')]},
            'parametersManagerConfig': {'parameterSetList': [{
                'name': 'shape', 'isEnabled': True, 'nominalStepSize': .1,
                'parameterDefinitions': [{'parameterName': 'p', 'priorValue': .5, 'priorType': 'Flat'}],
                'dialSetDefinitions': [{'dialType': 'Spline', 'dialBranchData': branch_data,
                    'applyCondition': 'enabled != 0', 'dialInputList': [{'name': 'p'}]}]}]}}}}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    likelihood = engine.getLikelihoodInterface()
    propagator = likelihood.getModelPropagator()
    interfaces = [dial.getDialInterface() for entry in propagator.getEventDialCache().getCache()
                  for dial in entry.getDialResponseCacheList()]
    # 9 enabled entries per file, referenced in two overlapping samples.
    assert len(interfaces) == 36, len(interfaces)
    assert len({id(interface) for interface in interfaces}) == 3, 'Expected one interface per unique curve'
    parameter = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    for value in (.5, 0., 1.5, .25, .5):
        parameter.setParameterValue(value)
        likelihood.propagateAndEvalLikelihood()
        expected = 6 + 6 * (1 + .125 * value * value + 1 + .25 * value + value)
        for sample in propagator.getSampleSet().getSampleList():
            assert len(sample.getEventList()) == 24
            actual = sample.getHistogram().getBinContentList()[0].sumWeights
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def main():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import_module("005PythonSetup-utils").requireVenv(__file__)
    paths = [Path.cwd() / f'227SharedSplineLoading-{i}.root' for i in range(2)]
    for path in paths:
        write_input(path)
    for threads in (1, 2):
        for branch_data in ('graph', {'parameterValuesTArray': 'parameters', 'weightValuesTArray': 'responses'}):
            exercise(paths, branch_data, threads)
    print('SUCCESS: branch splines are shared across events, samples and datasets.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--write-input':
        write_input(Path(sys.argv[2]))
    else:
        main()
