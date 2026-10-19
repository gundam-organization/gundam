#!/usr/bin/env python3
"""Binned curves can share data while keeping independent bin associations."""
import json
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path


def write_input(path):
    # ROOT object branches/containers are written by a separate C++ process.
    macro = Path.cwd() / "228SharedBinnedDials-input.C"
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TSystem.h>
#include <TObjArray.h>
{
 {
  TFile output(OUTPUT_PATH, "RECREATE");
  TTree tree("events", "events");
  double coordinate = 0.;
  tree.Branch("X", &coordinate, "X/D");
  for (double x : {.25, .75, 1.25, 1.75, 2.25, 2.75, 3.25}) {
    coordinate = x;
    tree.Fill();
  }
  tree.Write();
  TObjArray dials;
  dials.SetOwner(true);
  for (double scale : {.125, .125, .25, 0.}) {
    auto* graph = new TGraph(4);
    for (int knot = 0; knot < 4; ++knot) {
      double x = knot - 1.;
      graph->SetPoint(knot, x, 1. + scale*x*x);
    }
    dials.Add(graph);
  }
  dials.Write("dials", TObject::kSingleKey);
 }
 gSystem->Exit(0);
}
'''.replace("OUTPUT_PATH", json.dumps(str(path))))
    subprocess.run(["root", "-n", "-l", "-b", "-q", str(macro)], check=True)

def exercise(path, threads):
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(threads)
    binning = {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2, 3, 4]}]}
    config = {'enablePca': False, 'scaleParStepWithChi2Response': False,
              'generateSamplePlots': False, 'generateOneSigmaPlots': False,
              'likelihoodInterfaceConfig': {
        'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
        'dataSetList': [{'name': 'input', 'model': {
            'tree': 'events', 'filePathList': [str(path)], 'allowMultipleSamplesPerEntry': True}}],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [{'name': name, 'binning': binning}
                                             for name in ('first', 'overlap')]},
            'parametersManagerConfig': {'parameterSetList': [{
                'name': 'shape', 'isEnabled': True, 'nominalStepSize': .1,
                'parameterDefinitions': [{'parameterName': 'p', 'priorValue': .5, 'priorType': 'Flat'}],
                'dialSetDefinitions': [{'dialType': 'Spline', 'binning': binning,
                    'dialsFilePath': str(path), 'dialsList': 'dials',
                    'dialInputList': [{'name': 'p'}]}]}]}}}}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    likelihood = engine.getLikelihoodInterface()
    propagator = likelihood.getModelPropagator()
    interfaces = [dial.getDialInterface() for entry in propagator.getEventDialCache().getCache()
                  for dial in entry.getDialResponseCacheList()]
    # The unit-response bin has no dial. Equal curves in the first two bins
    # retain separate interfaces, each referenced by two events in each sample.
    assert len(interfaces) == 12, len(interfaces)
    assert len({id(interface) for interface in interfaces}) == 3
    parameter = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    for value in (.5, 0., 1.5, .25, .5):
        parameter.setParameterValue(value)
        likelihood.propagateAndEvalLikelihood()
        expected = [2 * (1 + .125 * value * value)] * 2 + [2 * (1 + .25 * value * value), 1.]
        for sample in propagator.getSampleSet().getSampleList():
            actual = [content.sumWeights for content in sample.getHistogram().getBinContentList()]
            assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                       for a, b in zip(actual, expected)), (actual, expected)


def main():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import_module('005PythonSetup-utils').requireVenv(__file__)
    path = Path.cwd() / '228SharedBinnedDials.root'
    write_input(path)
    for threads in (1, 2):
        exercise(path, threads)
    print('SUCCESS: binned dial sharing preserves bin selection and event weights.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--write-input':
        write_input(Path(sys.argv[2]))
    else:
        main()
