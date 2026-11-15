#!/usr/bin/env python3
"""Repeated branch curves remain shared as the loading hash index grows."""
import json
import math
import subprocess
from pathlib import Path


def write_input(path, offset):
    macro = path.with_suffix('.C')
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TSystem.h>
{
 {
  TFile output(OUTPUT_PATH, "RECREATE");
  TTree tree("events", "events");
  double x = .5, enabled = 1;
  TGraph graphObject(4); TGraph* graph = &graphObject;
  tree.Branch("X", &x, "X/D");
  tree.Branch("enabled", &enabled, "enabled/D");
  tree.Branch("graph", "TGraph", &graph, 32000, 0);
  for (int entry = 0; entry < 771; ++entry) {
    int curve = (entry * 37 + OFFSET) % 257;
    enabled = entry % 7 != 0;
    for (int i = 0; i < 4; ++i) {
      double knot = i - 1.;
      graph->SetPoint(i, knot, 1. + (curve + 1) * knot * knot / 1024.);
    }
    tree.Fill();
  }
  tree.Write();
 }
 gSystem->Exit(0);
}
'''.replace('OUTPUT_PATH', json.dumps(str(path))).replace('OFFSET', str(offset)))
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)


def exercise(paths, threads):
    import GUNDAM
    GUNDAM.setNumberOfThreads(threads)
    GUNDAM.setLightOutputMode(True)
    config = {'enablePca': False, 'scaleParStepWithChi2Response': False,
              'generateSamplePlots': False, 'generateOneSigmaPlots': False,
              'likelihoodInterfaceConfig': {
        'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
        'dataSetList': [{'name': f'input{i}', 'model': {'tree': 'events',
                         'filePathList': [str(path)], 'allowMultipleSamplesPerEntry': True}}
                        for i, path in enumerate(paths)],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [{'name': name, 'binning': {
                'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}
                for name in ('first', 'overlap')]},
            'parametersManagerConfig': {'parameterSetList': [{
                'name': 'shape', 'isEnabled': True, 'nominalStepSize': .1,
                'parameterDefinitions': [{'parameterName': 'p', 'priorValue': 0., 'priorType': 'Flat'}],
                'dialSetDefinitions': [{'dialType': 'Graph', 'dialBranchData': 'graph',
                    'applyCondition': 'enabled != 0', 'dialInputList': [{'name': 'p'}]}]}]}}}}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    likelihood = engine.getLikelihoodInterface()
    model = likelihood.getModelPropagator()
    interfaces = [response.getDialInterface() for entry in model.getEventDialCache().getCache()
                  for response in entry.getDialResponseCacheList()]
    assert len({id(interface) for interface in interfaces}) == 257
    parameter = model.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    for value, interpolated_square in ((0., 0.), (.25, .25), (1.5, 2.5), (-.5, .5), (0., 0.)):
        parameter.setParameterValue(value)
        likelihood.propagateAndEvalLikelihood()
        expected = sum(1. + (((entry * 37 + offset) % 257) + 1) * interpolated_square / 1024.
                       if entry % 7 else 1. for offset in (0, 11) for entry in range(771))
        for sample in model.getSampleSet().getSampleList():
            actual = sample.getHistogram().getBinContentList()[0].sumWeights
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def main():
    paths = [Path.cwd() / f'445DialsGrowingSharedIndex-{i}.root' for i in range(2)]
    for path, offset in zip(paths, (0, 11)):
        write_input(path, offset)
    # Repeated engine creation also exercises index clearing and rebuilding.
    for threads in (1, 4):
        exercise(paths, threads)
    print('SUCCESS: loading-index growth preserves exact sharing and event weights.')


if __name__ == '__main__':
    main()
