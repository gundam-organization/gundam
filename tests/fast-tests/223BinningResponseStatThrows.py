#!/usr/bin/env python3
"""Bin-wise pseudo-data throws with shared event contributions."""
import json
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def reference_throws(nominal, gaussian):
    # Preserve the exact ROOT random stream without loading PyROOT in Python.
    mode = 'gaussian' if gaussian else 'poisson'
    output = Path.cwd() / f'223BinningResponseStatThrows-{mode}.json'
    macro = Path.cwd() / f'223BinningResponseStatThrows-{mode}.C'
    values = json.dumps(nominal).replace('[', '{').replace(']', '}')
    macro.write_text(r'''#include <TRandom.h>
#include <TSystem.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
{
 {
  std::vector<std::vector<double>> nominal = NOMINAL;
  gRandom->SetSeed(12345);
  std::ofstream output(OUTPUT_PATH);
  output << "[";
  bool firstSample = true;
  for (const auto& sample : nominal) {
    if (!firstSample) output << ",";
    firstSample = false;
    output << "[";
    bool firstBin = true;
    for (double mean : sample) {
      if (!firstBin) output << ",";
      firstBin = false;
      int count = mean == 0. ? 0 : (GAUSSIAN
          ? std::max(int(gRandom->Gaus(mean, std::sqrt(mean))), 0)
          : int(gRandom->Poisson(mean)));
      output << count;
    }
    output << "]";
  }
  output << "]";
 }
 gSystem->Exit(0);
}
'''.replace('NOMINAL', values).replace('OUTPUT_PATH', json.dumps(str(output)))
        .replace('GAUSSIAN', str(gaussian).lower()))
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)
    return json.loads(output.read_text())


def exercise(path, gaussian, toy):
    import GUNDAM

    samples = []
    for name in ('hard', 'smooth'):
        sample = {'name': name, 'binning': {
            'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2, 3]}]}}
        if name == 'smooth':
            sample['binningResponse'] = {
                'isEnabled': True, 'variables': [{'name': 'X', 'width': .2}]}
        samples.append(sample)
    config = {'likelihoodInterfaceConfig': {
        'enableStatThrowInToys': True, 'enableEventMcThrow': False,
        'gaussStatThrowInToys': gaussian,
        'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
        'datasetList': [{'name': 'inputs', 'model': {
            'tree': 'events', 'filePathList': [str(path)],
            'allowMultipleSamplesPerEntry': True, 'nominalWeightFormula': '100'}}],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': samples},
            'parametersManagerConfig': {'parameterSetList': [{
                'name': 'normalization', 'isEnabled': True, 'enableThrowToyParameters': False,
                'nominalStepSize': .1, 'parameterDefinitions': [{
                    'name': 'scale', 'priorValue': 1., 'priorType': 'Flat',
                    'dialSetDefinitions': [{'dialType': 'Normalization'}]}]}]}}},
        'enablePca': False, 'scaleParStepWithChi2Response': False,
        'generateSamplePlots': False, 'generateOneSigmaPlots': False}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    li = engine.getLikelihoodInterface()
    if toy:
        li.setDataType(GUNDAM.LikelihoodInterface.DataType.Toy)
    engine.initialize()
    model = li.getModelPropagator()
    data = li.getDataPropagator()

    def contents(propagator):
        return [[b.sumWeights for b in sample.getHistogram().getBinContentList()]
                for sample in propagator.getSampleSet().getSampleList()]

    nominal = contents(model)
    assert nominal[1][0] > 0 and nominal[1][1] > 0 and nominal[1][2] == 0
    if not toy:
        # Use identical random streams to check both distributions bin by bin.
        expected = reference_throws(nominal, gaussian)
        GUNDAM.FitterEngine.setRandomSeed(12345)
        li.throwStatErrors(data)
        assert contents(data) == expected, (contents(data), expected)
    thrown = contents(data)
    assert thrown != nominal
    smooth = data.getSampleSet().getSampleList()[1]
    assert smooth.getEventList()[0].getEventWeight() == 100
    parameter = model.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    for scale in (.5, 2., 1.):
        parameter.setParameterValue(scale)
        li.propagateAndEvalLikelihood()
        assert contents(data) == thrown
        for actual, prior in zip(contents(model), nominal):
            for value, mean in zip(actual, prior):
                assert math.isclose(value, scale * mean, abs_tol=1e-10)
    # Exclusive bins retain their historical refill behavior. Shared-event
    # histograms instead recover the prediction when explicitly refilled.
    for sample in data.getSampleSet().getSampleList():
        sample.getHistogram().refillHistogram()
    assert contents(data)[0] == thrown[0]
    assert contents(data)[1] == nominal[1]


def main():
    import GUNDAM
    import numpy as np
    import uproot

    print('Testing module:', GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)
    path = Path.cwd() / '223BinningResponseStatThrows.root'
    with uproot.recreate(path) as output:
        tree = output.mktree('events', {'X': 'float64'})
        tree.extend({'X': np.array([.95])})
    for gaussian in (False, True):
        for toy in (False, True):
            exercise(path, gaussian, toy)
    print('SUCCESS: bin-wise Poisson/Gaussian toys support shared events and stay fixed during model propagation')


if __name__ == '__main__':
    main()
