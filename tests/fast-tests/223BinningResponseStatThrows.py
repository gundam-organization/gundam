#!/usr/bin/env python3
"""Bin-wise pseudo-data throws with shared event contributions."""
import json
import math
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def exercise(path, gaussian, toy):
    import GUNDAM
    import ROOT

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
        ROOT.gRandom.SetSeed(12345)
        expected = [[(max(int(ROOT.gRandom.Gaus(mean, math.sqrt(mean))), 0) if gaussian
                      else int(ROOT.gRandom.Poisson(mean))) if mean else 0
                     for mean in sample] for sample in nominal]
        ROOT.gRandom.SetSeed(12345)
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
    import ROOT

    print('Testing module:', GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)
    path = Path.cwd() / '223BinningResponseStatThrows.root'
    with ROOT.TFile(str(path), 'RECREATE') as output:
        tree = ROOT.TTree('events', 'events')
        x = array('d', [.95])
        tree.Branch('X', x, 'X/D')
        tree.Fill()
        tree.Write()
    for gaussian in (False, True):
        for toy in (False, True):
            exercise(path, gaussian, toy)
    print('SUCCESS: bin-wise Poisson/Gaussian toys support shared events and stay fixed during model propagation')


if __name__ == '__main__':
    main()
