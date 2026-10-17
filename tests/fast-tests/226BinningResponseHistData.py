#!/usr/bin/env python3
"""Imported data histograms bypass the model's binning response."""
import json
import math
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def exercise(path, histName, threads):
    import GUNDAM

    GUNDAM.setNumberOfThreads(threads)
    sample = {'name': 'sample', 'binning': {
        'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2, 3]}]},
        'binningResponse': {'isEnabled': True,
                            'variables': [{'name': 'X', 'width': .8}]}}
    config = {'likelihoodInterfaceConfig': {
        'jointProbabilityConfig': {'type': 'LeastSquares'},
        'datasetList': [{'name': 'input', 'model': {
            'tree': 'events', 'filePathList': [str(path)], 'nominalWeightFormula': '100'},
            'selectedDataEntry': 'external', 'data': [{'name': 'external',
                'fromHistContent': {'fromRootFile': str(path), 'sampleList': [{
                    'name': 'sample', 'hist': histName, 'axisList': ['X']}]}}]}],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [sample]},
            'parametersManagerConfig': {'parameterSetList': [{
                'name': 'normalization', 'isEnabled': True, 'nominalStepSize': .1,
                'parameterDefinitions': [{'name': 'scale', 'priorValue': 1.,
                    'priorType': 'Flat',
                    'dialSetDefinitions': [{'dialType': 'Normalization'}]}]}]}}},
        'enablePca': False, 'scaleParStepWithChi2Response': False,
        'generateSamplePlots': False, 'generateOneSigmaPlots': False}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    li = engine.getLikelihoodInterface()
    li.setDataType(GUNDAM.LikelihoodInterface.DataType.RealData)
    engine.initialize()
    model = li.getModelPropagator()
    data = li.getDataPropagator()
    modelHist = model.getSampleSet().getSampleList()[0].getHistogram()
    dataHist = data.getSampleSet().getSampleList()[0].getHistogram()
    assert modelHist.isBinningResponseEnabled()
    assert not dataHist.isBinningResponseEnabled()

    def contents(hist):
        return [b.sumWeights for b in hist.getBinContentList()]

    expected = [13., 0., 7.]
    assert contents(dataHist) == expected
    nominal = contents(modelHist)
    assert nominal[0] > 0 and nominal[1] > 0 and nominal[2] == 0
    parameter = model.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    for scale in (.5, 2., 1.):
        parameter.setParameterValue(scale)
        li.propagateAndEvalLikelihood()
        assert contents(dataHist) == expected
        prediction = contents(modelHist)
        for value, prior in zip(prediction, nominal):
            assert math.isclose(value, scale * prior, abs_tol=1e-10)
        expectedLlh = sum((observed - predicted) ** 2
                          for observed, predicted in zip(expected, prediction))
        assert math.isclose(li.getLastStatLikelihood(), expectedLlh, abs_tol=1e-10)
    dataHist.refillHistogram()
    assert contents(dataHist) == expected


def main():
    import GUNDAM
    import ROOT

    print('Testing module:', GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '226BinningResponseHistData.root'
    with ROOT.TFile(str(path), 'RECREATE'):
        tree = ROOT.TTree('events', 'events')
        x = array('d', [.95])
        tree.Branch('X', x, 'X/D')
        tree.Fill()
        tree.Write()
        hist = ROOT.TH1D('flat', 'flat', 3, 0., 3.)
        histNd = ROOT.THnD('nd', 'nd', 1, array('i', [3]),
                           array('d', [0.]), array('d', [3.]))
        for index, value in enumerate((13., 0., 7.)):
            hist.SetBinContent(index + 1, value)
            histNd.SetBinContent(histNd.GetBin(array('d', [index + .5])), value)
        hist.Write()
        histNd.Write()
    for threads in (1, 4):
        for histName in ('flat', 'nd'):
            exercise(path, histName, threads)
    print('SUCCESS: imported TH1D/THnD data stay fixed in the likelihood while model binning responses remain enabled')


if __name__ == '__main__':
    main()
