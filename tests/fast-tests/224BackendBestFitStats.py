#!/usr/bin/env python3
"""Backend likelihood semantics and host diagnostics at the best-fit point."""
import json
import math
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def make_engine(path, backend, probability, fixed_errors=False):
    import GUNDAM
    joint = {'type': probability}
    if probability == 'BarlowLLH_BANFF_OA2021':
        joint['BBNoUpdateWeights'] = fixed_errors
    li = {
        'jointProbabilityConfig': joint,
        'datasetList': [{'name': 'input', 'model': {
            'tree': 'events', 'filePathList': [str(path)], 'nominalWeightFormula': 'weight'}}],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [{
                'name': 'SFGD FHC 0p',
                'binning': {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2]}]}}]},
            'parametersManagerConfig': {'parameterSetList': [{
                'name': 'normalization', 'isEnabled': True, 'nominalStepSize': .1,
                'parameterDefinitions': [{'name': 'scale', 'priorValue': 1., 'priorType': 'Flat',
                                          'dialSetDefinitions': [{'dialType': 'Normalization'}]}]}]}}}
    if backend:
        li['backendManagerConfig'] = {'isEnabled': True, 'type': 'CPU'}
    config = {'likelihoodInterfaceConfig': li,
              'minimizerConfig': {'enablePostFitErrorFit': False, 'print_level': 0},
              'enablePca': False, 'scaleParStepWithChi2Response': False,
              'generateSamplePlots': False, 'generateOneSigmaPlots': False}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    return engine


def set_scale(engine, value):
    li = engine.getLikelihoodInterface()
    parameter = li.getModelPropagator().getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    parameter.setParameterValue(value)
    engine.evaluateLikelihood()
    return li.getLastLikelihood()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (actual, expected)


def main():
    import GUNDAM
    import ROOT
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: built without backends')
        return
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(2)
    path = Path.cwd() / '224BackendBestFitStats-input.root'
    with ROOT.TFile(str(path), 'RECREATE'):
        tree = ROOT.TTree('events', 'events')
        x, weight = array('d', [0]), array('d', [0])
        tree.Branch('X', x, 'X/D')
        tree.Branch('weight', weight, 'weight/D')
        for xv, w in [(.25, 1.5), (.75, 2.5), (1.25, 3.5), (1.75, 4.5)]:
            x[0], weight[0] = xv, w
            tree.Fill()
        tree.Write()
    for probability, fixed_errors in [('BarlowLLH_BANFF_OA2021', False),
                                       ('BarlowLLH_BANFF_OA2021', True),
                                       ('BarlowLLH_BANFF_OA2021_SFGD', False)]:
        reference = make_engine(path, False, probability, fixed_errors)
        backend = make_engine(path, True, probability, fixed_errors)
        backend.getBackendsManager().setEnableAutoMaterialize(False)
        for value in (.7, 1.4, 2., 1.):
            expected = set_scale(reference, value)
            actual = set_scale(backend, value)
            close(actual, expected)
            backend.getBackendsManager().materialize('Histograms')
            li = backend.getLikelihoodInterface()
            li.getSummary()
            close(li.getLastLikelihood(), actual)

    # A one-parameter, four-event fit verifies the saved tree's actual contents.
    output = Path.cwd() / '224BackendBestFitStats-output.root'
    app = GUNDAM.GundamApp('best-fit statistics regression')
    app.openOutputFile(str(output))
    backend.setSaveDir(app, 'FitterEngine')
    set_scale(backend, 1.4)
    backend.getBackendsManager().materialize('Histograms')
    backend.fit()
    expected = backend.getLikelihoodInterface().getLastStatLikelihood()
    # Fit output is flushed by RootMinimizer; inspect through a read-only handle.
    with ROOT.TFile(str(output), 'READ') as saved:
        stats = saved.Get('FitterEngine/postFit/bestFitStats')
        assert stats
        stats.GetEntry(0)
        close(stats.GetLeaf('statLikelihoodAtBestFit').GetValue(), expected)
        sample_sum = 0.
        for branch in stats.GetListOfBranches():
            leaves = list(branch.GetListOfLeaves())
            if leaves and leaves[0].GetName() == 'llhSample':
                sample_sum += leaves[0].GetValue()
                close(sum(leaf.GetValue() for leaf in leaves[1:]), leaves[0].GetValue())
        close(sample_sum, expected)
    print('SUCCESS: Barlow-Beeston backend likelihoods and bestFitStats agree with host samples and bins')


if __name__ == '__main__':
    main()
