#!/usr/bin/env python3
"""Observable migration gradients against the complete smoothed likelihood."""
import copy
from importlib import import_module
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
fixture = import_module('590NumericsWeightGradient')


def add_reference_events(path):
    # Keep one non-migrating event in every bin. Compact-support smoothing
    # cannot keep a Poisson likelihood finite after all MC leaves a data bin.
    macro = Path.cwd() / '595NumericsObservableGradient-anchors.C'
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TSystem.h>
{
 {
  TFile file(INPUT, "UPDATE");
  TTree* tree = nullptr;
  file.GetObject("events", tree);
  double x = 0., rawQ3 = 0.;
  TGraph* graph = nullptr;
  TGraph* flat = nullptr;
  tree->SetBranchAddress("X", &x);
  tree->SetBranchAddress("rawQ3", &rawQ3);
  tree->SetBranchAddress("graph", &graph);
  tree->SetBranchAddress("flat", &flat);
  tree->GetEntry(0);
  for (int bin = 0; bin < 4; ++bin) {
    x = bin + .5;
    rawQ3 = -1.;
    for (int knot = 0; knot < 5; ++knot) {
      graph->SetPoint(knot, knot - 2., 1.);
      flat->SetPoint(knot, knot - 2., 1.);
    }
    tree->Fill();
  }
  tree->Write("", TObject::kOverwrite);
 }
 gSystem->Exit(0);
}
'''.replace('INPUT', json.dumps(str(path))))
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)


def check_threads(threads):
    config, build, check, _ = fixture.make_fixture(threads)
    li = config['likelihoodInterfaceConfig']
    add_reference_events(li['dataSetList'][0]['model']['filePathList'][0])
    li['jointProbabilityConfig']['ignoreBinsWithZeroPredictionAtPrior'] = False
    prop = li['propagatorConfig']
    sample = prop['sampleSetConfig']['sampleList'][0]
    definitions = prop['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    # Same parameters affect both weights and observables. Composition is
    # (override or loaded X)*q + p, independently of dial declaration order.
    definitions[0]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'X', 'variableResponseMode': 'Add',
        'dialResponseLimits': [-2., 2.], 'enableResponseSmoothing': True,
        'dialResponseSmoothingWidth': .1, 'applyCondition': 'rawQ3 >= 0'})
    definitions[1]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'X', 'variableResponseMode': 'Multiply',
        'applyCondition': 'rawQ3 >= 0'})
    definitions[2]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'X', 'variableResponseMode': 'Override',
        'applyCondition': 'X < .3 && rawQ3 >= 0'})
    for width in ({'width': .4}, {'widthFraction': .3}):
        sample['binningResponse'] = {'isEnabled': True, 'variables': [dict(name='X', **width)]}
        for normalized, eigen in ((False, False), (True, False), (False, True)):
            engine = build(config, normalized, eigen, analytic=True)
            data = engine.getLikelihoodInterface().getDataPropagator().getSampleSet().getSampleList()[0]
            before = [(b.sumWeights, b.sqrtSumSqWeights) for b in data.getHistogram().getBinContentList()]
            points = ([.1, -.13], [-.08, .05]) if eigen else (
                [.36, .92, 1.1], [-.1, .85, .45], [1.95, 1.05, 1.2])
            for point in points:
                engine.getMinimizer().evalFit(point)
                assert math.isfinite(engine.getLikelihoodInterface().getLastLikelihood()), (
                    'Finite-difference fixture left the LLH domain', width, normalized, eigen, point)
                check(engine, point)
            assert before == [(b.sumWeights, b.sqrtSumSqWeights) for b in data.getHistogram().getBinContentList()]
    # Clamped weight derivatives must not suppress observable migration.
    capped = copy.deepcopy(config)
    capped['likelihoodInterfaceConfig']['propagatorConfig']['globalEventReweightCap'] = .1
    engine = build(capped, False, False, analytic=True)
    check(engine, [.36, .92, 1.1])

    # Zero products are differentiated without division, for both weights
    # and observable multipliers. LeastSquares remains finite at zero yield.
    zero = copy.deepcopy(config)
    zero['likelihoodInterfaceConfig']['jointProbabilityConfig'] = {'type': 'LeastSquares'}
    zero_parameters = zero['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    zero_parameters[1]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'X', 'variableResponseMode': 'Multiply',
        'applyCondition': 'rawQ3 >= 0'})
    engine = build(zero, False, False, analytic=True)
    check(engine, [.36, 0., 1.1], one_sided=(1,))

    # Without stationary support, a populated data bin can lose every MC
    # contribution. Preserve the explicit failure instead of masking NaNs.
    unsupported = copy.deepcopy(config)
    unsupported_parameters = unsupported['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    for index in (0, 1):
        unsupported_parameters[index]['dialSetDefinitions'][-1].pop('applyCondition')
    engine = build(unsupported, False, False)
    minimizer = engine.getMinimizer()
    point = [1.95, 1.05, 1.2]
    minimizer.evalFit(point)
    assert not math.isfinite(engine.getLikelihoodInterface().getLastStatLikelihood())
    try:
        minimizer.evalFitGradientAt(point)
    except RuntimeError as error:
        assert 'Non-finite likelihood bin gradient' in str(error), str(error)
    else:
        raise AssertionError('Gradient of an infinite Poisson LLH was accepted')

    # Smoothing an unrelated coordinate does not make moving hard edges safe.
    bad = copy.deepcopy(config)
    bad_sample = bad['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList'][0]
    bad_sample['binning']['binningDefinition'].append({'name': 'rawQ3', 'edges': [-1., 2000.]})
    bad_parameters = bad['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['parameterDefinitions']
    bad_parameters[0]['dialSetDefinitions'].append({
        'dialType': 'Normalization', 'applyToVariable': 'rawQ3', 'variableResponseMode': 'Add'})
    try:
        build(bad, False, False, analytic=True)
    except RuntimeError as error:
        assert 'smoothing' in str(error).lower(), str(error)
    else:
        raise AssertionError('Moving unsmoothed axis was accepted')
    print(f'SUCCESS: smoothed observable gradients ({threads} threads)')


def main():
    for threads in (1, 4):
        check_threads(threads)


if __name__ == '__main__':
    main()
