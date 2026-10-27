#!/usr/bin/env python3
"""Explicit bin sources, ROOT curve branches and historical curve defaults."""
import json
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path


def write_input(path):
    macro = Path.cwd() / '237BinnedDialsBranch-input.C'
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TSpline.h>
#include <TSystem.h>
{
 {
  TFile output(OUTPUT_PATH, "RECREATE");
  TTree events("events", "events");
  double x, theta = .5;
  int category;
  events.Branch("X", &x, "X/D");
  events.Branch("category", &category, "category/I");
  events.Branch("theta_mu", &theta, "theta_mu/D");
  for (double coordinate : {.25, .75, 1.25, 1.75}) {
    x = coordinate;
    category = (coordinate == .25 || coordinate == 1.25) ? 7 : 8;
    events.Fill();
  }
  events.Write();
  for (int mode : {0, 1, 2, 3}) {
    const bool legacy = mode == 0;
    TTree curves(legacy ? "legacy" : mode == 1 ? "curves" : mode == 2 ? "noIndex" : "flat", "curves");
    int kinematicBin = 999, split = 7;
    double minEnu, maxEnu, minThetaMu = -.25, maxThetaMu = .75;
    float numericFloat = .5;
    double array[2] = {0., 1.};
    TGraph* graph = nullptr;
    TSpline3* spline = nullptr;
    if (mode != 2) curves.Branch("kinematicBin", &kinematicBin, "kinematicBin/I");
    curves.Branch("minEnu", &minEnu, "minEnu/D");
    curves.Branch("maxEnu", &maxEnu, "maxEnu/D");
    curves.Branch("minThetaMu", &minThetaMu, "minThetaMu/D");
    curves.Branch("maxThetaMu", &maxThetaMu, "maxThetaMu/D");
    curves.Branch("numericFloat", &numericFloat, "numericFloat/F");
    curves.Branch("array", array, "array[2]/D");
    curves.Branch("category", &split, "category/I");
    curves.Branch(legacy ? "Graph" : "response", "TGraph", &graph, 32000, 0);
    curves.Branch(legacy ? "Spline" : "otherResponse", "TSpline3", &spline, 32000, 0);
    for (int bin = 0; bin < 2; ++bin) {
      minEnu = bin + .125;
      maxEnu = bin + .875;
      graph = new TGraph(4);
      TGraph splineKnots(4);
      for (int knot = 0; knot < 4; ++knot) {
        double p = knot - 1.;
        graph->SetPoint(knot, p, 1. + (mode == 3 && bin == 0 ? 0. : .1*(bin + 1)*p));
        splineKnots.SetPoint(knot, p, 1. + (mode == 3 && bin == 0 ? 0. : .2*(bin + 1)*p));
      }
      spline = new TSpline3("spline", &splineKnots);
      curves.Fill();
      delete graph;
      delete spline;
    }
    curves.Write();
  }
 }
 gSystem->Exit(0);
}
'''.replace('OUTPUT_PATH', json.dumps(str(path))))
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)


def binning_definition():
    return {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2]}]}


def exercise(path, cases):
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    binning = binning_definition()
    for case_index, (dial_type, source, scale, error) in enumerate(cases):
        GUNDAM.flushOutput()
        print(f'BEGIN_CASE {case_index}\n', flush=True)
        dial = dict(dialType=dial_type, dialInputList=[{'name': 'p'}], **source)
        if 'dialsBinningDefinition' not in dial and 'binning' not in dial:
            dial['binning'] = binning
        config = {'enablePca': False, 'scaleParStepWithChi2Response': False,
                  'generateSamplePlots': False, 'generateOneSigmaPlots': False,
                  'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
            'dataSetList': [{'name': 'input', 'model': {
                'tree': 'events', 'filePathList': [str(path)]}}],
            'propagatorConfig': {
                'sampleSetConfig': {'sampleList': [{'name': 'sample', 'binning': binning}]},
                'parametersManagerConfig': {'parameterSetList': [{
                    'name': 'shape', 'isEnabled': True, 'numberOfParameters': 1, 'nominalStepSize': .1,
                    'parameterDefinitions': [{'name': 'p', 'priorValue': .5, 'priorType': 'Flat'}],
                    'dialSetDefinitions': [dial]}]}}}}
        engine = GUNDAM.FitterEngine()
        try:
            engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
            engine.initialize()
        except RuntimeError:
            if error:
                GUNDAM.flushOutput()
                continue
            raise
        assert error is None, f'Expected error: {error}'
        likelihood = engine.getLikelihoodInterface()
        propagator = likelihood.getModelPropagator()
        parameter = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
        for value in (.5, 0., 1.5):
            parameter.setParameterValue(value)
            likelihood.propagateAndEvalLikelihood()
            actual = [b.sumWeights for b in propagator.getSampleSet().getSampleList()[0]
                      .getHistogram().getBinContentList()]
            selected = 1 if 'dialsBinningDefinition' in source else 2
            expected = [2 + selected*scale*(i + 1)*value for i in range(2)]
            if ':flat:' in source.get('binnedDialsPath', ''):
                expected[0] = 2.
            assert len(actual) == len(expected) and all(
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                for a, b in zip(actual, expected)), (source, actual, expected)


def main():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import_module('005PythonSetup-utils').requireVenv(__file__)
    path = Path.cwd() / '237BinnedDialsBranch.root'
    write_input(path)
    explicit = []
    for dial_type in ('Graph', 'Spline'):
        for branch, scale in (('response', .1), ('otherResponse', .2)):
            for source in (
                {'binnedDialsPath': f'{path}:curves:{branch}'},
                {'binnedDialsPath': f'{path}:curves:{branch}', 'dialsTreePath': 'curves'},
                {'dialsFilePath': str(path), 'dialsTreePath': f'curves:{branch}'},
                {'dialsFilePath': str(path), 'dialsTreePath': 'curves', 'dialsBranch': branch},
            ):
                explicit.append((dial_type, source, scale, None))
    tree_binning = [
        {'name': 'X', 'minValueBranch': 'minEnu', 'maxBranchValue': 'maxEnu'},
        {'name': 'theta_mu', 'minValueBranch': 'minThetaMu', 'maxBranchValue': 'maxThetaMu'},
        {'name': 'category', 'valueBranch': 'category'},
    ]
    for kind in ('Graph', 'Spline'):
        for branch, scale in (('response', .1), ('otherResponse', .2)):
            for tree in ('curves', 'noIndex', 'flat'):
                explicit.append((kind, {'binnedDialsPath': f'{path}:{tree}:{branch}',
                                       'dialsBinningDefinition': tree_binning}, scale, None))
            explicit.append((kind, {'binnedDialsPath': f'{path}:noIndex:{branch}'}, scale, None))
            explicit.append((kind, {'binnedDialsPath': f'{path}:flat:{branch}'}, scale, None))
        # Floating point equality selections also use the branch's numeric value.
        explicit.append((kind, {'binnedDialsPath': f'{path}:curves:response',
                               'dialsBinningDefinition': [tree_binning[0], tree_binning[2],
                                   {'name': 'theta_mu', 'valueBranch': 'numericFloat'}]}, .1, None))
    for source, error in (
        ({'binnedDialsPath': f'{path}:curves:response', 'binning': binning_definition(),
          'dialsBinningDefinition': tree_binning}, 'Specify either binning or dialsBinningDefinition'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBinningDefinition': []}, 'must be a non-empty list'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBinningDefinition': {}}, 'must be a non-empty list'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBinningDefinition': [tree_binning[0]]*2}, 'Duplicate binning variable'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBinningDefinition': [
            {'name': 'X', 'minValueBranch': 'minEnu'}]}, 'requires either valueBranch or both'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBinningDefinition': [
            dict(tree_binning[0], valueBranch='category')]}, 'requires either valueBranch or both'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBinningDefinition': [
            {'name': 'X', 'valueBranch': 'missing'}]}, 'Could not find binning branch'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBinningDefinition': [
            {'name': 'X', 'valueBranch': 'array'}]}, 'must contain one numeric scalar'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBinningDefinition': [
            {'name': 'X', 'valueBranch': 'otherResponse'}]}, 'must contain one numeric scalar'),
        ({'dialsFilePath': str(path), 'dialsList': 'dials',
          'dialsBinningDefinition': tree_binning}, 'dialsBinningDefinition requires a TTree'),
        ({'binnedDialsPath': f'{path}:curves:response', 'binning': {
            'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}, 'does not match the number of bins'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBinningDefinition': [
            {'name': 'X', 'minValueBranch': 'maxEnu', 'maxBranchValue': 'minEnu'}]}, 'Invalid bin bounds'),
        ({'binnedDialsPath': f'{path}:curves'}, 'file:tree:branch'),
        ({'binnedDialsPath': f'{path}:curves:'}, 'Empty dialsBranch'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsBranch': 'otherResponse'}, 'Conflicting dialsBranch'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsTreePath': 'legacy'}, 'Conflicting dialsTreePath'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsFilePath': 'other.root'}, 'Conflicting dialsFilePath'),
        ({'binnedDialsPath': f'{path}:curves:missing'}, 'Could not find binned dial branch'),
        ({'binnedDialsPath': f'{path}:curves:category'}, 'must contain TGraph or TSpline3'),
        ({'binnedDialsPath': f'{path}:curves:response', 'dialsList': 'dials'}, 'cannot be combined with dialsList'),
    ):
        explicit.append(('Spline', source, 0, error))
    legacy = [(kind, {'dialsFilePath': str(path), 'dialsTreePath': 'legacy'}, scale, None)
              for kind, scale in (('Graph', .1), ('Spline', .2))]
    warning = 'Implicit binned dial branch selection is deprecated'
    for name, cases, count in (('explicit', explicit, 0), ('legacy', legacy, 2)):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--exercise',
                                 str(path), json.dumps(cases)], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        Path(f'237BinnedDialsBranch-{name}.log').write_text(result.stdout)
        assert result.returncode == 0, result.stdout
        occurrences = result.stdout.count(warning)
        assert (occurrences == 0 if count == 0 else occurrences >= count), result.stdout
        outputs = result.stdout.split('BEGIN_CASE ')[1:]
        assert len(outputs) == len(cases), result.stdout
        for case, output in zip(cases, outputs):
            error = case[3]
            if error:
                assert error in output, (error, output)
    print('SUCCESS: explicit binning sources, ignored metadata and legacy curve warnings.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--exercise':
        exercise(Path(sys.argv[2]), json.loads(sys.argv[3]))
    else:
        main()
