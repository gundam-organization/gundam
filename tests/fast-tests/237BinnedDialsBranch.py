#!/usr/bin/env python3
"""Explicit ROOT tree branches and the historical binned dial defaults."""
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
  double x;
  int category;
  events.Branch("X", &x, "X/D");
  events.Branch("category", &category, "category/I");
  for (double coordinate : {.25, .75, 1.25, 1.75}) {
    x = coordinate;
    category = (coordinate == .25 || coordinate == 1.25) ? 7 : 8;
    events.Fill();
  }
  events.Write();
  for (bool legacy : {true, false}) {
    TTree curves(legacy ? "legacy" : "curves", "curves");
    int kinematicBin, split = 7;
    TGraph* graph = nullptr;
    TSpline3* spline = nullptr;
    curves.Branch("kinematicBin", &kinematicBin, "kinematicBin/I");
    curves.Branch("category", &split, "category/I");
    curves.Branch(legacy ? "Graph" : "response", "TGraph", &graph, 32000, 0);
    curves.Branch(legacy ? "Spline" : "otherResponse", "TSpline3", &spline, 32000, 0);
    for (kinematicBin = 0; kinematicBin < 2; ++kinematicBin) {
      graph = new TGraph(4);
      TGraph splineKnots(4);
      for (int knot = 0; knot < 4; ++knot) {
        double p = knot - 1.;
        graph->SetPoint(knot, p, 1. + .1*(kinematicBin + 1)*p);
        splineKnots.SetPoint(knot, p, 1. + .2*(kinematicBin + 1)*p);
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


def exercise(path, cases):
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    binning = {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2]}]}
    for dial_type, source, scale, error in cases:
        dial = dict(dialType=dial_type, binning=binning,
                    dialInputList=[{'name': 'p'}], **source)
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
            expected = [2 + scale*(i + 1)*value for i in range(2)]
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
    for source, error in (
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
        for _, _, _, error in cases:
            if error:
                assert error in result.stdout, (error, result.stdout)
    print('SUCCESS: explicit binned branches, split selection and legacy warnings.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--exercise':
        exercise(Path(sys.argv[2]), json.loads(sys.argv[3]))
    else:
        main()
