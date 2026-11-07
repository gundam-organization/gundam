#!/usr/bin/env python3
"""Compare bounded raw-input caching to the complete dial factory path."""
import json
import math
import subprocess
import sys
from pathlib import Path


def write_input(path, offset):
    macro = path.with_suffix('.C')
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TSpline.h>
#include <TClonesArray.h>
#include <TH2D.h>
#include <TSystem.h>
#include <memory>
{
 {
  TFile output(OUTPUT_PATH, "RECREATE");
  TTree tree("events", "events");
  double coordinate = .5, enabled = 0., index = 0.;
  TGraph graphObject(4); TGraph* graph = &graphObject;
  TGraphErrors errorsObject(4); TGraphErrors* errors = &errorsObject;
  TClonesArray clones("TGraph", 2); TClonesArray* graphs = &clones;
  TSpline3* spline = nullptr;
  TH2D surfaceObject("surface", "", 4, -2., 2., 4, -2., 2.);
  TH2D* surface = &surfaceObject;
  tree.Branch("X", &coordinate, "X/D");
  tree.Branch("enabled", &enabled, "enabled/D");
  tree.Branch("index", &index, "index/D");
  tree.Branch("graph", "TGraph", &graph, 32000, 0);
  tree.Branch("errors", "TGraphErrors", &errors, 32000, 0);
  tree.Branch("graphs", "TClonesArray", &graphs, 32000, 0);
  tree.Branch("spline", "TSpline3", &spline, 32000, 0);
  tree.Branch("surface", "TH2D", &surface, 32000, 0);
  for (int entry = 0; entry < 32; ++entry) {
    enabled = entry % 4 != 0; // The first entry must not read its dial objects.
    index = entry % 2;
    double x[]{-1., 0., 1., 2.};
    double y[4];
    const int group = entry / 4;
    const int points = group < 3 ? group : (group == 4 ? 70 : 4);
    graph->Set(points);
    for (int i = 0; i < points; ++i) {
      double knot = points == 1 ? 0. : -1. + 3. * i / (points - 1);
      double response = group == 7 ? knot : (group == 5 ? 1. : (group == 6 || points == 1 ? 1.5
          : 1. + .125 * (group + 1 + OFFSET) * knot * knot));
      graph->SetPoint(i, knot, response);
    }
    for (int i = 0; i < 4; ++i) {
      y[i] = 1. + .125 * (entry / 4 + 1 + OFFSET) * x[i] * x[i];
      errors->SetPoint(i, x[i], y[i]);
      errors->SetPointError(i, .125 * entry, .25);
      for (int j = 0; j < 2; ++j) {
        auto* clone = static_cast<TGraph*>(clones.ConstructedAt(j));
        clone->SetPoint(i, x[i], y[i] + j * .25);
      }
    }
    // Supplied derivatives differ from the default spline for identical knots.
    auto ownedSpline = std::make_unique<TSpline3>("clamped", x, y, 4, "b1e1", .25 + .125 * (entry % 2), -.5);
    spline = ownedSpline.get();
    for (int ix = 1; ix <= 4; ++ix) for (int iy = 1; iy <= 4; ++iy)
      surface->SetBinContent(ix, iy, 1. + .125 * ix + .25 * iy + .0625 * entry);
    tree.Fill();
  }
  tree.Write();
 }
 gSystem->Exit(0);
}
'''.replace('OUTPUT_PATH', json.dumps(str(path))).replace('OFFSET', str(offset)))
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)


def exercise(paths, branch, dial_type, options, cached, threads):
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(threads)
    source = {'tree': 'events', 'enableDialInputCache': cached,
              'filePathList': [str(path) for path in paths], 'allowMultipleSamplesPerEntry': True,
              'dialIndexFormula': 'index'}
    inputs = [{'name': 'p'}, {'name': 'q'}] if dial_type == 'Surface' else [{'name': 'p'}]
    dial = {'dialType': dial_type, 'options': options,
            'dialBranchData': branch,
            'applyCondition': 'enabled != 0', 'dialInputList': inputs}
    # Several collections consume the same ROOT object, with independent inputs.
    definitions = [dial, {**dial, 'applyCondition': 'enabled != 0 && index == 1',
                          'allowDialExtrapolation': False, 'definitionRange': [0., 2.],
                          'knotResponseLimits': [.9, 1.6], 'minDialResponse': 1.1, 'maxDialResponse': 1.8}]
    config = {'enablePca': False, 'scaleParStepWithChi2Response': False,
              'generateSamplePlots': False, 'generateOneSigmaPlots': False,
              'likelihoodInterfaceConfig': {
        'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
        'dataSetList': [{'name': 'input', 'model': source}],
        'propagatorConfig': {
            'sampleSetConfig': {'sampleList': [{'name': name, 'binning': {
                'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}
                for name in ('first', 'overlap')]},
            'parametersManagerConfig': {'parameterSetList': [{
                'name': 'shape', 'isEnabled': True, 'nominalStepSize': .1,
                'parameterDefinitions': [{'parameterName': name, 'priorValue': 0., 'priorType': 'Flat'}
                                         for name in ('p', 'q')],
                'dialSetDefinitions': definitions}]}}}}
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    likelihood = engine.getLikelihoodInterface()
    model = likelihood.getModelPropagator()
    parameters = model.getParametersManager().getParameterSetsList()[0].getParameterList()
    snapshots = []
    for p, q in ((0., 0.), (.25, -.25), (1.5, .5), (-.5, 1.), (0., 0.)):
        parameters[0].setParameterValue(p)
        parameters[1].setParameterValue(q)
        likelihood.propagateAndEvalLikelihood()
        snapshots.append(tuple(event.getEventWeight() for sample in model.getSampleSet().getSampleList()
                               for event in sample.getEventList()))
    return snapshots


def main():
    if '--worker' not in sys.argv:
        # Debug builds log this API misuse but return an empty vector; Release
        # throws. Catch both behaviours rather than accepting a false pass.
        result = subprocess.run([sys.executable, __file__, '--worker'],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors='replace')
        print(result.stdout)
        assert result.returncode == 0, result.returncode
        assert 'getDialData not implemented' not in result.stdout
        return
    paths = [Path.cwd() / f'435DialsInputCache-{i}.root' for i in range(2)]
    for path, offset in zip(paths, (0, 16)):
        write_input(path, offset)
    cases = [('graph', 'Graph', ''),
             ('errors', 'Spline', ''),
             ('graphs', 'Spline', 'catmull-rom, monotonic'),
             ('spline', 'Spline', ''),
             ('spline', 'Spline', 'ROOT'),
             ('surface', 'Surface', 'Bilinear'),
             ('surface', 'Surface', 'Bicubic')]
    for options in ('', 'natural', 'not-a-knot, monotonic', 'akima', 'catmull-rom', 'ROOT', 'uniformity(0.0001)'):
        cases.append(('graph', 'Spline', options))
    for threads in (1, 4):
        for branch, dial_type, options in cases:
            reference = exercise(paths, branch, dial_type, options, False, threads)
            actual = exercise(paths, branch, dial_type, options, True, threads)
            differences = [(step, i, a, b) for step, (xs, ys) in enumerate(zip(reference, actual))
                           for i, (a, b) in enumerate(zip(xs, ys))
                           if a != b and not (math.isnan(a) and math.isnan(b))]
            # Preserve existing non-finite outputs too; never turn a failed
            # interpolation into a different curve via input memoization.
            assert not differences, (threads, branch, dial_type, options, differences[:8])
    print('SUCCESS: cached and uncached factories produce identical event weights across representations and options.')


if __name__ == '__main__':
    main()
