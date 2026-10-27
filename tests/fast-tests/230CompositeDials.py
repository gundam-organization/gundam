#!/usr/bin/env python3
"""Composite DAGs reuse ordinary global, binned and event-by-event dials."""
import copy
import json
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def write_input(path):
    macro = Path.cwd() / '230CompositeDials-input.C'
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TH2D.h>
#include <TObjArray.h>
#include <TSystem.h>
{
 {
  TFile output(OUTPUT_PATH, "RECREATE");
  TTree tree("events", "events");
  double coordinate = 0., rawQ3 = 0.;
  TGraph curve, flatCurve;
  TGraph *graph = &curve, *flat = &flatCurve;
  tree.Branch("X", &coordinate, "X/D");
  tree.Branch("rawQ3", &rawQ3, "rawQ3/D");
  tree.Branch("graph", "TGraph", &graph, 32000, 0);
  tree.Branch("flat", "TGraph", &flat, 32000, 0);
  for (int event = 0; event < 6; ++event) {
    coordinate = .25 + .5*event;
    rawQ3 = 250.*event;
    for (int knot = 0; knot < 5; ++knot) {
      double x = knot - 2.;
      graph->SetPoint(knot, x, 1. + .1*(event + 1)*x*x);
      flat->SetPoint(knot, x, 1.);
    }
    tree.Fill();
  }
  tree.Write();
  TObjArray curves, surfaces, wideCurves;
  curves.SetOwner(true); surfaces.SetOwner(true); wideCurves.SetOwner(true);
  for (int bin = 0; bin < 3; ++bin) {
    auto* curve = new TGraph(5);
    for (int knot = 0; knot < 5; ++knot) {
      double x = knot - 2.;
      curve->SetPoint(knot, x, 1. + .2*(bin + 1)*x);
    }
    curves.Add(curve);
    auto* wideCurve = new TGraph(5);
    for (int knot = 0; knot < 5; ++knot) {
      double x = -40. + 20.*knot;
      wideCurve->SetPoint(knot, x, 5. + .001*(bin + 1)*x*x);
    }
    wideCurves.Add(wideCurve);
    auto* surface = new TH2D(("surface" + std::to_string(bin)).c_str(), "", 5, -2.5, 2.5, 5, -2.5, 2.5);
    surface->SetDirectory(nullptr);
    for (int ix = 1; ix <= 5; ++ix) for (int iy = 1; iy <= 5; ++iy) {
      surface->SetBinContent(ix, iy, 2. + .1*(bin + 1)*surface->GetXaxis()->GetBinCenter(ix)
                                             + .2*surface->GetYaxis()->GetBinCenter(iy));
    }
    surfaces.Add(surface);
  }
  curves.Write("curves", TObject::kSingleKey);
  wideCurves.Write("wideCurves", TObject::kSingleKey);
  surfaces.Write("surfaces", TObject::kSingleKey);
 }
 gSystem->Exit(0);
}
'''.replace('OUTPUT_PATH', json.dumps(str(path))))
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)


def formula(name, expression, *inputs, **options):
    return dict(name=name, dialType='Formula', dialInputList=[dict(source, name=f'arg{i}') for i, source in enumerate(inputs)],
                options={'formulaStr': expression}, **options)


def composite(nodes, output, **options):
    return dict(name='graph', dialType='Composite',
                options=dict(nodes=nodes, output=output), **options)


def legacy_options(dial):
    result = copy.deepcopy(dial)
    if result['dialType'] == 'Composite':
        result['options']['nodes'] = [legacy_options(node) for node in result['options']['nodes']]
    if result['dialType'] in ('Composite', 'Formula', 'CompiledLibDial'):
        result['dialConfig'] = result.pop('options')
    return result


def make_config(path, dial):
    binning = {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2, 3]}]}
    return {'enablePca': False, 'scaleParStepWithChi2Response': False,
            'generateSamplePlots': False, 'generateOneSigmaPlots': False,
            'likelihoodInterfaceConfig': {
                'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
                'dataSetList': [{'name': 'input', 'model': {'tree': 'events', 'filePathList': [str(path)],
                    'variableDict': [{'name': 'q3', 'expr': 'rawQ3 + 0.'}]}}],
                'propagatorConfig': {'sampleSetConfig': {'sampleList': [{'name': 'sample', 'binning': binning}]},
                    'parametersManagerConfig': {'parameterSetList': [{
                        'name': 'parameters', 'isEnabled': True, 'nominalStepSize': .1,
                        'parameterDefinitions': [dict(name=name, priorValue=value, priorType='Flat')
                                                 for name, value in [('p', .5), ('q', .25)]],
                        'dialSetDefinitions': [dial]}]}}}}


def build(config):
    import GUNDAM
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    return engine


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11), (actual, expected)


def exercise(path, dial, expected, observable=False, points=None):
    engine = build(make_config(path, dial))
    likelihood = engine.getLikelihoodInterface()
    model = likelihood.getModelPropagator()
    parameters = model.getParametersManager().getParameterSetsList()[0].getParameterList()
    sample = model.getSampleSet().getSampleList()[0]
    if points is None:
        points = [(.5, .25), (.5, .25), (-.5, .25), (-.5, .75), (1.2, -.25), (.5, .25)]
    for p, q in points:
        parameters[0].setParameterValue(p)
        parameters[1].setParameterValue(q)
        for _ in range(2):
            likelihood.propagateAndEvalLikelihood()
            sums = [0., 0., 0.]
            for event in sample.getEventList():
                index = event.getIndices().treeEntry
                value = expected(p, q, index)
                if observable:
                    close(event.getEventWeight(), 1.)
                    close(event.getVariables().fetchVariable('X').getVarAsDouble(), value)
                    bin_index = int(value) if 0 <= value < 3 else -1
                    assert event.getIndices().bin == bin_index
                    if bin_index >= 0:
                        sums[bin_index] += 1.
                else:
                    close(event.getEventWeight(), value)
                    sums[index // 2] += value
            for content, expected_sum in zip(sample.getHistogram().getBinContentList(), sums):
                close(content.sumWeights, expected_sum)
            # The Asimov copy must retain the graph and its original inputs.
            data = likelihood.getDataPropagator().getSampleSet().getSampleList()[0]
            for event in data.getEventList():
                prior = expected(.5, .25, event.getIndices().treeEntry)
                if observable:
                    close(event.getVariables().fetchVariable('X').getVarAsDouble(), prior)
                else:
                    close(event.getEventWeight(), prior)


def reject(config, message):
    try:
        build(config)
    except Exception as error:
        assert message.lower() in str(error).lower(), (message, str(error))
    else:
        raise AssertionError('Invalid composite accepted: ' + message)


def main():
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '230CompositeDials.root'
    write_input(path)
    first = {'parameterName': 'p'}
    second = {'parameterIndex': 1}
    upstream = {'dialResponse': 'source'}
    binning = {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2, 3]}]}
    for threads in (1, 2):
        GUNDAM.setNumberOfThreads(threads)
        # Forward references, shared subgraph, signed intermediates, mixed inputs.
        dag = composite([
            formula('output', '2+[arg0]+[arg1]+[arg2]', {'dialResponse': 'left'}, {'dialResponse': 'right'}, second),
            formula('right', '[arg0]*[arg0]', upstream),
            formula('source', '[arg0]', first),
            formula('left', '2*[arg0]', upstream)], 'output')
        exercise(path, dag, lambda p, q, i: max(0., 2 + 2*p + p*p + q))
        if threads == 1:
            exercise(path, legacy_options(dag), lambda p, q, i: max(0., 2 + 2*p + p*p + q))

        # Nested composites infer their own dependencies from direct references.
        nested = composite([formula('sum', '2+[arg0]+[arg1]', upstream, second),
                            formula('source', '[arg0]*[arg0]', first)], 'sum')
        nested['name'] = 'nested'
        exercise(path, composite([nested], 'nested'), lambda p, q, i: 2 + p*p + q)

        # Use ordinary branch dials, including the constant-one shortcut.
        for dial_type in ('Spline', 'Graph'):
            node = dict(name='curve', dialType=dial_type, dialBranchData='graph', dialInputList=[upstream])
            curve_dial = composite([formula('source', '[arg0]', first), node], 'curve')
            if dial_type == 'Spline':
                reference = lambda p, q, i: 1 + .1*(i + 1)*p*p
            else:
                reference = lambda p, q, i: 1 + .1*(i + 1)*(abs(p) if abs(p) <= 1 else 3*abs(p)-2)
            exercise(path, curve_dial, reference)
        exercise(path, composite([dict(name='one', dialType='Spline', dialBranchData='flat',
                                      dialInputList=[first]), formula('output', '2*[arg0]', {'dialResponse': 'one'})],
                                 'output'), lambda p, q, i: 2.)

        # Per-bin selection is delegated to the existing collection loader.
        for dial_type, objects, inputs in [('Graph', 'curves', [upstream]),
                                            ('Surface', 'surfaces', [upstream, second])]:
            node = dict(name='curve', dialType=dial_type, binning=binning, dialsFilePath=str(path),
                        dialsList=objects, dialInputList=inputs)
            if dial_type == 'Surface':
                node['options'] = 'Bilinear'
                reference = lambda p, q, i: 2 + .1*(i//2 + 1)*p + .2*q
            else:
                reference = lambda p, q, i: 1 + .2*(i//2 + 1)*p
            exercise(path, composite([formula('source', '[arg0]', first), node], 'curve'), reference)

        # Limits act at each node, then separately at the graph output.
        limited = composite([formula('source', '[arg0]', first, dialResponseLimits=[-.2, .8]),
                             formula('output', '1+[arg0]', upstream)], 'output', dialResponseLimits=[0., 1.5])
        exercise(path, limited, lambda p, q, i: min(1.5, 1 + min(.8, max(-.2, p))))
        mirrored = composite([dict(name='source', dialType='Normalization', dialInputList=[first],
                                   useMirrorDial=True, mirrorLowEdge=0., mirrorHighEdge=1.),
                              formula('output', '1+[arg0]', upstream)], 'output')
        exercise(path, mirrored, lambda p, q, i: 1 + (abs(p) if abs(p) <= 1 else 2-abs(p)))
        smoothed = composite([dict(name='source', dialType='Normalization', dialInputList=[first],
                                   dialResponseLimits={'min': 0.}, enableResponseSmoothing=True,
                                   dialResponseSmoothingWidth=.2),
                              formula('output', '1+[arg0]', upstream)], 'output')
        def smooth_reference(p, q, i):
            t = (p + .2)/.4
            return 1 + (0. if p <= -.2 else p if p >= .2 else .2*(2*t**3-t**4))
        exercise(path, smoothed, smooth_reference, points=[(.5, .25), (0., .25), (-.1, .25), (.1, .25)])
        observable = composite([formula('source', '[arg0]+[arg1]', first, second)], 'source',
                               applyToVariable='X', variableResponseMode='Override')
        exercise(path, observable, lambda p, q, i: p + q, observable=True)

        exercise(path, composite([formula('eventOnly', '1+.001*[arg0]', {'eventVariable': 'q3'})], 'eventOnly'),
                 lambda p, q, i: 1+.25*i)

        event_only_nested = composite([formula('eventOnly', '1+.001*[arg0]', {'eventVariable': 'q3'})], 'eventOnly')
        event_only_nested['name'] = 'nested'
        exercise(path, composite([event_only_nested], 'nested'), lambda p, q, i: 1+.25*i)

        # The requested delta/alpha/q3 abscissa, followed by a discretely selected spline.
        source_node = formula('source', '[arg0]+[arg1]*(.056*[arg2]-41.6)', first, second,
                              {'eventVariable': 'q3'}, minDialResponse=-40., maxDialResponse=40.)
        spline_node = dict(name='spline', dialType='Spline', binning=binning,
                           dialsFilePath=str(path), dialsList='wideCurves', dialInputList=[upstream])
        points = [(.5, .25), (.5, .25), (-.5, .25), (-.5, .75), (0., 1.), (30., 2.),
                  (-40., 0.), (40., 0.), (.5, .25)]
        def abscissa(p, q, i):
            return max(-40., min(40., p+q*(.056*250*i-41.6)))
        exercise(path, composite([source_node, spline_node], 'spline'),
                 lambda p, q, i: 5+.001*(i//2+1)*abscissa(p, q, i)**2, points=points)
        named_source = dict(name='source', dialType='Formula', dialInputList=[
            {'name': 'momentum', 'eventVariable': 'q3'},
            {'name': 'slope', 'parameterIndex': 1}, {'name': 'offset', 'parameterName': 'p'}],
            options={'formulaStr': '[offset]+[slope]*(.056*[momentum]-41.6)'},
            minDialResponse=-40., maxDialResponse=40.)
        exercise(path, composite([named_source, spline_node], 'spline'),
                 lambda p, q, i: 5+.001*(i//2+1)*abscissa(p, q, i)**2, points=points)
        exercise(path, composite([source_node], 'source', applyToVariable='X', variableResponseMode='Override'),
                 abscissa, observable=True, points=points)

        # Event inputs can also accompany upstream dial responses, inside a nested graph.
        nested = composite([formula('sum', '2+[arg0]+.001*[arg1]', upstream, {'eventVariable': 'q3'}),
                            formula('source', '[arg0]*[arg0]', first)], 'sum')
        nested['name'] = 'nested'
        exercise(path, composite([nested], 'nested'),
                 lambda p, q, i: 2+p*p+.25*i)

        # Reading an observable captures its loaded value; repeated migrations do not accumulate.
        snapshot = formula('loadedX', '[arg0]+[arg1]', {'eventVariable': 'X'}, {'parameterName': 'p'},
                           applyToVariable='X', variableResponseMode='Override')
        exercise(path, snapshot, lambda p, q, i: .25+.5*i+p, observable=True)

    # Compiled functions can consume intermediate responses through DialInputBuffer.
    source = Path.cwd() / '230CompositeDials-library.cpp'
    library = source.with_suffix('.so')
    source.write_text('extern "C" double evalVariable(double* x) { return 2 + x[0] + 3*x[1]; }\n')
    subprocess.run(['c++', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    compiled = dict(name='compiled', dialType='CompiledLibDial', dialInputList=[upstream, second],
                    options={'libraryFile': str(library)})
    exercise(path, composite([formula('source', '[arg0]*[arg0]', first), compiled], 'compiled'),
             lambda p, q, i: 2 + p*p + 3*q)
    exercise(path, legacy_options(composite([formula('source', '[arg0]*[arg0]', first), compiled], 'compiled')),
             lambda p, q, i: 2 + p*p + 3*q)

    # Prepared factories remain usable as sources driven by graph parameters.
    source.write_text(r'''
extern "C" int initialize(const char*, int, const char*[], int) { return 4; }
extern "C" int update(const char*, double* table, int bins, const double* par, int) {
  for (int i = 0; i < bins; ++i) table[i] = 1 + par[0] + 2*par[1];
  return 0;
}
extern "C" double bin(const char*, int, double*, int) { return .5; }
extern "C" int weights(const char*, int, int, double*, int, int* indices, double* weights) {
  indices[0] = 0; weights[0] = 1.; return 1;
}
''')
    table_library = source.with_name('230CompositeDials-table.so')
    subprocess.run(['c++', '-shared', '-fPIC', str(source), '-o', str(table_library)], check=True)
    for dial_type in ('Tabulated', 'Kriged'):
        table_config = dict(name=dial_type, libraryPath=str(table_library), initFunction='initialize',
                            initArguments=[], updateFunction='update')
        if dial_type == 'Tabulated':
            table_config.update(binningFunction='bin', binningVariables=['X'])
        else:
            table_config.update(weightFunction='weights', weightVariables=['X'], weightNormalization=0.)
        table = dict(name='source', dialType=dial_type, dialInputList=[first, second], tableConfig=table_config)
        exercise(path, composite([table, formula('output', '[arg0]*[arg0]', upstream)], 'output'),
                 lambda p, q, i: (1+p+2*q)**2)

    worker = Path.cwd() / '230CompositeDials-worker.py'
    worker.write_text("""def run(command):
    if command['command'] == 'evaluate':
        command['weights'][:] = 1 + command['parameters']['p'] + 2*command['parameters']['q']
    return {'status': 'ok'}
""")
    external = dict(name='source', dialType='ExternalWeight', dialInputList=[first, second], options={
        'type': 'PythonWorker', 'inputEventVarList': ['X'],
        'workerConfig': {'pythonExecutable': sys.executable, 'evalScript': str(worker)}})
    exercise(path, composite([external, formula('output', '[arg0]*[arg0]', upstream)], 'output'),
             lambda p, q, i: (1+p+2*q)**2)

    # Direct references also work for parameter-attached composites.
    implicit = composite([formula('output', '1+[arg0]', first)], 'output')
    config = make_config(path, implicit)
    parameter_set = config['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]
    parameter_set.pop('dialSetDefinitions')
    parameter_set['parameterDefinitions'] = [dict(name='p', priorValue=.5, priorType='Flat',
                                                 dialSetDefinitions=[implicit])]
    engine = build(config)
    likelihood = engine.getLikelihoodInterface()
    model = likelihood.getModelPropagator()
    model.getParametersManager().getParameterSetsList()[0].getParameterList()[0].setParameterValue(1.2)
    likelihood.propagateAndEvalLikelihood()
    for event in model.getSampleSet().getSampleList()[0].getEventList():
        close(event.getEventWeight(), 2.2)

    good = make_config(path, composite([formula('source', '[arg0]', first),
                                       formula('output', '1+[arg0]', upstream)], 'output'))
    def changed_graph():
        result = copy.deepcopy(good)
        graph = result['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig'][
            'parameterSetList'][0]['dialSetDefinitions'][0]['options']
        return result, graph

    config = copy.deepcopy(good)
    root = config['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig'][
        'parameterSetList'][0]['dialSetDefinitions'][0]
    root['dialInputList'] = [{'name': 'p'}]
    reject(config, 'without an outer dialInputList')

    config, graph = changed_graph()
    graph['nodes'][0]['dialInputList'] = [{'name': 'arg0', 'dialResponse': 'output'}]
    reject(config, 'cycle')
    for bad_source, message in [({'parameterName': 'missing'}, 'Could not find parameter'),
                                ({'parameterIndex': 2}, 'parameterIndex'),
                                ({'parameterIndex': -1}, 'parameterIndex'),
                                ({'parameterIndex': .5}, 'parameterIndex'),
                                ({'parameterName': 'p', 'parameterIndex': 0}, 'exactly one of')]:
        config, graph = changed_graph()
        graph['nodes'][0]['dialInputList'] = [dict(bad_source, name='arg0')]
        reject(config, message)
    config, graph = changed_graph()
    graph['nodes'][0]['name'] = 'output'
    reject(config, 'duplicate')
    config, graph = changed_graph()
    graph['output'] = 'missing'
    reject(config, 'unknown output')
    config, graph = changed_graph()
    graph['nodes'][0]['dialInputList'] = [{'name': 'arg0', 'inputIndex': 2}]
    reject(config, 'inputIndex')
    config, graph = changed_graph()
    graph['nodes'][0]['dialInputList'] = [{'name': 'arg0', 'inputIndex': .5}]
    reject(config, 'inputIndex')
    config, graph = changed_graph()
    graph['nodes'][0]['dialInputList'] = [{'parameterName': 'p', 'dialResponse': 'output'}]
    reject(config, 'exactly one of')
    for bad_input in ({'eventVariable': ''}, {'eventVariable': 'q3', 'parameterName': 'p'}):
        config, graph = changed_graph()
        graph['nodes'][0]['dialInputList'] = [dict(bad_input, name='arg0')]
        reject(config, 'eventVariable')
    config, graph = changed_graph()
    graph['nodes'][0] = dict(name='source', dialType='Normalization', dialInputList=[{'eventVariable': 'q3'}])
    reject(config, 'only supported by Formula')
    config, graph = changed_graph()
    graph['nodes'][1]['dialInputList'] = [{'name': 'arg0', 'dialResponse': 'missing'}]
    reject(config, 'unknown dialResponse')
    config, graph = changed_graph()
    graph['nodes'][1]['applyCondition'] = 'X < 1'
    reject(config, 'missing node')
    config, graph = changed_graph()
    graph['nodes'][1] = dict(name='output', dialType='Normalization', dialInputList=[upstream, second])
    reject(config, 'exactly one input')
    config, graph = changed_graph()
    graph['nodes'][1]['options']['formulaStr'] = '[missing]'
    reject(config, 'undeclared input')
    config, graph = changed_graph()
    graph['nodes'].append(formula('unused', '[arg0]', first))
    reject(config, 'every node')
    for dial_type in ('Tabulated', 'Kriged', 'ExternalWeight'):
        config, graph = changed_graph()
        graph['nodes'][1] = dict(name='output', dialType=dial_type, dialInputList=[upstream])
        reject(config, 'requires parameter inputs')
        config, graph = changed_graph()
        child = composite([dict(name='prepared', dialType=dial_type, dialInputList=[first])], 'prepared')
        child['name'] = 'output'
        graph['nodes'] = [child]
        root = config['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig'][
            'parameterSetList'][0]['dialSetDefinitions'][0]
        root.update(useMirrorDial=True, mirrorLowEdge=0., mirrorHighEdge=1.)
        reject(config, 'requires parameter inputs')
    if hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        config = copy.deepcopy(good)
        config['likelihoodInterfaceConfig']['backendManagerConfig'] = {'isEnabled': True, 'type': 'CPU'}
        reject(config, 'CPU composite')
    print('SUCCESS: composite dial graphs, event/bin binding, observables and validation.')


if __name__ == '__main__':
    main()
