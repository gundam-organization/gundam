#!/usr/bin/env python3
"""Compare CPU table dials with native propagation and an analytic producer."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

import_module("005PythonSetup-utils").requireVenv(__file__)


def float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def main():
    import GUNDAM

    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: GUNDAM was built without backends')
        return 0
    script_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('cpu_fixture', script_dir / '211CpuBackendPropagation.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    work = Path.cwd()
    root_path = work / '215CpuTableDials.root'
    library = work / '215CpuTableDials.so'
    source = work / '215CpuTableDials.cpp'
    fixture.write_input_root_file(root_path)
    source.write_text(r'''
extern "C" int initialize(const char*, int, const char*[], int) { return 4; }
extern "C" int update(const char*, double table[], int bins, const double par[], int) {
  for (int i = 0; i < bins; ++i) table[i] = 1. + par[0] * (i + 1) + par[1] * i * i;
  return 0;
}
extern "C" double bin(const char*, int, double values[], int) {
  return 3. * (values[0] + .5);
}
extern "C" int weights(const char*, int, int, double values[], int, int index[], double weight[]) {
  index[0] = 3; weight[0] = -.25;
  index[1] = 0; weight[1] = .5;
  index[2] = 2; weight[2] = .75;
  return values[0] < 0. ? 2 : 3;
}
''')
    subprocess.run(['c++', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    GUNDAM.setRuntimeWorkingDirectory(str(work))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(2)

    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(fixture.build_config_text(root_path, False))
    base_config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
    sets = []
    for dial_type in ('Tabulated', 'Kriged'):
        table_config = dict(name=dial_type, libraryPath=str(library), initFunction='initialize',
                            initArguments=[], updateFunction='update')
        if dial_type == 'Tabulated':
            table_config.update(binningFunction='bin', binningVariables=['X'])
        else:
            table_config.update(weightFunction='weights', weightVariables=['X'], weightNormalization=0.)
        sets.append(dict(name=dial_type, isEnabled=True, nominalStepSize=.1,
                         parameterDefinitions=[dict(name=name, priorValue=0., priorType='Flat') for name in ('a', 'b')],
                         dialSetDefinitions=[dict(dialType=dial_type, dialInputList=[dict(name='a'), dict(name='b')],
                                                  minDialResponse=.1, maxDialResponse=3., tableConfig=table_config)]))
    sets.append(dict(name='Scale', isEnabled=True, nominalStepSize=.1,
                     parameterDefinitions=[dict(name='scale', priorValue=1., priorType='Flat')],
                     dialSetDefinitions=[dict(dialType='Normalization', dialInputList=[dict(name='scale')])]))
    base_config['likelihoodInterfaceConfig']['propagatorConfig']['parametersManagerConfig']['parameterSetList'] = sets
    # Repeated point, each table independently, second input only, scale only,
    # negative responses (clamped) and return to a previously evaluated point.
    points = [(0., 0., 0., 0., 1.), (0., 0., 0., 0., 1.),
              (.2, .03, 0., 0., 1.), (.2, .03, .4, 0., 1.),
              (.2, .03, .4, -.1, 1.), (.2, .03, .4, -.1, 2.),
              (-2., .01, .1, .03, 1.), (0., 0., 0., 0., 1.)]

    def run(backend):
        config = json.loads(json.dumps(base_config))
        if backend:
            config['likelihoodInterfaceConfig']['backendManagerConfig'] = dict(isEnabled=True, type=backend)
        engine = GUNDAM.FitterEngine()
        engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
        engine.initialize()
        if backend and not engine.getBackendsManager().hasBackend():
            raise RuntimeError('CPU backend was not initialized')
        li = engine.getLikelihoodInterface()
        parameters = li.getModelPropagator().getParametersManager().getParameterSetsList()
        sample = li.getModelPropagator().getSampleSet().getSampleList()[0]
        results = []
        for a, b, c, d, scale in points:
            for parameter_set, values in zip(parameters, [(a, b), (c, d), (scale,)]):
                for parameter, value in zip(parameter_set.getParameterList(), values):
                    parameter.setParameterValue(value)
            engine.evaluateLikelihood()
            ordered = sorted((event.getIndices().treeEntry, float(event.getEventWeight())) for event in sample.getEventList())
            actual = [weight for _, weight in ordered]
            table_a = [1. + a * (i + 1) + b * i * i for i in range(4)]
            table_b = [1. + c * (i + 1) + d * i * i for i in range(4)]
            for event, x in enumerate([-.5, -.4, -.3, .2, .3, .4]):
                position = 3. * (x + .5)
                index = min(int(position), 2)
                fraction = float32(position - index)
                tabulated = fraction * table_a[index] + (1. - fraction) * table_a[index + 1]
                kriged = -.25 * table_b[3] + .5 * table_b[0]
                if x >= 0.:
                    kriged += .75 * table_b[2]
                expected = scale * min(3., max(.1, tabulated)) * min(3., max(.1, kriged))
                if not math.isclose(actual[event], expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError(f'{backend} event {event}: {actual[event]} != {expected}')
            bins = sample.getHistogram().getBinContentList()
            results.append(actual + [float(item.sumWeights) for item in bins]
                           + [float(item.sqrtSumSqWeights)**2 for item in bins] + [float(li.getLastLikelihood())])
            GUNDAM.flushOutput()
        return results

    reference = run(None)
    result = run('CPU')
    for point, (actual, expected) in enumerate(zip(result, reference)):
        for output, (a, b) in enumerate(zip(actual, expected)):
            if not math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11):
                raise RuntimeError(f'CPU point {point} output {output}: {a} != {b}')
    print('SUCCESS: CPU Tabulated/Kriged weights, histograms and likelihood agree with native propagation')
    return 0


if __name__ == '__main__':
    sys.exit(main())
