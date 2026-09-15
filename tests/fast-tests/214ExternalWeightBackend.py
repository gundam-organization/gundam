#!/usr/bin/env python3
"""Exercise the actual fitter backend path with two independent Python workers."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys


def assert_close(label, actual, expected, tolerance):
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"{label}: {actual} != {expected}")


def main():
    import GUNDAM

    if not hasattr(GUNDAM.FitterEngine, "getBackendsManager"):
        print("SKIP: GUNDAM was built without backends")
        return 0
    script_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("external_fixture", script_dir / "213ExternalWeightResponse.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    work = Path.cwd()
    root_path = work / "214ExternalWeightBackend.root"
    script_path = work / "214ExternalWeightBackend_worker.py"
    fixture.write_input_root_file(root_path)
    fixture.write_external_weight_script(script_path)
    GUNDAM.setRuntimeWorkingDirectory(str(work))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(2)

    def run(backend, binned):
        builder = GUNDAM.ConfigUtils.ConfigBuilder()
        builder.setConfigFromYamlString(fixture.build_config_text(root_path, script_path, sys.executable, binned))
        config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
        config.update(enablePca=False, scaleParStepWithChi2Response=False,
                      generateSamplePlots=False, generateOneSigmaPlots=False)
        likelihood_config = config['likelihoodInterfaceConfig']
        if backend:
            likelihood_config['backendManagerConfig'] = {'isEnabled': True, 'type': backend}
        sets = likelihood_config['propagatorConfig']['parametersManagerConfig']['parameterSetList']
        second = copy.deepcopy(sets[0])
        second['name'] = 'SecondExternalSource'
        sets.append(second)
        # Apply per-dial limits before multiplying the two external responses.
        sets[0]['dialSetDefinitions'][0]['minDialResponse'] = 0.02
        sets[0]['dialSetDefinitions'][0]['maxDialResponse'] = 0.8
        sets.append({
            'name': 'Normalization', 'isEnabled': True, 'nominalStepSize': 0.1,
            'parameterDefinitions': [{'name': 'scale', 'priorValue': 1.0, 'priorType': 'Flat'}],
            'dialSetDefinitions': [{'dialType': 'Normalization', 'dialInputList': [{'name': 'scale'}]}],
        })
        engine = GUNDAM.FitterEngine()
        engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
        engine.initialize()
        li = engine.getLikelihoodInterface()
        manager = engine.getBackendsManager()
        if backend and (not manager.hasBackend() or manager.getType() != backend):
            raise RuntimeError('Requested backend was not initialized')
        parameters = li.getModelPropagator().getParametersManager().getParameterSetsList()
        samples = li.getModelPropagator().getSampleSet().getSampleList()
        results = []
        # Repeated point, first source only, normalization only, second source only.
        points = [(0.2, 0.2, 1.), (0.2, 0.2, 1.), (0.4, 0.2, 1.),
                  (0.4, 0.2, 2.), (0.4, 0.6, 2.), (0.4, 0.6, 2.)]
        for i, (first, second, scale) in enumerate(points):
            parameters[0].getParameterList()[1].setParameterValue(first)
            parameters[1].getParameterList()[1].setParameterValue(second)
            parameters[2].getParameterList()[0].setParameterValue(scale)
            engine.evaluateLikelihood()
            sample = samples[0]
            ordered = sorted((e.getIndices().treeEntry, float(e.getEventWeight())) for e in sample.getEventList())
            weights = [weight for _, weight in ordered]
            bins = sample.getHistogram().getBinContentList()
            sums = [float(b.sumWeights) for b in bins]
            squares = [float(b.sqrtSumSqWeights)**2 for b in bins]
            results.append(weights + sums + squares + [float(li.getLastLikelihood())])
            if backend == 'MPS':
                timing = manager.getLastTimingSummary()
                changed = 1 if i in (2, 4) else 0
                count = 12 if binned else 6
                if timing.externalWeightUploadBlocks != changed or timing.externalWeightUploadBytes != changed * count * 4:
                    raise RuntimeError(f'Unexpected MPS transfers at point {i}: {timing.externalWeightUploadBlocks} blocks / {timing.externalWeightUploadBytes} bytes')
            # Validate against the analytic producer formula, independent of backends.
            energies = [0.6, 0.8, 1.2, 1.5, 2., 2.5]
            for j, energy in enumerate(energies):
                probability = math.sin(1.267 * 2.5e-3 * 295. / energy)**2
                a = 1. - first * probability if j % 2 else first * probability
                b = 1. - second * probability if j % 2 else second * probability
                expected = scale * min(0.8, max(0.02, a)) * b
                assert_close(f'{backend} event {j}', weights[j], expected, 2e-6 if backend == 'MPS' else 1e-10)
            GUNDAM.flushOutput()
        return results

    backends = ['CPU']
    if sys.platform == 'darwin':
        backends.append('MPS')
    for binned in (False, True):
        reference = run(None, binned)
        for backend in backends:
            result = run(backend, binned)
            for point, (actual, expected) in enumerate(zip(result, reference)):
                for index, (a, b) in enumerate(zip(actual, expected)):
                    assert_close(f'{backend} binned={binned} point={point} output={index}', a, b,
                                 2e-5 if backend == 'MPS' else 1e-10)
    print('SUCCESS: external weights, histograms, likelihood and selective block uploads agree')
    return 0


if __name__ == '__main__':
    sys.exit(main())
