#!/usr/bin/env python3
"""Compare the actual CPU backend with variable migration and smoothed bins."""
import math
import json
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
fixture = import_module('219VariableDials')


def assert_same(actual, expected):
    assert len(actual) == len(expected)
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (i, a, b)


def capture(engine):
    li = engine.getLikelihoodInterface()
    result = [li.getLastLikelihood()]
    for sample in li.getModelPropagator().getSampleSet().getSampleList():
        for event in sample.getEventList():
            result += [event.getEventWeight(), event.getIndices().bin]
            result += [event.getVariables().fetchVariable(name).getVarAsDouble() for name in ('X', 'aux')]
        histogram = sample.getHistogram()
        for content in histogram.getBinContentList():
            result += [content.sumWeights, content.sqrtSumSqWeights]
        if histogram.isBinningResponseEnabled():
            outside = histogram.getOutsideBinContent()
            result += [outside.sumWeights, outside.sqrtSumSqWeights]
    return result


def exercise_manual_binning(path):
    import GUNDAM
    engines = []
    for backend in (False, True):
        li = {
            'jointProbabilityConfig': {'type': 'LeastSquares', 'lsqPoissonianApproximation': True},
            'dataSetList': [{'name': 'inputs', 'model': {
                'tree': 'events', 'filePathList': [str(path)], 'nominalWeightFormula': 'weight',
                'variableDict': [{'name': 'X', 'expr': 'raw + 0'}]}}],
            'propagatorConfig': {'sampleSetConfig': {'sampleList': [{
                'name': 'manual', 'useDynamicBinning': True,
                'binning': {'binningDefinition': [{'name': 'X', 'edges': [0, 1, 2]}]},
                'binningResponse': {'isEnabled': True, 'variables': [{'name': 'X', 'widthFraction': .3}]},
            }]}},
        }
        if backend:
            li['backendManagerConfig'] = {'isEnabled': True, 'type': 'CPU'}
        config = {'likelihoodInterfaceConfig': li, 'enablePca': False, 'scaleParStepWithChi2Response': False,
                  'generateSamplePlots': False, 'generateOneSigmaPlots': False}
        engine = GUNDAM.FitterEngine()
        engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
        engine.initialize()
        engine.evaluateLikelihood()
        engines.append(engine)
    def sums(engine):
        histogram = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()[0].getHistogram()
        return [b.sumWeights for b in histogram.getBinContentList()] + [histogram.getOutsideBinContent().sumWeights]
    initial = sums(engines[0])
    assert_same(sums(engines[1]), initial)
    for engine in engines:
        sample = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()[0]
        for event in sample.getEventList():
            event.getVariables().fetchVariable('X').setValue(1.5)
        engine.evaluateLikelihood()
        # Without variable dials, rebinning remains explicit.
        assert_same(sums(engine), initial)
        sample.updateEventBinning()
        engine.evaluateLikelihood()
    assert_same(sums(engines[1]), sums(engines[0]))
    assert_same(sums(engines[1]), [0., 10., 0.])


def main():
    import GUNDAM
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: built without backends')
        return
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '220CpuDynamicBinning.root'
    fixture.write_input(path)
    for threads in (1, 2):
        GUNDAM.setNumberOfThreads(threads)
        reference = fixture.make_engine(path, explicit_dynamic=True)
        backend = fixture.make_engine(path, explicit_dynamic=True, backend='CPU')
        assert backend.getBackendsManager().hasBackend()
        for point in ((1., .5, 1.25, 2.), (2., -.25, -.5, 1.5),
                      (2., -.25, -.5, 3.), (1., .5, 1.25, 2.)):
            for engine in (reference, backend):
                propagator = engine.getLikelihoodInterface().getModelPropagator()
                parameters = {p.getName(): p for p in propagator.getParametersManager().getParameterSetsList()[0].getParameterList()}
                for name, value in zip(('multiply', 'add', 'override', 'weight'), point):
                    parameters[name].setParameterValue(value)
                engine.evaluateLikelihood()
            assert_same(capture(backend), capture(reference))
        # Host edits in explicitly dynamic samples remain observable even when
        # all dial inputs are unchanged.
        for value in (2., .5):
            for engine in (reference, backend):
                sample = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()[2]
                sample.getEventList()[0].getVariables().fetchVariable('Y').setValue(value)
                engine.evaluateLikelihood()
            assert_same(capture(backend), capture(reference))
        # Fit mode keeps intermediates resident; explicit materialization must
        # recover the same variables, pointer lists and fractional histograms.
        backend.getBackendsManager().setEnableAutoMaterialize(False)
        for engine in (reference, backend):
            propagator = engine.getLikelihoodInterface().getModelPropagator()
            parameters = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()
            next(p for p in parameters if p.getName() == 'multiply').setParameterValue(2.)
            engine.evaluateLikelihood()
        assert_same([backend.getLikelihoodInterface().getLastLikelihood()],
                    [reference.getLikelihoodInterface().getLastLikelihood()])
        for output in ('EventState', 'EventWeights', 'Histograms'):
            backend.getBackendsManager().materialize(output)
        assert_same(capture(backend), capture(reference))
        backend.getBackendsManager().setEnableAutoMaterialize(True)
        # An explicitly dynamic sample must not invalidate the cached state
        # of automatic samples when only the event weights change.
        probes = []
        for engine in (reference, backend):
            samples = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
            for sample in list(samples)[:2]:
                event = sample.getEventList()[0]
                probes.append((event, event.getVariables().fetchVariable('X').getVarAsDouble(), event.getIndices().bin))
                event.getVariables().fetchVariable('X').setValue(123.)
                event.getIndices().bin = -123
            engine.evaluateLikelihood()
        for event, value, bin_index in probes:
            assert event.getVariables().fetchVariable('X').getVarAsDouble() == 123.
            assert event.getIndices().bin == -123
            event.getVariables().fetchVariable('X').setValue(value)
            event.getIndices().bin = bin_index
        # Unchanged automatic migration and weight-only changes reuse bins.
        cached = fixture.make_engine(path, backend='CPU')
        cached.evaluateLikelihood()
        cached.evaluateLikelihood()
        assert cached.getBackendsManager().getLastTimingSummary().rebinnedEventCount == 0
        parameters = cached.getLikelihoodInterface().getModelPropagator().getParametersManager().getParameterSetsList()[0].getParameterList()
        next(p for p in parameters if p.getName() == 'weight').setParameterValue(1.5)
        cached.evaluateLikelihood()
        assert cached.getBackendsManager().getLastTimingSummary().rebinnedEventCount == 0
        # Explicit dial input invalidation must be consumed even at identical values.
        propagator = cached.getLikelihoodInterface().getModelPropagator()
        for entry in propagator.getEventDialCache().getCache():
            for response in entry.getDialResponseCacheList():
                response.getDialInterface().getInputBuffer().invalidateBuffers()
        cached.evaluateLikelihood()
        # The legacy path must still work after backend propagation.
        before = capture(cached)
        cached.getLikelihoodInterface().propagateAndEvalLikelihood()
        assert_same(capture(cached), before)
    exercise_manual_binning(path)
    if sys.platform == 'darwin':
        try:
            fixture.make_engine(path, backend='MPS')
        except RuntimeError as error:
            assert 'MPS does not yet support' in str(error), str(error)
        else:
            raise AssertionError('MPS silently accepted unsupported migration')
    print('SUCCESS: CPU migration, variables, weights, errors, outside contributions and likelihood agree')


if __name__ == '__main__':
    main()
