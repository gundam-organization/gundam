#!/usr/bin/env python3
"""Variable dial composition, automatic migration, smoothing, and Asimov copies."""
import json
import math
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import_module("005PythonSetup-utils").requireVenv(__file__)

RAW = [-0.25, 0.25, 1.25, 2.25]


def make_engine(path, conflict=False, disjoint=False, reverse=False, bad_mode=None, explicit_dynamic=False, backend=None):
    import GUNDAM

    def dial(mode=None, variable="X", condition=None):
        result = {"dialType": "Normalization"}
        if mode:
            result.update(applyToVariable=variable, variableResponseMode=mode)
        if condition:
            result["applyCondition"] = condition
        return result

    definitions = [
        ("multiply", 1.0, [dial("Multiply")]),
        ("add", 0.5, [dial("Add"), dial("Add", "aux")]),
        ("override", 1.25, [dial("Override", condition="category == 1")]),
        ("multiply2", 0.75, [dial("Multiply")]),
        ("add2", -0.25, [dial("Add")]),
        ("weight", 2.0, [dial()]),
    ]
    # The default mode is Multiply; mode names are case insensitive.
    definitions[3][2][0].pop("variableResponseMode")
    definitions[2][2][0]["variableResponseMode"] = "override"
    if conflict or disjoint:
        definitions.append(("otherOverride", -0.5, [
            dial("Override", condition="category == 0" if disjoint else "category == 1")]))
    definitions[1][2][1].update(minDialResponse=-0.1, maxDialResponse=0.6)
    if bad_mode:
        definitions[0][2][0]["variableResponseMode"] = bad_mode
    if reverse:
        definitions.reverse()
    parameters = [{"name": name, "priorValue": prior, "priorType": "Flat", "dialSetDefinitions": dials}
                  for name, prior, dials in definitions]
    binning = {"binningDefinition": [{"name": "X", "edges": [0, 1, 2]}]}
    samples = [
        {"name": "hard", "binning": binning},
        {"name": "smooth", "binning": binning, "binningResponse": {
            "isEnabled": True, "variables": [{"name": "X", "width": 0.2}]}},
        {"name": "static", "useDynamicBinning": explicit_dynamic,
         "binning": {"binningDefinition": [{"name": "Y", "edges": [0, 1]}]}},
    ]
    config = {"likelihoodInterfaceConfig": {
        "jointProbabilityConfig": {"type": "LeastSquares", "lsqPoissonianApproximation": True},
        "dataSetList": [{"name": "inputs", "model": {
            "tree": "events", "filePathList": [str(path)], "allowMultipleSamplesPerEntry": True,
            "nominalWeightFormula": "weight", "variableDict": [{"name": "X", "expr": "raw + 0"}],
        }}],
        "propagatorConfig": {
            "globalEventReweightCap": 2.5,
            "sampleSetConfig": {"sampleList": samples},
            "parametersManagerConfig": {"parameterSetList": [{
                "name": "response", "isEnabled": True, "nominalStepSize": 0.1, "parameterDefinitions": parameters}]}},
    }}
    if backend:
        config["likelihoodInterfaceConfig"]["backendManagerConfig"] = {"isEnabled": True, "type": backend}
    config.update(enablePca=False, scaleParStepWithChi2Response=False,
                  generateSamplePlots=False, generateOneSigmaPlots=False)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(json.dumps(config))
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(builder.getConfig()))
    engine.configure()
    propagator = engine.getLikelihoodInterface().getModelPropagator()
    engine.initialize()
    hard, smooth, static = propagator.getSampleSet().getSampleList()
    assert hard.isUseDynamicBinning() and smooth.isUseDynamicBinning()
    assert static.isUseDynamicBinning() == explicit_dynamic
    assert not hard.getHistogram().isBinningResponseEnabled()
    return engine


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def check(propagator, values, aux_shift, weight):
    weight = min(weight, 2.5)
    for sample in propagator.getSampleSet().getSampleList():
        events = sample.getEventList()
        assert len(events) == len(values), "Initially unbinned events must survive loading"
        histogram = sample.getHistogram()
        sums = [0.] * histogram.getNbBins()
        squares = [0.] * histogram.getNbBins()
        outside = 0.
        for event in events:
            entry = event.getIndices().entry
            close(event.getVariables().fetchVariable("X").getVarAsDouble(), values[entry])
            close(event.getVariables().fetchVariable("aux").getVarAsDouble(), 10 + entry + max(-0.1, min(aux_shift, 0.6)))
            close(event.getEventWeight(), (entry + 1) * weight)
            expected_bin = 0 if sample.getName() == "static" else (
                int(values[entry]) if 0 <= values[entry] < 2 else -1)
            assert event.getIndices().bin == expected_bin
            def cumulative(position):
                if position <= -1:
                    return 0.
                if position >= 1:
                    return 1.
                t = (position + 1) / 2
                return t**3 * (10 + t * (-15 + 6 * t))

            fractions = []
            for index in range(histogram.getNbBins()):
                if histogram.isBinningResponseEnabled():
                    fraction = (cumulative((index + 1 - values[entry]) / .2)
                                - cumulative((index - values[entry]) / .2))
                else:
                    fraction = float(index == expected_bin)
                fractions.append(fraction)
                contribution = (entry + 1) * weight * fraction
                sums[index] += contribution
                squares[index] += contribution**2
            outside += (entry + 1) * weight * (1 - sum(fractions))
        for index, content in enumerate(histogram.getBinContentList()):
            close(content.sumWeights, sums[index])
            close(content.sqrtSumSqWeights, math.sqrt(squares[index]))
        if histogram.isBinningResponseEnabled():
            close(histogram.getOutsideBinContent().sumWeights, outside)


def exercise(path, threads, reverse=False, disjoint=False):
    import GUNDAM

    GUNDAM.setNumberOfThreads(threads)
    engine = make_engine(path, reverse=reverse, disjoint=disjoint)
    likelihood = engine.getLikelihoodInterface()
    propagator = likelihood.getModelPropagator()
    parameters = {parameter.getName(): parameter for parameter in
                  propagator.getParametersManager().getParameterSetsList()[0].getParameterList()}

    def expected(mult=1., add=.5, override=1.25, mult2=.75, add2=-.25):
        return [(override if entry % 2 else (-.5 if disjoint else raw)) * mult * mult2 + add + add2
                for entry, raw in enumerate(RAW)]

    prior = expected()
    check(propagator, prior, .5, 2.)
    # The initial Asimov must include the variable responses, not the loaded bins.
    check(likelihood.getDataPropagator(), prior, .5, 2.)
    for mult, add, override, weight in [(2., -.25, -.5, 1.5), (1., .75, 1., 3.), (1., .5, 1.25, 2.)]:
        for name, value in [("multiply", mult), ("add", add), ("override", override), ("weight", weight)]:
            parameters[name].setParameterValue(value)
        for _ in range(2):
            likelihood.propagateAndEvalLikelihood()
            check(propagator, expected(mult, add, override), add, weight)
            check(likelihood.getDataPropagator(), prior, .5, 2.)


def write_input(path, count=4):
    import uproot

    with uproot.recreate(path) as output:
        tree = output.mktree("events", {"raw": "float64", "Y": "float64", "aux": "int32",
                                         "category": "int32", "weight": "float64"})
        tree.extend({"raw": array("d", RAW * (count // len(RAW))), "Y": array("d", [.5] * count),
                     "aux": array("i", range(10, 10 + count)),
                     "category": array("i", [entry % 2 for entry in range(count)]),
                     "weight": array("d", range(1, count + 1))})


def exercise_cache(path, threads):
    import GUNDAM

    GUNDAM.setNumberOfThreads(threads)
    engine = make_engine(path, explicit_dynamic=True)
    likelihood = engine.getLikelihoodInterface()
    propagator = likelihood.getModelPropagator()
    hard, smooth, dynamic = propagator.getSampleSet().getSampleList()
    parameters = {parameter.getName(): parameter for parameter in
                  propagator.getParametersManager().getParameterSetsList()[0].getParameterList()}
    prior = [(1.25 if entry % 2 else raw) * .75 + .25 for entry, raw in enumerate(RAW)]

    # Deliberately alter stored state as a probe: a skipped pass must leave it
    # untouched. Histogram contributions must still come from the cached bins.
    probes = []
    for sample in (hard, smooth):
        event = sample.getEventList()[0]
        probes.append((event, event.getIndices().bin))
        event.getVariables().fetchVariable("X").setValue(123.)
        event.getIndices().bin = -123
    for weight in (2., 1.5, 2.):
        parameters["weight"].setParameterValue(weight)
        likelihood.propagateAndEvalLikelihood()
        for event, original_bin in probes:
            close(event.getVariables().fetchVariable("X").getVarAsDouble(), 123.)
            assert event.getIndices().bin == -123, "Unchanged migration must skip rebinning"
        # Restore the probes to check the cached histogram sums and smoothing.
        for event, original_bin in probes:
            event.getVariables().fetchVariable("X").setValue(prior[0])
            event.getIndices().bin = original_bin
        check(propagator, prior, .5, weight)
        for event, original_bin in probes:
            event.getVariables().fetchVariable("X").setValue(123.)
            event.getIndices().bin = -123

    # An explicit buffer invalidation must force variable application even
    # though all parameter values are unchanged.
    for entry in propagator.getEventDialCache().getCache():
        for response in entry.getDialResponseCacheList():
            response.getDialInterface().getInputBuffer().invalidateBuffers()
    likelihood.propagateAndEvalLikelihood()
    check(propagator, prior, .5, 2.)

    # Multiple buffer updates before propagation must not lose a pending change.
    for mult in (2., 1.):
        parameters["multiply"].setParameterValue(mult)
        for entry in propagator.getEventDialCache().getCache():
            for response in entry.getDialResponseCacheList():
                buffer = response.getDialInterface().getInputBuffer()
                buffer.update()
                buffer.update()
        likelihood.propagateAndEvalLikelihood()
        expected = [(1.25 if entry % 2 else raw) * mult * .75 + .25
                    for entry, raw in enumerate(RAW)]
        check(propagator, expected, .5, 2.)

    # Explicit dynamic samples must rebin even with unchanged migration inputs.
    event = dynamic.getEventList()[0]
    event.getVariables().fetchVariable("Y").setValue(2.)
    likelihood.propagateAndEvalLikelihood()
    assert event.getIndices().bin == -1
    close(dynamic.getHistogram().getBinContentList()[0].sumWeights, 18.)
    event.getVariables().fetchVariable("Y").setValue(.5)
    likelihood.propagateAndEvalLikelihood()
    check(propagator, prior, .5, 2.)


def main():
    import GUNDAM

    print("Testing module:", GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / "219VariableDials.root"
    write_input(path)
    for threads in [1, 4, 8]:
        exercise(path, threads)
        exercise_cache(path, threads)
    exercise(path, 2, reverse=True)
    exercise(path, 2, disjoint=True)
    # The reader runs independently even with one filler thread. Alternating
    # conditions must use this event, never the reader's next entry.
    stress_path = Path.cwd() / "219VariableDials-conditions.root"
    write_input(stress_path, 512)
    for threads in [1, 4]:
        GUNDAM.setNumberOfThreads(threads)
        engine = make_engine(stress_path)
        prior = [(1.25 if entry % 2 else RAW[entry % 4]) * .75 + .25 for entry in range(512)]
        check(engine.getLikelihoodInterface().getModelPropagator(), prior, .5, 2.)
        check(engine.getLikelihoodInterface().getDataPropagator(), prior, .5, 2.)
    for options, message in [({"conflict": True}, "Multiple Override"),
                             ({"bad_mode": "Invalid"}, "Invalid")]:
        try:
            make_engine(path, **options)
        except RuntimeError as error:
            assert message.lower() in str(error).lower(), str(error)
        else:
            raise AssertionError(f"Expected rejection: {options}")
    print("SUCCESS: variable responses compose, migrate, smooth, and preserve Asimov events.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
