#!/usr/bin/env python3
"""Retain unbinned events and binning variables only when requested by a sample."""
import math
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import_module("005PythonSetup-utils").requireVenv(__file__)


def write_input(path):
    import uproot

    with uproot.recreate(path) as output:
        tree = output.mktree("events", {"raw": "float64", "weight": "float64"})
        tree.extend({
            "raw": array("d", [-1, 0.25, 0.75, 2, 3]),
            "weight": array("d", [1, 2, 0, 3, 4]),
        })


def evaluate(path, sample_specs, threads):
    import GUNDAM

    GUNDAM.setNumberOfThreads(threads)
    samples_yaml = ""
    for name, dynamic, selection, sample_weight in sample_specs:
        samples_yaml += f"""
        - name: {name}
          binning:
            binningDefinition:
              - {{name: X, edges: [10, 11]}}
              - {{name: Y, edges: [25, 40]}}
          selectionCutStr: "{selection}"
          sampleWeightFormula: "{sample_weight}"
"""
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(f"""
likelihoodInterfaceConfig:
  jointProbabilityConfig:
    type: LeastSquares
    lsqPoissonianApproximation: true
  backendManagerConfig:
    isEnabled: false
  dataSetList:
    - name: inputs
      model:
        tree: events
        allowMultipleSamplesPerEntry: true
        filePathList: ["{path}"]
        nominalWeightFormula: "weight"
        variableDict:
          - {{name: X, expr: "raw + 10"}}
          - {{name: Y, expr: "2 * raw + 30"}}
  propagatorConfig:
    sampleSetConfig:
      sampleList:
{samples_yaml}
""")
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(builder.getConfig()))
    engine.configure()
    likelihood = engine.getLikelihoodInterface()
    propagator = likelihood.getModelPropagator()
    samples = propagator.getSampleSet().getSampleList()
    for sample, (_, dynamic, _, _) in zip(samples, sample_specs):
        assert not sample.isUseDynamicBinning(), "Dynamic binning must default to false"
        sample.setUseDynamicBinning(dynamic)
        assert sample.isUseDynamicBinning() == dynamic

    engine.initialize()
    likelihood.propagateAndEvalLikelihood()
    stored_names = set(propagator.getSampleSet().getEventVariableNameList())
    if any(spec[1] for spec in sample_specs):
        assert {"X", "Y"} <= stored_names, stored_names
    else:
        assert not ({"X", "Y"} & stored_names), stored_names

    raw_values = {0: -1, 1: 0.25, 3: 2}
    weights = {0: 1, 1: 2, 3: 3}
    for sample, (_, dynamic, selection, sample_weight) in zip(samples, sample_specs):
        selected = [0] if selection == "raw < 0" else [0, 1, 3]
        expected = selected if dynamic else [entry for entry in selected if entry == 1]
        if sample_weight == "0":
            expected = []
        events = sample.getEventList()
        assert sorted(event.getIndices().entry for event in events) == expected, sample.getSummary()
        assert sample.getNbBinnedEvents() == (1 if 1 in expected else 0)
        for event in events:
            entry = event.getIndices().entry
            assert event.getIndices().bin == (0 if entry == 1 else -1)
            assert event.getWeights().base == weights[entry]
            if dynamic:
                variables = event.getVariables()
                assert variables.fetchVariable("X").getVarAsDouble() == raw_values[entry] + 10
                assert variables.fetchVariable("Y").getVarAsDouble() == 2 * raw_values[entry] + 30
        content = sample.getHistogram().getBinContentList()[0]
        expected_sum = 2 if 1 in expected else 0
        assert math.isclose(content.sumWeights, expected_sum)
        assert math.isclose(content.sqrtSumSqWeights, expected_sum)

    assert len(propagator.getEventDialCache().getCache()) == sum(
        len(sample.getEventList()) for sample in samples
    ), "Every retained event must have a dial cache entry"


def main():
    import GUNDAM

    print("Testing module:", GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / "216DynamicBinningLoading.root"
    write_input(path)
    static = ("static", False, "raw < 3", "1")
    dynamic = ("dynamic", True, "raw < 3", "1")
    zero = ("zero", True, "raw < 3", "0")
    outside = ("outside", True, "raw < 0", "1")
    for threads in [1, 2]:
        for specs in [[static], [dynamic], [static, dynamic, zero], [outside]]:
            evaluate(path, specs, threads)
    print("SUCCESS: dynamic samples retain unbinned events and binning variables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
