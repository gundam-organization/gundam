#!/usr/bin/env python3
"""Move stored event variables and reassign histogram bins after initialization."""
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
        tree = output.mktree("events", {
            "raw": "float64", "Y": "float32", "category": "int32",
            "weight": "float64", "large": "uint64",
        })
        tree.extend({
            "raw": array("d", [0.25, 1.25, -0.25, 2, 0.25]),
            "Y": array("f", [0, 0, 0, 0, 2]),
            "category": array("i", [0, 1, 0, 1, 1]),
            "weight": array("d", [1, 2, 3, 4, 5]),
            "large": array("Q", [2**53 + 1] * 5),
        })


def evaluate(path, threads):
    import GUNDAM

    GUNDAM.setNumberOfThreads(threads)
    binning = """{binningDefinition: [
            {name: X, edges: [0, 1, 2]},
            {name: Y, edges: [-1, 1]},
            {name: category, edges: [0, 1, 2]}]}"""
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
        nominalWeightFormula: weight
        additionalLeavesStorage: [large]
        variableDict:
          - {{name: X, expr: "raw + 0"}}
  propagatorConfig:
    sampleSetConfig:
      sampleList:
        - name: dynamic
          binning: {binning}
        - name: static
          binning: {binning}
        - name: empty
          binning: {binning}
          selectionCutStr: "raw > 100"
    parametersManagerConfig:
      parameterSetList:
        - name: norm
          isEnabled: true
          nominalStepSize: 0.1
          parameterDefinitions:
            - name: scale
              priorValue: 1.0
              priorType: Flat
              dialSetDefinitions:
                - dialType: Normalization
""")
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(builder.getConfig()))
    engine.configure()
    likelihood = engine.getLikelihoodInterface()
    propagator = likelihood.getModelPropagator()
    dynamic, static, empty = propagator.getSampleSet().getSampleList()
    dynamic.setUseDynamicBinning(True)
    empty.setUseDynamicBinning(True)
    engine.initialize()
    parameter = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    events = {event.getIndices().entry: event for event in dynamic.getEventList()}
    assert set(events) == set(range(5))
    assert len(empty.getEventList()) == 0

    # Identify bins by their lower edges, independently of their ordering.
    bin_indices = {}
    for context in dynamic.getHistogram().getBinContextList():
        edges = {edge.varName: edge.min for edge in context.bin.getEdgesList()}
        bin_indices[(edges["X"], int(edges["category"]))] = context.bin.getIndex()
    assert len(bin_indices) == 4
    original = {
        entry: {name: event.getVariables().fetchVariable(name).getValue()
                for name in ["X", "Y", "category"]}
        for entry, event in events.items()
    }
    identities = {
        entry: (event.getIndices().dataset, event.getIndices().sample, event.getIndices().treeEntry)
        for entry, event in events.items()
    }

    def set_value(entry, name, value):
        variable = events[entry].getVariables().fetchVariable(name)
        variable.setValue(value)
        assert variable.getValue() == value
        assert variable.getVarAsDouble() == float(value)

    def check(expected_bins, scale=1.0):
        likelihood.propagateAndEvalLikelihood()
        sums = [0.0] * 4
        squares = [0.0] * 4
        for entry, key in enumerate(expected_bins):
            event = events[entry]
            index = -1 if key is None else bin_indices[key]
            assert event.getIndices().bin == index, (entry, event.getIndices().bin, index)
            assert identities[entry] == (
                event.getIndices().dataset, event.getIndices().sample, event.getIndices().treeEntry
            )
            assert event.getWeights().base == entry + 1
            assert math.isclose(event.getEventWeight(), (entry + 1) * scale)
            if index != -1:
                sums[index] += (entry + 1) * scale
                squares[index] += ((entry + 1) * scale) ** 2
        assert dynamic.getNbBinnedEvents() == sum(key is not None for key in expected_bins)
        for index, content in enumerate(dynamic.getHistogram().getBinContentList()):
            assert math.isclose(content.sumWeights, sums[index], abs_tol=1e-12)
            assert math.isclose(content.sqrtSumSqWeights, math.sqrt(squares[index]), abs_tol=1e-12)
        # Changes to stored variables never migrate events in a static sample.
        assert [event.getIndices().bin for event in static.getEventList()] == [
            bin_indices[(0, 0)], bin_indices[(1, 1)]
        ]
        static_contents = static.getHistogram().getBinContentList()
        assert math.isclose(static_contents[bin_indices[(0, 0)]].sumWeights, scale)
        assert math.isclose(static_contents[bin_indices[(1, 1)]].sumWeights, 2 * scale)
        # All events, including unbinned ones, keep their normalization dial.
        cache = propagator.getEventDialCache().getCache()
        assert len(cache) == 7
        assert all(entry.getDialIndicesAffecting(parameter) == [0] for entry in cache)

    initial = [(0, 0), (1, 1), None, None, None]
    check(initial)
    propagator.updateEventSampleBinning()
    check(initial)  # Also exercises rebinding the loader's variable-index caches.

    # The setter preserves scalar types and exact 64-bit integer values.
    large = events[0].getVariables().fetchVariable("large")
    assert large.getValue() == 2**53 + 1
    large.setValue(2**63 + 123)
    assert large.getValue() == 2**63 + 123
    try:
        large.setValue(-1)
    except (RuntimeError, TypeError, OverflowError):
        pass
    else:
        raise AssertionError("Negative value accepted for an unsigned variable")
    assert large.getValue() == 2**63 + 123
    y = events[0].getVariables().fetchVariable("Y")
    y.setValue(0.1)
    assert math.isclose(y.getValue(), 0.1, abs_tol=1e-8)
    assert y.getValue() != 0.1, "Float32 variable was promoted to double"
    assert y.getVarAsDouble() == y.getValue()
    y.setValue(0.0)

    set_value(0, "X", 1.25)
    set_value(1, "X", -1.0)
    set_value(2, "X", 0.0)  # Lower edge is included.
    set_value(3, "X", 1.0)  # Internal boundary belongs to the next bin.
    set_value(4, "Y", -1.0)
    for event in static.getEventList():
        event.getVariables().fetchVariable("X").setValue(100.0)
    check(initial)  # Rebinning is explicit, rather than triggered by propagation.
    propagator.updateEventSampleBinning()
    check([(1, 0), None, (0, 0), (1, 1), (0, 1)])

    set_value(0, "category", 1)
    set_value(1, "X", 1.25)
    set_value(2, "X", 2.0)  # Upper edges are excluded.
    set_value(3, "Y", 1.0)
    set_value(4, "category", 2)
    dynamic.updateEventBinning()  # The sample API also rebuilds bin event lists.
    parameter.setParameterValue(1.25)
    check([(1, 1), (1, 1), None, None, None], scale=1.25)

    for entry in events:
        set_value(entry, "X", 2.0)
    propagator.updateEventSampleBinning()
    check([None] * 5, scale=1.25)  # Previously occupied bins must be emptied.

    for entry, values in original.items():
        for name, value in values.items():
            set_value(entry, name, value)
    parameter.setParameterValue(1.0)
    for _ in range(2):
        propagator.updateEventSampleBinning()
        check(initial)


def main():
    import GUNDAM

    print("Testing module:", GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / "217DynamicRebinning.root"
    write_input(path)
    # Include more workers than events or bins, as well as uneven partitions.
    for threads in [1, 2, 4, 8]:
        evaluate(path, threads)
    print("SUCCESS: rebinning updates indices, histogram contents and errors after event migration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
