#!/usr/bin/env python3
"""Check relative boundary widths, conservation and the 1D two-bin limit."""
import json
import math
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast-tests"))
import_module("005PythonSetup-utils").requireVenv(__file__)
response_test = import_module("218BinningResponse")


def grid_binning(x_edges, y_edges=None):
    dimensions = [{"name": "X", "edges": x_edges}]
    if y_edges is not None:
        dimensions.append({"name": "Y", "edges": y_edges})
    return {"binningDefinition": dimensions}


def make_bin(x_bounds, y_bounds=None, category=None):
    edges = [{"name": "X", "bounds": x_bounds}]
    if y_bounds is not None:
        edges.append({"name": "Y", "bounds": y_bounds})
    if category is not None:
        edges.append({"name": "category", "bounds": [category, category + 1]})
    return {"edgesList": edges}


def make_engine(path, binning, variables, threads=1, initialize=True):
    import GUNDAM

    GUNDAM.setNumberOfThreads(threads)
    config = {"likelihoodInterfaceConfig": {
        "jointProbabilityConfig": {"type": "LeastSquares", "lsqPoissonianApproximation": True},
        "dataSetList": [{"name": "events", "model": {
            "tree": "events", "filePathList": [str(path)], "nominalWeightFormula": "weight",
        }}],
        "propagatorConfig": {"sampleSetConfig": {"sampleList": [{
            "name": "sample", "useDynamicBinning": True, "binning": binning,
            "binningResponse": {"isEnabled": True, "variables": variables},
        }]}},
    }}
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.configure()
    if initialize:
        engine.initialize()
    sample = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()[0]
    return engine, sample


def response_at(sample, x, y=None, category=None):
    event = sample.getEventList()[0]
    event.getVariables().fetchVariable("X").setValue(float(x))
    if y is not None:
        event.getVariables().fetchVariable("Y").setValue(float(y))
    if category is not None:
        event.getVariables().fetchVariable("category").setValue(float(category))
    response = sample.getHistogram().evaluateEventBinningResponse(event)
    assert 0 <= response.outsideFraction <= 1
    assert all(0 <= entry.fraction <= 1 for entry in response.binFractions)
    assert math.isclose(sum(entry.fraction for entry in response.binFractions) + response.outsideFraction, 1., abs_tol=2e-15)
    return response


def check_cached_histogram(engine, sample):
    engine.getLikelihoodInterface().getModelPropagator().updateEventSampleBinning()
    engine.getLikelihoodInterface().propagateAndEvalLikelihood()
    response_test.check_histogram(sample)


def check_configuration(path):
    invalid = [({}, "exactly one"), ({"width": 0.2, "widthFraction": 0.1}, "exactly one"),
               ({"widthFraction": 0}, "widthFraction must be in"),
               ({"widthFraction": -0.1}, "widthFraction must be in"),
               ({"widthFraction": 0.500001}, "widthFraction must be in")]
    for options, message in invalid:
        try:
            make_engine(path, grid_binning([0, 1, 2]), [{"name": "X", **options}], initialize=False)
        except RuntimeError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError(f"Accepted invalid width configuration: {options}")


def check_one_dimension(path):
    import numpy as np

    edges = [0, 0.01, 1, 1.001, 3]
    for fraction in [0.1, 0.5]:
        for threads in [1, 2, 4, 8]:
            engine, sample = make_engine(path, grid_binning(edges), [{"name": "X", "widthFraction": fraction}], threads)
            spacings = np.diff(edges)
            for index, boundary in enumerate(edges):
                spacing = min(spacings[max(0, index - 1)], spacings[min(index, len(spacings) - 1)])
                width = fraction * spacing
                for x in np.linspace(boundary - 2 * width, boundary + 2 * width, 21):
                    response = response_at(sample, x)
                    assert sum(entry.fraction > 1e-14 for entry in response.binFractions) <= 2
                response = response_at(sample, boundary)
                if 0 < index < len(edges) - 1:
                    assert len(response.binFractions) == 2
                    assert all(math.isclose(entry.fraction, 0.5, abs_tol=1e-12) for entry in response.binFractions)
                    left_index = next(context.bin.getIndex() for context in sample.getHistogram().getBinContextList()
                                      if context.bin.getEdgesList()[0].min == edges[index - 1])
                    step = width * 1e-3
                    lower = response_at(sample, boundary - step)
                    upper = response_at(sample, boundary + step)
                    left_lower = next(entry.fraction for entry in lower.binFractions if entry.binIndex == left_index)
                    left_upper = next(entry.fraction for entry in upper.binFractions if entry.binIndex == left_index)
                    assert math.isclose((left_upper - left_lower) / (2 * step), -15 / (16 * width), rel_tol=3e-6)
                    assert abs(left_upper - 1 + left_lower) < 1e-11
                else:
                    assert math.isclose(response.outsideFraction, 0.5, abs_tol=1e-12)
                check_cached_histogram(engine, sample)
            for lower, upper in zip(edges, edges[1:]):
                response = response_at(sample, (lower + upper) / 2)
                assert math.isclose(max(entry.fraction for entry in response.binFractions), 1., abs_tol=1e-12)
            assert response_at(sample, 10).outsideFraction == 1
            check_cached_histogram(engine, sample)


def check_absolute_equivalence(path):
    import numpy as np

    for mixed in [False, True]:
        binning = grid_binning([0, 1, 2, 3], [0, 1, 2] if mixed else None)
        absolute_variables = [{"name": "X", "width": 0.25}]
        relative_variables = [{"name": "X", "widthFraction": 0.25}]
        if mixed:
            absolute_variables.append({"name": "Y", "width": 0.3})
            relative_variables.append({"name": "Y", "width": 0.3})
        absolute_engine, absolute = make_engine(path, binning, absolute_variables)
        relative_engine, relative = make_engine(path, binning, relative_variables)
        for x in np.linspace(-0.5, 3.5, 121):
            for y in ([0.1, 1, 2.1] if mixed else [None]):
                first = response_at(absolute, x, y)
                second = response_at(relative, x, y)
                assert first.outsideFraction == second.outsideFraction
                assert [(entry.binIndex, entry.fraction) for entry in first.binFractions] == [
                    (entry.binIndex, entry.fraction) for entry in second.binFractions]
        check_cached_histogram(absolute_engine, absolute)
        check_cached_histogram(relative_engine, relative)


def check_two_dimensions(path):
    import numpy as np

    staggered = [make_bin([0, 1], [0, 1]), make_bin([1, 3], [0, 1]),
                 make_bin([0, 0.5], [1, 2]), make_bin([0.5, 2], [1, 2])]
    variables = [{"name": "X", "widthFraction": 0.5}, {"name": "Y", "widthFraction": 0.5}]
    reference = None
    for threads in [1, 2, 4, 8]:
        engine, sample = make_engine(path, {"binList": staggered}, variables, threads)
        values = []
        for y in np.linspace(-0.5, 2.5, 17):
            for x in np.linspace(-0.5, 3.5, 23):
                response = response_at(sample, x, y)
                assert sum(entry.fraction > 1e-14 for entry in response.binFractions) <= 4
                values.append([response.outsideFraction] + [entry.fraction for entry in response.binFractions])
        if reference is None:
            reference = values
        else:
            assert values == reference
        response = response_at(sample, 2, 1)
        assert math.isclose(response.outsideFraction, 0.25, abs_tol=1e-12)
        assert sorted(entry.fraction for entry in response.binFractions) == [0.25, 0.5]
        check_cached_histogram(engine, sample)
        response_at(sample, 5, 5)
        check_cached_histogram(engine, sample)


def check_selection_groups(path):
    bins = [make_bin([0, 1], category=0), make_bin([1, 2], category=0),
            make_bin([0, 0.01], category=1), make_bin([0.01, 0.02], category=1)]
    engine, sample = make_engine(path, {"binList": bins}, [{"name": "X", "widthFraction": 0.5}])
    response = response_at(sample, 0.75, category=0)
    assert math.isclose(response.binFractions[0].fraction, 0.896484375, abs_tol=1e-12)
    assert response.outsideFraction == 0
    check_cached_histogram(engine, sample)
    response = response_at(sample, 0.01, category=1)
    assert [entry.fraction for entry in response.binFractions] == [0.5, 0.5]
    check_cached_histogram(engine, sample)

    bins = [make_bin([0, 1], [0, 2]), make_bin([1, 2], [0, 1]), make_bin([1, 3], [1, 2])]
    engine, sample = make_engine(path, {"binList": bins}, [{"name": "X", "widthFraction": 0.5}])
    for y in [0.5, 1.5]:
        response = response_at(sample, 1, y)
        assert [entry.fraction for entry in response.binFractions] == [0.5, 0.5]
        assert response.outsideFraction == 0
        check_cached_histogram(engine, sample)


def plot_comparison(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    edges = [0, 0.1, 1, 1.1, 3]
    absolute_engine, absolute = make_engine(path, grid_binning(edges), [{"name": "X", "width": 0.2}])
    relative_engine, relative = make_engine(path, grid_binning(edges), [{"name": "X", "widthFraction": 0.25}])
    positions = np.linspace(0.75, 1.35, 601)
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for sample, style, label in [(absolute, "--", "absolute: 0.2"), (relative, "-", "fraction: 0.25")]:
        values = []
        counts = []
        for position in positions:
            response = response_at(sample, position)
            fractions = [0.] * (len(edges) - 1)
            for entry in response.binFractions:
                fractions[entry.binIndex] = entry.fraction
            values.append(fractions)
            counts.append(sum(value > 1e-14 for value in fractions))
        values = np.asarray(values)
        for index in range(len(edges) - 1):
            axes[0].plot(positions, values[:, index], style, color=f"C{index}", label=f"{label}, bin {index}")
        axes[1].plot(positions, counts, style, label=label)
    assert len(response_at(absolute, 1.05).binFractions) >= 3
    assert len(response_at(relative, 1.05).binFractions) == 1
    axes[0].set_ylabel("Fraction of event weight")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Event X", ylabel="Contributing bins", yticks=[1, 2, 3, 4])
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
        for edge in edges:
            axis.axvline(edge, color="gray", linewidth=0.5)
    axes[1].set_xlim(positions[0], positions[-1])
    figure.tight_layout()
    for extension in ["png", "pdf"]:
        figure.savefig(f"220BinningWidthFraction.{extension}")
    plt.close(figure)


def main():
    import GUNDAM
    import uproot

    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / "220BinningWidthFraction.root"
    with uproot.recreate(path) as output:
        tree = output.mktree("events", {name: "float64" for name in ["X", "Y", "category", "weight"]})
        tree.extend({"X": array("d", [0.5, 0.5]), "Y": array("d", [0.5, 0.5]),
                     "category": array("d", [0, 1]), "weight": array("d", [3, 5])})
    check_configuration(path)
    check_one_dimension(path)
    check_absolute_equivalence(path)
    check_two_dimensions(path)
    check_selection_groups(path)
    plot_comparison(path)
    print("SUCCESS: relative widths conserve weights, support irregular domains and limit 1D responses to two bins.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
