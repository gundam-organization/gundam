#!/usr/bin/env python3
"""Plot hard and smooth bin migrations using the compiled Python interface."""
import csv
import json
import math
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast-tests"))
import_module("005PythonSetup-utils").requireVenv(__file__)


def make_engine(path, threads, dynamic=True, l_shaped=False):
    import GUNDAM

    GUNDAM.setNumberOfThreads(threads)
    samples = []
    for dimensions in [1, 2]:
        for smooth in [False, True]:
            name = f"{'smooth' if smooth else 'hard'}{dimensions}d"
            binning = {"binningDefinition": [
                {"name": "X", "edges": [0, 1, 2.5, 3]},
                {"name": "category", "edges": [0, 1, 2]},
            ]}
            if dimensions == 2:
                binning["binningDefinition"].append({"name": "Y", "edges": [0, 1, 2]})
            if l_shaped and dimensions == 2:
                bins = []
                for category in [0, 1]:
                    for y in [0, 1]:
                        for lower, upper in [(0, 1), (1, 2.5), (2.5, 3)]:
                            if y == 1 and lower == 2.5:
                                continue
                            bins.append({"edgesList": [
                                {"name": "X", "bounds": [lower, upper]},
                                {"name": "Y", "bounds": [y, y + 1]},
                                {"name": "category", "bounds": [category, category + 1]},
                            ]})
                binning = {"binList": bins}
            response = "\n          binningResponse: {isEnabled: false}\n" if dimensions == 1 else ""
            if smooth:
                response = """
          binningResponse:
            isEnabled: true
            variables:
              - {name: X, width: 0.2}
"""
                if dimensions == 2:
                    response += "              - {name: Y, width: 0.3}\n"
            samples.append(f"""
        - name: {name}
          useDynamicBinning: {str(dynamic).lower()}
          binning: {json.dumps(binning)}
{response}""")
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(f"""
likelihoodInterfaceConfig:
  jointProbabilityConfig:
    type: LeastSquares
    lsqPoissonianApproximation: true
  dataSetList:
    - name: events
      model:
        tree: events
        allowMultipleSamplesPerEntry: true
        filePathList: ["{path}"]
        nominalWeightFormula: weight
  propagatorConfig:
    sampleSetConfig:
      sampleList:
{''.join(samples)}
""")
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(builder.getConfig()))
    engine.configure()
    engine.initialize()
    return engine


def bin_indices(sample):
    result = {}
    for context in sample.getHistogram().getBinContextList():
        edges = {edge.varName: edge.min for edge in context.bin.getEdgesList()}
        result[(edges["X"], edges.get("Y", 0), int(edges["category"]))] = context.bin.getIndex()
    return result


def check_histogram(sample):
    histogram = sample.getHistogram()
    sums = [0.] * histogram.getNbBins()
    squares = [0.] * len(sums)
    outside_sum, outside_square = 0., 0.
    for event in sample.getEventList():
        response = histogram.evaluateEventBinningResponse(event)
        fractions = response.binFractions
        assert 0 <= response.outsideFraction <= 1
        assert math.isclose(sum(entry.fraction for entry in fractions) + response.outsideFraction, 1., abs_tol=2e-15)
        for entry in fractions:
            assert 0 <= entry.fraction <= 1
            weight = event.getEventWeight() * entry.fraction
            sums[entry.binIndex] += weight
            squares[entry.binIndex] += weight * weight
        if histogram.isBinningResponseEnabled():
            weight = event.getEventWeight() * response.outsideFraction
            outside_sum += weight
            outside_square += weight * weight
    for index, content in enumerate(histogram.getBinContentList()):
        assert math.isclose(content.sumWeights, sums[index], abs_tol=1e-12)
        assert math.isclose(content.sqrtSumSqWeights, math.sqrt(squares[index]), abs_tol=1e-12)
    outside = histogram.getOutsideBinContent()
    assert math.isclose(outside.sumWeights, outside_sum, abs_tol=1e-12)
    assert math.isclose(outside.sqrtSumSqWeights, math.sqrt(outside_square), abs_tol=1e-12)
    if histogram.isBinningResponseEnabled():
        assert math.isclose(sum(sums) + outside.sumWeights, sample.getSumWeights(), abs_tol=1e-12)


def scan(engine, positions):
    import numpy as np

    samples = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
    results = {}
    for sample in samples:
        event = sample.getEventList()[0]
        if "2d" in sample.getName():
            event.getVariables().fetchVariable("Y").setValue(1.)
        indices = bin_indices(sample)
        keys = sorted(key for key in indices if key[2] == 0)
        values = []
        for scan_index, position in enumerate(positions):
            event.getVariables().fetchVariable("X").setValue(float(position))
            sample.updateEventBinning()
            sample.getHistogram().refillHistogram()
            if scan_index in [0, len(positions) // 2, len(positions) - 1]:
                engine.getLikelihoodInterface().propagateAndEvalLikelihood()
            check_histogram(sample)
            content = sample.getHistogram().getBinContentList()
            values.append([content[indices[key]].sumWeights / event.getEventWeight() for key in keys])
            values[-1].append(sample.getHistogram().evaluateEventBinningResponse(event).outsideFraction)
        results[sample.getName()] = (keys + [(None, None, 0)], np.asarray(values))
    return results


def plot_results(positions, results, suffix=""):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    for column, dimensions in enumerate([1, 2]):
        hard_keys, hard = results[f"hard{dimensions}d"]
        smooth_keys, smooth = results[f"smooth{dimensions}d"]
        assert hard_keys == smooth_keys
        for index, key in enumerate(smooth_keys):
            color = f"C{index}"
            axes[0, column].plot(positions, hard[:, index], "--", color=color, alpha=0.6)
            axes[0, column].plot(positions, smooth[:, index], color=color, label="outside" if key[0] is None else f"bin lower edges {key[:dimensions]}")
        axes[0, column].set_title(f"{dimensions}D: solid = smooth, dashed = hard")
        axes[0, column].set_ylabel("Fraction of event weight")
        axes[0, column].legend(fontsize=8)
        axes[1, column].plot(positions, np.sum(hard[:, :-1], axis=1), "--", label="hard: inside")
        axes[1, column].plot(positions, np.sum(smooth[:, :-1], axis=1), label="smooth: inside")
        axes[1, column].plot(positions, np.sum(smooth, axis=1), ":", color="black", label="smooth: inside + outside")
        axes[1, column].set_ylabel("Sum of fractions")
        axes[1, column].legend()
        first_x_bins = [index for index, key in enumerate(smooth_keys) if key[0] == 0]
        axes[2, column].plot(positions, np.gradient(hard[:, first_x_bins].sum(axis=1), positions), "--", label="hard")
        axes[2, column].plot(positions, np.gradient(smooth[:, first_x_bins].sum(axis=1), positions), label="smooth")
        axes[2, column].set_ylabel("Derivative of first X-bin fraction")
        axes[2, column].set_xlabel("Event X (Y = 1 for 2D)")
        for row in range(3):
            axes[row, column].grid(alpha=0.25)
            for edge in [0, 1, 2.5, 3]:
                axes[row, column].axvline(edge, color="gray", linewidth=0.5)
    figure.tight_layout()
    for extension in ["png", "pdf"]:
        figure.savefig(f"218BinningResponse{suffix}.{extension}")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, mode in zip(axes, ["hard", "smooth"]):
        values = results[f"{mode}1d"][1][:, :-1]
        target = np.asarray([0.2, 0.5, 0.3])
        objective = np.sum((values - target) ** 2, axis=1)
        axis.plot(positions, objective)
        axis.set(title=f"Quadratic objective: {mode}", xlabel="Event X", ylabel="Sum of squared residuals")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(f"218BinningResponse{suffix}-objective.png")
    plt.close(figure)


def check_l_shaped(engine, positions):

    propagator = engine.getLikelihoodInterface().getModelPropagator()
    sample = next(sample for sample in propagator.getSampleSet().getSampleList() if sample.getName() == "smooth2d")
    histogram = sample.getHistogram()
    event = sample.getEventList()[0]
    x = event.getVariables().fetchVariable("X")
    y = event.getVariables().fetchVariable("Y")
    for position, expected_outside in [((0.5, 0.5), 0.), ((2.5, 1.), 0.25),
                                       ((3., 0.), 0.75), ((2.75, 1.5), 1.), ((4., 3.), 1.)]:
        x.setValue(position[0])
        y.setValue(position[1])
        propagator.updateEventSampleBinning()
        engine.getLikelihoodInterface().propagateAndEvalLikelihood()
        check_histogram(sample)
        response = histogram.evaluateEventBinningResponse(event)
        assert math.isclose(response.outsideFraction, expected_outside, abs_tol=1e-12)
        if position == (2.5, 1.):
            assert len(response.binFractions) == 3
            assert all(math.isclose(entry.fraction, 0.25, abs_tol=1e-12) for entry in response.binFractions)

    def outside_on_diagonal(displacement):
        x.setValue(2.5 + displacement)
        y.setValue(1. + displacement)
        return histogram.evaluateEventBinningResponse(event).outsideFraction

    step = 1e-4
    lower, center, upper = [outside_on_diagonal(position) for position in [-step, 0., step]]
    assert math.isclose((upper - lower) / (2 * step), 3.90625, rel_tol=1e-5)
    assert math.isclose((upper - 2 * center + lower) / step**2, 29.296875, rel_tol=1e-4)
    x.setValue(0.5)
    y.setValue(0.5)
    category = event.getVariables().fetchVariable("category")
    category.setValue(3.)
    propagator.updateEventSampleBinning()
    engine.getLikelihoodInterface().propagateAndEvalLikelihood()
    check_histogram(sample)
    assert histogram.evaluateEventBinningResponse(event).outsideFraction == 1.
    category.setValue(0.)
    x.setValue(2.5)
    y.setValue(1.)
    event.getWeights().current = -3.
    propagator.updateEventSampleBinning()
    histogram.refillHistogram()
    check_histogram(sample)
    event.getWeights().current = 3.
    return scan(engine, positions)


def plot_l_shaped_domain(engine):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    samples = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
    sample = next(sample for sample in samples if sample.getName() == "smooth2d")
    histogram = sample.getHistogram()
    event = sample.getEventList()[0]
    x = event.getVariables().fetchVariable("X")
    y = event.getVariables().fetchVariable("Y")
    original = (x.getValue(), y.getValue())
    x_values = np.linspace(-0.4, 3.4, 121)
    y_values = np.linspace(-0.4, 2.4, 101)
    outside = np.empty((len(y_values), len(x_values)))
    for row, y_value in enumerate(y_values):
        y.setValue(float(y_value))
        for column, x_value in enumerate(x_values):
            x.setValue(float(x_value))
            outside[row, column] = histogram.evaluateEventBinningResponse(event).outsideFraction
    x.setValue(original[0])
    y.setValue(original[1])
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    hard = np.zeros_like(outside)
    rectangles = []
    for context in histogram.getBinContextList():
        edges = {edge.varName: edge for edge in context.bin.getEdgesList()}
        if edges["category"].min != 0:
            continue
        left, right = edges["X"].min, edges["X"].max
        bottom, top = edges["Y"].min, edges["Y"].max
        hard[(mesh_x >= left) & (mesh_x < right) & (mesh_y >= bottom) & (mesh_y < top)] = 1.
        rectangles.append((left, bottom, right - left, top - bottom))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for axis, values, title in zip(axes, [hard, 1 - outside, outside],
                                  ["Hard: inside", "Smooth: inside", "Smooth: outside"]):
        mesh = axis.pcolormesh(x_values, y_values, values, shading="auto", vmin=0, vmax=1)
        for rectangle in rectangles:
            axis.add_patch(Rectangle(rectangle[:2], rectangle[2], rectangle[3], fill=False, edgecolor="white", linewidth=0.7))
        axis.set(title=title, xlabel="Event X", ylabel="Event Y", aspect="equal")
    figure.colorbar(mesh, ax=list(axes), label="Fraction of event weight", shrink=0.8)
    figure.savefig("218BinningResponse-L-domain.png", bbox_inches="tight")
    plt.close(figure)


def main():
    import GUNDAM
    import numpy as np
    import uproot

    print("Testing module:", GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / "218BinningResponse.root"
    with uproot.recreate(path) as output:
        tree = output.mktree("events", {name: "float64" for name in ["X", "Y", "category", "weight"]})
        tree.extend({"X": array("d", [0.5, 0.5]), "Y": array("d", [0.5, 0.5]),
                     "category": array("d", [0, 1]), "weight": array("d", [3, 5])})
    positions = np.linspace(-0.5, 3.5, 801)
    engine = make_engine(path, 1)
    results = scan(engine, positions)
    for threads in [2, 4, 8]:
        threaded_engine = make_engine(path, threads)
        threaded = scan(threaded_engine, [-0.5, 0.8, 1., 1.2, 2.5, 3.5])
        for name, (_, values) in threaded.items():
            indices = [int(np.argmin(abs(positions - value))) for value in [-0.5, 0.8, 1., 1.2, 2.5, 3.5]]
            assert np.allclose(values, results[name][1][indices], atol=1e-12)
    outside_path = Path.cwd() / "218BinningResponse-outside.root"
    with uproot.recreate(outside_path) as output:
        tree = output.mktree("events", {name: "float64" for name in ["X", "Y", "category", "weight"]})
        tree.extend({"X": array("d", [-2, 4]), "Y": array("d", [0.5, 0.5]),
                     "category": array("d", [0, 1]), "weight": array("d", [3, 5])})
    static_engine = make_engine(outside_path, 2, dynamic=False)
    samples = static_engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
    for sample in samples:
        assert not sample.isUseDynamicBinning()
        if sample.getHistogram().isBinningResponseEnabled():
            assert len(sample.getEventList()) == 2
            assert sample.getNbBinnedEvents() == 0
            assert all(event.getIndices().bin == -1 for event in sample.getEventList())
            check_histogram(sample)
        else:
            assert len(sample.getEventList()) == 0

    with open("218BinningResponse.csv", "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["X"] + [f"{name}_{key}" for name, (keys, _) in results.items() for key in keys])
        for index, position in enumerate(positions):
            writer.writerow([position] + [value for _, values in results.values() for value in values[index]])
    plot_results(positions, results)
    for threads in [1, 2, 4, 8]:
        l_engine = make_engine(path, threads, l_shaped=True)
        scan_indices = [0, 260, 300, 340, 600, 800]
        l_results = check_l_shaped(l_engine, positions if threads == 1 else positions[scan_indices])
        if threads == 1:
            l_reference = l_results
            plot_results(positions, l_results, suffix="-L")
            plot_l_shaped_domain(l_engine)
        else:
            for name, (_, values) in l_results.items():
                assert np.array_equal(values, l_reference[name][1][scan_indices])
    print("SUCCESS: conserved weights and histogram errors; generated 218BinningResponse plots and CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
