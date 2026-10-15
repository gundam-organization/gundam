#!/usr/bin/env python3
"""Plot hard and smooth bin migrations using the compiled Python interface."""
import csv
import math
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast-tests"))
import_module("005PythonSetup-utils").requireVenv(__file__)


def make_engine(path, threads, dynamic=True):
    import GUNDAM

    GUNDAM.setNumberOfThreads(threads)
    samples = []
    for dimensions in [1, 2]:
        for smooth in [False, True]:
            name = f"{'smooth' if smooth else 'hard'}{dimensions}d"
            y_binning = "\n              - {name: Y, edges: [0, 1, 2]}" if dimensions == 2 else ""
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
          binning:
            binningDefinition:
              - {{name: X, edges: [0, 1, 2.5, 3]}}{y_binning}
              - {{name: category, edges: [0, 1, 2]}}
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
    for event in sample.getEventList():
        fractions = histogram.evaluateBinningResponse(event)
        if histogram.isBinningResponseEnabled():
            assert math.isclose(sum(entry.fraction for entry in fractions), 1., abs_tol=2e-15)
        for entry in fractions:
            assert 0 <= entry.fraction <= 1
            weight = event.getEventWeight() * entry.fraction
            sums[entry.binIndex] += weight
            squares[entry.binIndex] += weight * weight
    for index, content in enumerate(histogram.getBinContentList()):
        assert math.isclose(content.sumWeights, sums[index], abs_tol=1e-12)
        assert math.isclose(content.sqrtSumSqWeights, math.sqrt(squares[index]), abs_tol=1e-12)
    if histogram.isBinningResponseEnabled():
        assert math.isclose(sum(sums), sample.getSumWeights(), abs_tol=1e-12)


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
        results[sample.getName()] = (keys, np.asarray(values))
    return results


def plot_results(positions, results):
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
            axes[0, column].plot(positions, smooth[:, index], color=color, label=f"bin lower edges {key[:dimensions]}")
        axes[0, column].set_title(f"{dimensions}D: solid = smooth, dashed = hard")
        axes[0, column].set_ylabel("Fraction of event weight")
        axes[0, column].legend(fontsize=8)
        axes[1, column].plot(positions, np.sum(hard, axis=1), "--", label="hard")
        axes[1, column].plot(positions, np.sum(smooth, axis=1), label="smooth")
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
        figure.savefig(f"218BinningResponse.{extension}")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, mode in zip(axes, ["hard", "smooth"]):
        values = results[f"{mode}1d"][1]
        target = np.asarray([0.2, 0.5, 0.3])
        objective = np.sum((values - target) ** 2, axis=1)
        axis.plot(positions, objective)
        axis.set(title=f"Quadratic objective: {mode}", xlabel="Event X", ylabel="Sum of squared residuals")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig("218BinningResponse-objective.png")
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
            assert sample.getNbBinnedEvents() == 2
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
    print("SUCCESS: conserved weights and histogram errors; generated 218BinningResponse plots and CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
