#!/usr/bin/env python3
"""Check centered response smoothing and save propagated scans as PNG/PDF/CSV."""
import csv
import json
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import_module("005PythonSetup-utils").requireVenv(__file__)


def write_input(path):
    # uproot does not write TGraph object branches. Keep ROOT in a subprocess.
    macro = Path.cwd() / "229DialResponseSmoothing-input.C"
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TSystem.h>
#include <cmath>
{
 {
  TFile output(OUTPUT_PATH, "RECREATE");
  TTree tree("events", "events");
  double coordinate = 0.5;
  TGraph graphObject, flatObject;
  TGraph *graph = &graphObject, *flat = &flatObject;
  tree.Branch("X", &coordinate, "X/D");
  tree.Branch("graph", "TGraph", &graph, 32000, 0);
  tree.Branch("flat", "TGraph", &flat, 32000, 0);
  int index = 0;
  for (double x : {-1., 0., 1., 2., 3.}) {
    graph->SetPoint(index, x, std::abs(x));
    flat->SetPoint(index++, x, 1.);
  }
  tree.Fill();
  tree.Write();
 }
 gSystem->Exit(0);
}
'''.replace("OUTPUT_PATH", json.dumps(str(path))))
    subprocess.run(["root", "-n", "-l", "-b", "-q", str(macro)], check=True)


def make_engine(path, options, dial_type="Normalization", branch="graph"):
    import GUNDAM

    dial = {"dialType": dial_type, **options}
    if dial_type != "Normalization":
        dial.update(dialBranchData=branch, allowDialExtrapolation=True)
    config = {"likelihoodInterfaceConfig": {
        "jointProbabilityConfig": {"type": "LeastSquares", "lsqPoissonianApproximation": True},
        "backendManagerConfig": {"isEnabled": False},
        "dataSetList": [{"name": "input", "model": {"tree": "events", "filePathList": [str(path)]}}],
        "propagatorConfig": {
            "sampleSetConfig": {"sampleList": [{"name": "sample", "binning": {
                "binningDefinition": [{"name": "X", "edges": [0, 1]}]}}]},
            "parametersManagerConfig": {"parameterSetList": [{"name": "response", "isEnabled": True,
                "nominalStepSize": 0.1, "parameterDefinitions": [{"name": "p", "priorValue": 1.,
                    "priorType": "Flat", "dialSetDefinitions": [dial]}]}]}}},
        "enablePca": False, "scaleParStepWithChi2Response": False,
        "generateSamplePlots": False, "generateOneSigmaPlots": False}
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(json.dumps(config))
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(builder.getConfig()))
    engine.configure()
    engine.initialize()
    return engine


def lower_reference(value, bound, width):
    if value <= bound - width:
        return bound
    if value >= bound + width:
        return value
    # Independent integral of the cubic slope 3t² - 2t³.
    t = (value - bound + width) / (2 * width)
    return bound + width * (2 * t**3 - t**4)


def exercise(path, options, minimum, maximum, widths=None, dial_type="Normalization", branch="graph"):
    engine = make_engine(path, options, dial_type, branch)
    likelihood = engine.getLikelihoodInterface()
    propagator = likelihood.getModelPropagator()
    parameter = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
    sample = propagator.getSampleSet().getSampleList()[0]
    for value in [-1., -.4, -.2, 0., .2, .4, .8, 1., 1.8, 1.9, 2., 2.1, 2.5]:
        raw = (1. if branch == "flat" else abs(value)) if dial_type == "Graph" else value
        expected = raw
        if minimum is not None:
            expected = lower_reference(expected, minimum, widths[0]) if widths else max(minimum, expected)
        if maximum is not None:
            expected = maximum - lower_reference(maximum - expected, 0., widths[1]) if widths else min(maximum, expected)
        parameter.setParameterValue(value)
        for _ in range(2):  # Repeated propagation must reuse the processed response.
            likelihood.propagateAndEvalLikelihood()
            actual = sample.getHistogram().getBinContentList()[0].sumWeights
            assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (options, value, actual, expected)


def plot_scans(path):
    import GUNDAM
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    GUNDAM.setNumberOfThreads(1)
    figure, axes = plt.subplots(3, 3, figsize=(15, 10), layout="constrained")
    cases = [("symmetric", "Normalization", 0.25, (0.25, 0.25), "Shared width: 0.25"),
             ("asymmetric", "Normalization", [0.4, 0.1], (0.4, 0.1), "Separate widths: [0.4, 0.1]"),
             ("v-shape", "Graph", [0.4, 0.1], (0.4, 0.1), "V-shaped dial: w = |parameter|")]
    csv_path = Path.cwd() / "229DialResponseSmoothing.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["case", "parameter", "raw_response", "hard_clamp", "smooth_reference", "gundam_response"])
        for column, (name, dial_type, width_config, widths, title) in enumerate(cases):
            options = {"dialResponseLimits": [0., 2.], "enableResponseSmoothing": True,
                       "dialResponseSmoothingWidth": width_config}
            engine = make_engine(path, options, dial_type)
            likelihood = engine.getLikelihoodInterface()
            propagator = likelihood.getModelPropagator()
            parameter = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()[0]
            sample = propagator.getSampleSet().getSampleList()[0]
            positions = np.linspace(-2.5 if dial_type == "Graph" else -0.5, 2.5,
                                    2501 if dial_type == "Graph" else 1501)
            raw = np.abs(positions) if dial_type == "Graph" else positions
            hard = np.clip(raw, 0., 2.)
            reference, actual = [], []
            for position, value, hard_value in zip(positions, raw, hard):
                expected = lower_reference(value, 0., widths[0])
                expected = 2. - lower_reference(2. - expected, 0., widths[1])
                parameter.setParameterValue(float(position))
                likelihood.propagateAndEvalLikelihood()
                result = sample.getHistogram().getBinContentList()[0].sumWeights
                assert math.isclose(result, expected, rel_tol=1e-10, abs_tol=1e-12), (name, position, result, expected)
                reference.append(expected)
                actual.append(result)
                writer.writerow([name, position, value, hard_value, expected, result])

            actual = np.asarray(actual)
            reference = np.asarray(reference)
            axes[0, column].plot(positions, raw, color="0.75", label="Raw dial")
            axes[0, column].plot(positions, hard, "--", color="0.4", label="Hard clamp")
            axes[0, column].plot(positions, reference, color="C0", linewidth=2.5, label="Smooth reference")
            axes[0, column].plot(positions[::25], actual[::25], ".", color="C1", markersize=4,
                                 label="GUNDAM propagation")
            axes[0, column].set_title(title)
            axes[0, column].set_ylim(-0.3, 2.7)
            axes[0, column].legend(fontsize=8)
            for values, color, style, label in [(hard, "0.4", "--", "Hard clamp"),
                                                (reference, "C0", "-", "Reference"),
                                                (actual, "C1", ":", "GUNDAM")]:
                first = np.gradient(values, positions, edge_order=2)
                second = np.gradient(first, positions, edge_order=2)
                # A centered response map retains the V's own derivative jump
                # at parameter zero. Do not display a numerical delta spike as
                # if it were a finite second derivative at that point.
                if dial_type == "Graph":
                    mask = np.abs(positions) < 0.006
                    first[mask], second[mask] = np.nan, np.nan
                axes[1, column].plot(positions, first, style, color=color, label=label)
                if label != "Hard clamp":
                    axes[2, column].plot(positions, second, style, color=color, label=label)
            if dial_type == "Graph":
                axes[1, column].annotate("Input V kink remains at zero", xy=(0., 0.5), xytext=(-2.3, 0.8),
                                         arrowprops={"arrowstyle": "->"}, fontsize=9)
            for row in range(3):
                axes[row, column].grid(alpha=0.2)
                axes[row, column].set_xlabel("Dial parameter")
            axes[1, column].set_ylim(-1.15, 1.15)
    for row, label in enumerate(["Returned response", "First derivative (numerical)", "Second derivative (numerical)"]):
        axes[row, 0].set_ylabel(label)
    figure.suptitle("Original GUNDAM engine: centered C2 response smoothing, limits [0, 2]", fontsize=16)
    for extension in ["png", "pdf"]:
        plot_path = Path.cwd() / f"229DialResponseSmoothing.{extension}"
        figure.savefig(plot_path, dpi=160)
        print("Saved plot:", plot_path, flush=True)
    plt.close(figure)
    print("Saved scan data:", csv_path, flush=True)


def main():
    import GUNDAM

    print("Testing module:", GUNDAM.__file__, flush=True)
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / "229DialResponseSmoothing.root"
    write_input(path)
    for threads in [1, 4]:
        GUNDAM.setNumberOfThreads(threads)
        for width in [0.1, [0.4, 0.1], {"min": 0.4, "max": 0.1}]:
            widths = (0.1, 0.1) if isinstance(width, float) else (0.4, 0.1)
            options = {"dialResponseLimits": [0., 2.], "enableResponseSmoothing": True,
                       "dialResponseSmoothingWidth": width}
            exercise(path, options, 0., 2., widths)
        exercise(path, {"dialResponseLimits": [0., 2.], "dialResponseSmoothingWidth": [0.4, 0.1]}, 0., 2.)
        exercise(path, {"dialResponseLimits": [0., 2.], "enableResponseSmoothing": False,
                        "dialResponseSmoothingWidth": [0.4, 0.1]}, 0., 2.)
        exercise(path, {"minDialResponse": 0., "maxDialResponse": 2., "enableResponseSmoothing": True,
                        "dialResponseSmoothingWidth": [0.4, 0.1]}, 0., 2., (0.4, 0.1))
        exercise(path, {"minDialResponse": 0.2, "maxDialResponse": 1.8}, 0.2, 1.8)
        exercise(path, {"minimumSplineResponse": 0.2}, 0.2, None)
        exercise(path, {"maxDialResponse": 1.8}, 0., 1.8)
        exercise(path, {"dialResponseLimits": {"max": 2.}, "enableResponseSmoothing": True,
                        "dialResponseSmoothingWidth": {"max": 0.1}}, None, 2., (None, 0.1))
        exercise(path, {"dialResponseLimits": {"min": 0.}, "enableResponseSmoothing": True,
                        "dialResponseSmoothingWidth": 0.4}, 0., None, (0.4, None), "Graph")
        exercise(path, {"dialResponseLimits": [0., 1.], "enableResponseSmoothing": True,
                        "dialResponseSmoothingWidth": 0.1}, 0., 1., (0.1, 0.1), "Graph", "flat")
        exercise(path, {"dialResponseLimits": [0., 2.], "enableResponseSmoothing": True,
                        "dialResponseSmoothingWidth": [0.4, 0.1],
                        "parametersBinningPath": {"binningDefinition": [{"name": "X", "edges": [0, 1]}]}},
                 0., 2., (0.4, 0.1))
    plot_scans(path)
    print("SUCCESS: original engine applies centered C2 response smoothing, including asymmetric widths and unity dials.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
