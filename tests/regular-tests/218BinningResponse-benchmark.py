"""Measure CPU rebinning and propagation separately; run explicitly with Python."""
import argparse
import json
import statistics
import sys
import time
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast-tests"))
import_module("005PythonSetup-utils").requireVenv(__file__)


def make_engine(path, bins):
    import GUNDAM
    import numpy as np

    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(f"""
likelihoodInterfaceConfig:
  jointProbabilityConfig:
    type: LeastSquares
    lsqPoissonianApproximation: true
  dataSetList:
    - name: inputs
      model:
        tree: events
        filePathList: ["{path}"]
        nominalWeightFormula: weight
  propagatorConfig:
    sampleSetConfig:
      sampleList:
        - name: smooth
          useDynamicBinning: true
          binning:
            binningDefinition:
              - {{name: X, edges: {np.linspace(0, 64, bins + 1).tolist()}}}
          binningResponse:
            isEnabled: true
            variables:
              - {{name: X, width: 0.2}}
""")
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(builder.getConfig()))
    engine.configure()
    engine.initialize()
    return engine


def main():
    import GUNDAM
    import numpy as np
    import uproot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=100000)
    parser.add_argument("--bins", type=int, nargs="+", default=[4, 64])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("218BinningResponse-benchmark.json"))
    args = parser.parse_args()
    if min([args.events, args.repeat] + args.bins + args.threads) <= 0:
        parser.error("Event, bin, thread and repetition counts must be positive")

    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / "218BinningResponse-benchmark.root"
    random = np.random.default_rng(724)
    with uproot.recreate(path) as output:
        tree = output.mktree("events", {"X": "float64", "weight": "float64"})
        tree.extend({"X": random.uniform(-1, 65, args.events), "weight": random.uniform(0.5, 2, args.events)})
    results = []
    for bins in args.bins:
        for threads in args.threads:
            GUNDAM.setNumberOfThreads(threads)
            engine = make_engine(path, bins)
            likelihood = engine.getLikelihoodInterface()
            propagator = likelihood.getModelPropagator()
            sample = propagator.getSampleSet().getSampleList()[0]
            propagator.updateEventSampleBinning()
            likelihood.propagateAndEvalLikelihood()
            rebinning_times, propagation_times = [], []
            for _ in range(args.repeat):
                start = time.perf_counter()
                propagator.updateEventSampleBinning()
                rebinning_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                likelihood.propagateAndEvalLikelihood()
                propagation_times.append(time.perf_counter() - start)
            results.append({
                "bins": bins, "events": args.events, "threads": threads,
                "rebinning_ms": statistics.median(rebinning_times) * 1000,
                "propagation_ms": statistics.median(propagation_times) * 1000,
                "sum_weights": [entry.sumWeights for entry in sample.getHistogram().getBinContentList()],
                "sqrt_sum_sq_weights": [entry.sqrtSumSqWeights for entry in sample.getHistogram().getBinContentList()],
            })
    args.output.write_text(json.dumps({"module": GUNDAM.__file__, "results": results}, indent=2) + "\n")
    print(f"Benchmark saved to {args.output.resolve()}", flush=True)
    for result in results:
        print(f"{result['bins']:4} bins, {result['threads']:2} threads: "
              f"rebinning {result['rebinning_ms']:.3f} ms, propagation {result['propagation_ms']:.3f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
