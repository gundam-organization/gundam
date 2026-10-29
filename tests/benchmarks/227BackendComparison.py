#!/usr/bin/env python3
"""Compare original, CPU and MPS propagation on the same production configuration.

Loads one model per process. Measures likelihood evaluation separately from
histogram materialization, then compares every point, bin content and bin error.
Uses Asimov data without minimization, Hesse, plotting or parameter-step scans.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "scheduling", Path(__file__).with_name("222CpuWeightSchedulingBenchmark.py"))
scheduling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduling)


def measure(args):
    import GUNDAM
    GUNDAM.setRuntimeWorkingDirectory(str(args.work_dir))
    GUNDAM.setNumberOfThreads(args.threads)
    GUNDAM.setLightOutputMode(True)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfig(str(args.config))
    if args.override:
        builder.override(args.override)
    config = json.loads(builder.getConfig().toString())["fitterEngineConfig"]
    config.update(enablePca=False, scaleParStepWithChi2Response=False,
                  generateSamplePlots=False, generateOneSigmaPlots=False)
    backend = config["likelihoodInterfaceConfig"].setdefault("backendManagerConfig", {})
    backend.update(isEnabled=args.worker != "original", type=args.worker if args.worker != "original" else "CPU")
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    engine.initialize()
    manager = engine.getBackendsManager()
    manager.setEnableAutoMaterialize(False)
    if args.worker != "original" and (not manager.hasBackend() or manager.getType() != args.worker):
        raise RuntimeError(f"Expected an active {args.worker} backend")
    metadata = None
    if args.worker == "MPS":
        metadata = json.loads(manager.getDiagnosticMps())
        if not metadata["available"] or metadata["path"] == "hostFallback":
            raise RuntimeError(f"GPU propagation unavailable: {metadata}")
    sequence = scheduling.ParameterSequence(engine, args.step_scale)
    result = {"backend": args.worker, "module": GUNDAM.__file__, "model": metadata,
              "parameters": sequence.manifest(), "scenarios": []}
    for scenario in ("constant", "local1", "local2", "global"):
        sequence.reset()
        engine.evaluateLikelihood()
        rng, digest = random.Random(args.seed), hashlib.sha256()
        calls = scheduling.scenario_calls(args, scenario, len(sequence.entries))
        values, timings = [], []
        for call in range(calls):
            sequence.apply(scenario, rng, call, digest)
            start = time.perf_counter()
            engine.evaluateLikelihood()
            timings.append(time.perf_counter() - start)
            if manager.hasBackend():
                manager.materialize("Histograms")
            histogram = [(b.sumWeights, b.sqrtSumSqWeights)
                         for sample in engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
                         for b in sample.getHistogram().getBinContentList()]
            likelihood = engine.getLikelihoodInterface().getLastLikelihood()
            if not math.isfinite(likelihood) or any(not math.isfinite(v) for pair in histogram for v in pair):
                raise RuntimeError(f"Nonfinite result: {args.worker}, {scenario}, point {call}")
            values.append({"likelihood": likelihood, "histogram": histogram})
        result["scenarios"].append({"scenario": scenario, "digest": digest.hexdigest(),
                                    "medianMs": statistics.median(timings) * 1000,
                                    "seconds": timings, "values": values})
    scheduling.write_json(args.output / f"{args.worker}.json", result)


def compare(reference, candidate):
    gpu = candidate["backend"] == "MPS"
    relative, absolute = (1e-5, 1e-3) if gpu else (1e-10, 1e-8)
    if reference["parameters"] != candidate["parameters"]:
        raise RuntimeError("Parameter manifests differ")
    rows = []
    for original, current in zip(reference["scenarios"], candidate["scenarios"]):
        if original["digest"] != current["digest"] or len(original["values"]) != len(current["values"]):
            raise RuntimeError("Parameter sequences differ")
        max_likelihood, max_histogram, max_relative = 0., 0., 0.
        for index, (expected, actual) in enumerate(zip(original["values"], current["values"])):
            left, right = expected["likelihood"], actual["likelihood"]
            max_likelihood = max(max_likelihood, abs(left - right))
            if not math.isclose(left, right, rel_tol=relative, abs_tol=absolute):
                raise RuntimeError(f"Likelihood mismatch: {candidate['backend']}, {original['scenario']}, point {index}: {left} != {right}")
            if len(expected["histogram"]) != len(actual["histogram"]):
                raise RuntimeError("Histogram lengths differ")
            for expected_pair, actual_pair in zip(expected["histogram"], actual["histogram"]):
                for left, right in zip(expected_pair, actual_pair):
                    max_histogram = max(max_histogram, abs(left - right))
                    max_relative = max(max_relative, abs(left - right) / max(1., abs(left)))
                    if not math.isclose(left, right, rel_tol=relative, abs_tol=absolute):
                        raise RuntimeError(f"Histogram mismatch: {candidate['backend']}, {original['scenario']}, point {index}: {left} != {right}")
        rows.append({"scenario": original["scenario"], "originalMedianMs": original["medianMs"],
                     "backendMedianMs": current["medianMs"], "maxLikelihoodDifference": max_likelihood,
                     "maxHistogramDifference": max_histogram, "maxScaledHistogramDifference": max_relative})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--global-calls", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--step-scale", type=float, default=.1)
    parser.add_argument('--output', type=Path,
                        default=Path(f'benchmark-{Path(__file__).stem}-{time.strftime("%Y%m%d%H%M%S")}'),
                        help='Output directory (default: ./benchmark-<script-name>-YYYYMMDDHHmmss)')
    parser.add_argument("--worker", choices=["original", "CPU", "MPS"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    args.config = (args.work_dir / args.config).resolve()
    args.output = args.output.resolve()
    if min(args.threads, args.calls, args.global_calls) <= 0 or not math.isfinite(args.step_scale) or args.step_scale <= 0:
        parser.error("Counts and step scale must be positive and finite")
    if args.worker:
        measure(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    scheduling.write_json(args.output / "settings.json", {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    modes = ["original", "CPU"] + (["MPS"] if sys.platform == "darwin" else [])
    for mode in modes:
        scheduling.progress(f"Loading and evaluating {mode}; log: {args.output / (mode + '.log')}")
        with (args.output / f"{mode}.log").open("w") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", mode],
                                     stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        scheduling.progress(f"{mode} worker PID {child.pid} running")
            except KeyboardInterrupt:
                child.terminate()
                child.wait()
                raise
        if code:
            raise RuntimeError(f"{mode} exited with code {code}; see {args.output / (mode + '.log')}")
    reference = json.loads((args.output / "original.json").read_text())
    comparisons = {mode: compare(reference, json.loads((args.output / f"{mode}.json").read_text())) for mode in modes[1:]}
    scheduling.write_json(args.output / "summary.json", comparisons)
    scheduling.progress(f"SUCCESS: every measured point agrees; results: {args.output}")


if __name__ == "__main__":
    main()
