# Gradient benchmark on existing inputs

Run this script directly with a matching GUNDAM Python module, ROOT and the
input environment variables available. Each backend/data-mode pair runs in an
isolated process. The output directory must be new.

```bash
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface:$PYTHONPATH" \
  python3 tests/python/benchmarks/RealInputGradientBenchmark.py \
  --work-dir /path/to/inputs -c config.yaml \
  -of override/dataset.yaml override/gradient.yaml \
  --backends Original CPU --data-modes Asimov Toy \
  --threads 4 --seed 1234 --output /tmp/real-gradient
```

The script retains input physics, applies overrides in order, and then selects
the requested backend. It forces analytic gradients and normalized fit space,
disables HESSE and plot generation, and benchmarks the actual minimizer callbacks.
Relative configuration and input paths resolve from `--work-dir`.

`--points` (default 8) includes the initial point and deterministic perturbations
of at most `--amplitude` prior standard deviations (default 0.01). Out-of-domain
components stay at the starting value. For parameters without a prior standard
deviation, the minimizer's step size is used. Warmup and initialization are
outside the callback timers. Trials alternate value/gradient order. Use
`--trials`, `--repeats` and `--skip-fit` to control duration. Fits default to
a budget of 30,000 function evaluations; use `--max-fit-calls 0` to retain the
input configuration budget, or specify another positive budget. Minuit can
overshoot this budget inside its initialization; it is not a wall-time limit.

Each case writes the resolved config, a log and a JSON report, including loaded
module, workload, points, data histograms, raw callback timings and medians.
The fit starts from the original initial point and reports its convergence,
EDM, likelihood and call counts. Nonconvergence is recorded, not hidden.

Backend stage clocks are collected in separate untimed passes. They describe
propagation inside the callback, not the derivative-only stages, and must not
be interpreted as a full additive gradient profile. Initialization is reported
separately. Fit timing covers `minimize()`, including its bookkeeping; it is
not just the sum of isolated callback times. These performance measurements do
not themselves validate derivatives by finite differences.
