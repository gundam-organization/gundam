# Gradient performance on existing inputs

`GradientPerformance.py` compares the original engine and the CPU backend using
`evalFitGradientAt`, the callback used by the minimizer. It accepts the fitter's
`-c`, `-of` (ordered overrides), `-t` and `-a` options. Without `-a`, it selects
`RealData`, just like `gundamFitter`. No fit or HESSE is run.

Run with the Python interpreter and GUNDAM module matching the build to measure.
The script does not silently select a build. For representative timings, use an
optimized build; `RelWithDebInfo` also keeps useful symbols for profiling. The
report records the module path and build flags when its CMake cache is available.

For the upgrade inputs, from any working directory:

```bash
repo=/Users/nadrino/Documents/Work/Repositories/gundam
env -u DYLD_LIBRARY_PATH python3 "$repo/tests/python/benchmarks/GradientPerformance.py" \
  --python-module-dir "$repo/cmake-build-debug/src/PythonInterface" \
  --work-dir /Users/nadrino/Documents/Work/Output/results/gundam/upgrade-cc-2026 \
  -c ndUpCcInc.yaml \
  -of override/datasetHighlandv5.27.1_CCLyon.yaml override/ebMigrationDial.yaml \
      override/binResponse.yaml override/gradientTest.yaml override/backendCpuTest.yaml \
  -t 4 -a \
  --output /tmp/upgrade-gradient-performance-smoothed
```

`override/binResponse.yaml` enables Pmu bin smoothing with a 1 MeV width.
It is required for the migration gradient in this example; neither the script
nor `gradientTest.yaml` enables smoothing automatically.

Replace the module directory with the optimized build directory when available.
Use a matching Python version for that module. Clearing `DYLD_LIBRARY_PATH`
avoids accidentally loading an older installed GUNDAM library on macOS. Keep
ROOT and the input environment variables available. The output directory must
be new. Relative configuration and input paths resolve from `--work-dir`, which
defaults to the invocation directory.

## What is measured

Each engine runs in a separate subprocess, sequentially, with the same random
seed and resolved physics configuration. The script overrides backend selection
in both cases, even if an override file enables CPU. It enables analytic
gradients, disables plots and LLH history, and disables CPU automatic host
materialization. Input fit coordinates, likelihood, bin smoothing, parameter
selection and covariance are retained. Unsupported analytic configurations fail
explicitly and leave logs/reports.

The starting point and seven small deterministic perturbations are used by
default (`--points 8 --amplitude .01`). Perturbations are in parameter standard
deviations, with the parameter step as fallback. Components outside the allowed
parameter domain retain their initial value. The reports retain the actual
points and parameter names, so unchanged or domain-limited components remain
visible. Reduce amplitude if points leave the likelihood domain.

Initialization, initial value/gradient comparisons and warmup are excluded from
measurements. Five trials (`--trials`) alternate value/gradient measurement
order, with three repetitions (`--repeats`) per trial. The workloads are:

- `varying`: cycle through all parameter points, exercising invalidation.
- `unchanged`: repeatedly evaluate the last point, exposing cache reuse.

Each trial records wall time and process CPU time per callback. Process CPU time
includes all worker threads; it can exceed wall time. Median wall times determine
the reported CPU speedup. Timings include Python argument/result conversion and
the complete callback: propagation, derivative computation and fit-coordinate /
penalty integration. Backend callbacks currently evaluate a scalar likelihood
inside the gradient path; the original gradient path does not. This is the
actual callback cost, not an equalized derivative kernel microbenchmark.

A separate untimed pass collects CPU propagation stage clocks: dial caching,
observable updates, bin search, weights, histogram construction and likelihood.
**These clocks do not partition the entire gradient calculation.** Some clocks
are inclusive/overlapping; do not sum every field or call the unaccounted time
an isolated derivative measurement. Use the native profile for the derivative
work, including fraction Jacobians and event accumulation.

Before timing, the script saves value and full gradient vectors at every point.
After both processes finish, it compares workload metadata, data histograms,
values and gradients (`--rtol 3e-5 --atol 3e-6`). A mismatch produces a nonzero
exit status; timings remain available for investigation. This is a comparison
between implementations, not a finite-difference validation.

## Native profiling

By default, after all timing trials, each worker runs a separate ten-second
pass of varying gradient callbacks. On macOS it attaches `/usr/bin/sample` to
itself; on Linux it attaches `perf record` and generates a text report. Adjust
with `--profile-seconds 20`, or disable with `--profile none`. There is no
profiler attached during the primary timing trials.

The macOS report includes native call trees for every thread and loaded binary
images. Look for `CpuBackend::getLikelihoodGradient`, `binningGradient`,
`nodeGradient` / `compositeGradient`, thread-pool jobs, and the original engine's
`LikelihoodInterface::evalStatLikelihoodGradient`,
`BinningResponse::evalGradient` and `EventDialCache` gradient methods. In optimized
builds, some helpers are inlined and appear under their caller. Waiting threads
also appear: inclusive stack counts and counts from different threads are not
additive percentages of elapsed time. Sampling identifies hot code; it does not
supply exact per-stage wall times. On Linux, perf availability and permissions
are recorded; profiler failure does not invalidate the unprofiled timings.

## Outputs to inspect or share

- `summary.json`: completion/status, source revision, timings, speedups, numerical
  comparison, and profile status for both cases.
- `original/report.json` and `cpu/report.json`: workload, module/build information,
  initialization time, raw trials, propagation stage clocks, points, values and
  gradients. Updated after each trial; failures retain their traceback.
- `original/config.json` and `cpu/config.json`: resolved input configuration.
- `original.log` and `cpu.log`: initialization and execution logs.
- `original/gradient-sample.txt` and `cpu/gradient-sample.txt` on macOS; or
  `gradient-perf.data` and `gradient-perf-report.txt` in each case on Linux.
- Each case's `profile.json` and `profile.log`: sampling status and diagnostics.

Keep the whole directory for analysis. Both callbacks are synchronous at return.
There is no machine-dependent performance pass/fail threshold. An incomplete or
failed case remains marked as such; failed comparisons must be investigated
before interpreting speedup as equivalent work.

## ROOT/Clad compatibility for the original engine

With the tested ROOT 6.40.04 installation, Clad rejects the unqualified
`min([shift], [cap])` in the Eb composite's `cappedShift` node while generating
its gradient (a `min<double>` custom derivative signature mismatch). Use
`TMath::Min([shift], [cap])` in those formula definitions. That spelling
generates an analytic ROOT gradient and is also supported by the CPU formula
compiler. The benchmark does not rewrite formulas automatically. The CPU
backend differentiates bytecode independently of Clad and can finish even
when the original-engine case fails.

## CPU preparation optimizations

The manager caches structural gradient validation for a prepared layout, with
invalidation on reactivation/rebuild, deactivation, data updates and interface
replacement. Producer checks remain per-request to arm publication of current
external/table Jacobians. The manager publishes only those producer rows, and
CPU retains compact snapshots; no external Jacobian rows are built for ordinary
dials. These changes remove repeated full-model validation and empty dense-row
copies. The local dial-derivative loop is still serial and is not cached.

Rebuild before comparing revisions, keep the build type and workload identical,
and use a new output directory for each measurement. The earlier Debug timings
are a baseline for that build only; they do not predict the Release speedup.
