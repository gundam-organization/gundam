# CPU analytic gradient benchmarks

`tests/benchmarks/236CpuGradientBenchmark.py` generates a deterministic Asimov
model using Gaussian Barlow-Beeston Lite, two event Graph weights and one
normalization per histogram bin. It alternates numerical and analytic Minuit2
fits, excluding input generation and engine initialization from fit timings.
Both modes start at the same displaced point. It checks convergence, finite
likelihood, EDM, fitted parameter agreement and actual gradient callback use.
There is no machine-dependent performance pass/fail threshold.

With the matching GUNDAM Python module on `PYTHONPATH` and ROOT on `PATH`:

```bash
python3 tests/benchmarks/236CpuGradientBenchmark.py \
  --events 120000 --bins 48 --threads 4 --trials 5 \
  --strategy 1 --output /tmp/cpu-gradient-strategy1
python3 tests/benchmarks/236CpuGradientBenchmark.py \
  --events 120000 --bins 48 --threads 4 --trials 5 \
  --strategy 0 --output /tmp/cpu-gradient-strategy0
```

Output directories must not already exist. Each contains the generated ROOT
fixture and `summary.json`, with individual fit times, value/gradient timings,
Minuit call counts, EDM, fitted values and median speedup. The module path is
recorded to help detect accidental use of an installed, older build.

## Measurements on 2026-10-31

Apple M4 Max, Release build, ROOT 6.40.04, four CPU threads, backends inactive.
The fixed synthetic model has 120,000 events and 50 fit parameters. The table
uses medians from five fits per mode, with alternating execution order:

| Minuit strategy | Numerical gradient | Analytic gradient | Speedup | Value calls, numerical / analytic |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.622 s | 0.484 s | 1.29× | 2,894 / 1,992 |
| 0 | 0.328 s | 0.110 s | 2.99× | 1,434 / 234 |

Analytic fits requested 12 gradients in both strategies. EDM values were below
`8e-10` in both modes. A subsequent three-trial run of the final script with
strategy 0 measured 0.321 s versus 0.109 s (2.95×).

The callback was initially dominated by serial event traversal, temporary
allocations per event and repeated linear parameter searches. Reusing scratch
buffers, sharing one indexed lookup and parallelizing the event derivative pass
reduced its median time from about 1.62 ms to 0.43 ms on the smaller 24,000-event,
26-parameter probe. These short callback timings are indicative; the alternating
full-fit comparisons above are the more useful speedup measurement.

`enablePostFitErrorFit: false` disables GUNDAM's separate post-fit error pass.
Minuit can still do numerical curvature/covariance work inside `minimize()`.
Strategy 1 retains substantially more value calls in this experiment. Strategy 0
is useful to measure a faster minimization path, but is not a validation of the
final covariance. This implementation supplies first derivatives, not a Hessian.

## OA2024 diagnostic

The local `configOA2024.yaml` plus `override/v13ProdRun45.yaml` initializes
successfully with the CPU analytic gradient and has 843 fit parameters. A
temporary model capped at approximately 1,000 events initialized in 4.16 s;
median value and gradient callback times were 0.235 ms and 1.96 ms respectively.
This is a compatibility and timing probe, not a representative physics fit:
most sample bins have no events at this cap. The production input files and
configuration were not modified.

A second probe used a deterministic single-threaded loader capped at 1,000
events, keeping propagation/gradients on four threads and all 843 parameters.
Three fits per mode, alternating order, used strategy 0, tolerance `1e-5` and no
separate Hesse pass. Median minimization times were 19.795 s numerical versus
1.550 s analytic (12.77×). Value calls fell from 92,807 to 3,495, with 48 analytic
gradient calls. Both modes converged, EDM below `1.4e-8`, likelihood below
`3e-7`; the largest difference in fitted parameters was `3.7e-4` in units of the
parameter standard deviation. Four selected gradient components also agreed
with central finite differences away from the prior. This strongly reduced
Asimov model is a compatibility test; the gain cannot be extrapolated to the
full OA2024 statistics or covariance calculation.

The initial capped probe exposed a reader/filler shutdown race in
`debugNbMaxEventsToLoad`: an early return could leave the reader waiting for a
buffer acknowledgement. The loader now completes the shutdown handshake.
`904EventLoadingCap.py` repeats initialization with a cap of one event and a
subprocess timeout, using one and four threads.

Validation also includes `903CpuWeightGradient.py` with one/four threads,
235 C++ fast tests, and four `BackendFitCovariance` tests. The latter compare
numerical/analytic fit parameters, errors and Hesse covariance, including
normalized coordinates and optional Simplex initialization.

The full OA2024 fit was not benchmarked here. Its input loading, parameter count,
curvature calculations and model degeneracies can substantially change the gain.
Value and gradient callbacks still propagate independently for correctness.

## CPU backend gradients

The same script accepts `--backend CPU` (the default remains `Original`). CPU
runs disable automatic materialization after initialization. For example:

```bash
python3 tests/benchmarks/236CpuGradientBenchmark.py \
  --backend CPU --events 120000 --bins 48 --threads 4 --trials 5 \
  --strategy 0 --output /tmp/cpu-backend-gradient-strategy0
```

Measurements with the backend gradient implementation, on the same M4 Max,
Release, ROOT 6.40.04, four threads, 120,000 events, 50 parameters, five fits per
mode with alternating order (setup excluded):

| Engine | Strategy | Numerical | Analytic | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU backend | 0 | 0.151 s | 0.051 s | 2.96× |
| CPU backend | 1 | 0.291 s | 0.218 s | 1.33× |
| Original engine | 0 | 0.320 s | 0.116 s | 2.77× |

For CPU strategy 0, value calls fell from 1,434 to 234, plus 12 gradient calls.
The full gradient callback took about 1.07 ms and a value callback 0.103 ms.
These timings describe the synthetic fixture, not the full OA2024 fit.
