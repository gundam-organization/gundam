# Analytic weight gradients on CPU, CUDA and MPS

Minuit2 can use the statistical gradient of a backend with the existing option:

```yaml
fitterEngineConfig:
  minimizerConfig:
    useAnalyticGradient: true
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: BarlowBeestonLiteGaussian
    backendManagerConfig:
      isEnabled: true
      useBackend: MPS # CPU or CUDA
```

Only first derivatives are supplied. No G2/Hessian calculation is added, and the
validated JointProbability classes remain unchanged. Minuit can still perform
numerical initialization and covariance calculations, even when the gradient
option is enabled. An Asimov fit starting at its minimum may never request the
gradient callback.

All three backends currently require fixed, unsmoothed bins and parameter-only
normalization, constant, Graph or supported spline weights. Input mirrors,
extrapolation, hard response limits and the global maximum event reweight cap
are included. Observable-response dials, dynamic/smoothed binning, smooth
response limits, complex dials, unsupported statistical likelihoods and external
constraints are rejected rather than falling back to a host gradient.

CUDA uses native binary64; MPS uses the existing double-float scalar throughout
the gradient kernels and reductions. The MPS exponent range remains that of a
float. The common portable evaluator supplies first derivatives of the same
piecewise Hermite/Graph functions as the propagation kernels.

CUDA and MPS execute four stages on device:

1. Evaluate each unique dial response and derivative.
2. Evaluate the Gaussian Barlow-Beeston profile derivatives for each bin.
3. Propagate derivatives through event-weight products using prefix/suffix
   factors, including zero responses, without dividing by dial responses.
4. Reduce sparse occurrence contributions by physical parameter.

The reduction uses a fixed tree per parameter and no floating-point atomics.
Weights, histogram sums, sum-of-squared weights and intermediate derivatives
remain on device. Only the final statistical gradient vector is read back, in
addition to the statistical LLH total from normal propagation. Automatic
materialization requested by the caller still works, but is not needed by the
gradient. The manager maps the vector to physical parameters; the common
likelihood layer adds Gaussian penalties and transforms to eigen/normalized fit
coordinates as before.

The immutable occurrence layout and scratch buffers are allocated lazily at the
first gradient request, then retained until the model is rebuilt. Memory scales
with dial occurrences, events and parameters; there is no dense
`events × parameters` Jacobian. Value and gradient callbacks still propagate
independently. This first implementation recomputes unique dial derivatives on
each gradient request.

## Validation

On a machine with the GUNDAM test Python environment and matching module:

```bash
cd tests
bash gundam-tests.sh -a -t fast-tests/903CpuWeightGradient.py -v
```

The test defaults to CPU. From the test run directory, the same script supports
`--backend MPS` and `--backend CUDA`; select these explicitly on a machine with
the corresponding compiled backend and accessible hardware. It checks
physical/normalized/eigen gradients against the original engine and finite
differences, with one/four threads and automatic materialization disabled.

The native tests include `DeviceGradient.*` and
`BackendFitCovariance.MetalAnalyticGradientMatchesCpuFitAndHesse`. They cover
cached/recomputed propagation, zero responses, repeated inputs, small rates,
ignored bins, response limits, Graphs/splines, extrapolation, event caps, stale
propagation tokens, fits and Hesse covariance.

On CC Lyon, in a V100 allocation with a CUDA build, run:

```bash
ctest --test-dir /path/to/build -R '^cuda-backend-gradient$' --output-on-failure
/path/to/build/tests/gundamGTest_regularTest.exe \
  --gtest_filter='Modes/CudaFitCovariance.AnalyticGradientMatchesCpuFitAndHesse/*'
```

This checks both CUDA propagation modes, both event-major and warp-interleaved
layouts, normalized and physical fits, and agreement with CPU. CUDA device tests
require a visible GPU and are separate from the ordinary fast test run.

For a short performance comparison, with the matching module on PYTHONPATH:

```bash
python3 tests/benchmarks/236CpuGradientBenchmark.py \
  --backend MPS --events 120000 --bins 48 --threads 4 --trials 3 \
  --strategy 0 --output /tmp/mps-gradient-benchmark
```

Use `--backend CUDA` on the V100. Initialization and lazy gradient-buffer upload
are excluded from timed probes. The script alternates numerical/analytic fits,
checks convergence and fitted-value agreement, and writes call counts and times
to `summary.json`. Performance depends on model size, dial reuse, and Minuit
strategy; gradient support does not remove the cost of Hesse.

Local validation uses Metal on an M4 Max. CUDA kernels and device tests are
implemented, but have not been compiled with nvcc or executed on NVIDIA hardware
in this session; this Mac has neither the CUDA toolkit nor a CUDA device.

Local results (M4 Max, Release, ROOT 6.40.04): 240 C++ fast tests and five
fit/covariance tests passed, as did the Python gradient checks on CPU and MPS
with one/four threads. A synthetic MPS benchmark with 120,000 events, 50
parameters and strategy 0, three alternating trials per mode, measured median
minimization times of 0.268 s numerical versus 0.089 s analytic (3.00×). Value
calls fell from 1,434 to 234 plus 12 gradient calls. This is a displaced Asimov
fixture, not the full OA2024 fit.
