# Backend fit and covariance regression

The compiled regular tests in
[`unitTest_BackendFitCovariance.cpp`](../../tests/GTests/unitTest_BackendFitCovariance.cpp)
run a complete `FitterEngine` fit through `BackendManager`, Migrad and Hesse.
They generate their own ROOT input: 540 weighted events, eight fixed bins,
two category normalizations and an event-dependent 1D graph shape response.
No OA2024 inputs or other external data are needed.

The data are Asimov at the Gaussian priors. The fit starts away from those
priors, so recovering the known minimum requires a real minimization. The
parameters share bins and have nonzero posterior correlations. Fits use physical
parameter coordinates (`useNormalizedFitSpace: false`) and the same explicit
function precision, seed and two host threads for all execution paths.
The legacy CacheManager is disabled during the comparison.

## Comparisons

- `BackendFitCovariance.CpuMatchesOriginalFitAndCovariance`: CPU backend versus
  the original engine. It also runs when CUDA is unavailable.
- `Modes/CudaFitCovariance.MatchesCpuFitAndCovariance/cachedResponses`: CUDA
  cached mode versus CPU.
- `Modes/CudaFitCovariance.MatchesCpuFitAndCovariance/recomputeResponses`: CUDA
  recompute mode versus CPU.

Each fit must converge with status zero, EDM below its configured target and an
accurate Hesse covariance (`CovMatrixStatus == 3`). The comparison checks best-fit
parameters, errors, total/statistical/penalty likelihoods, histogram sums and
squared sums. It checks every covariance entry, the correlation matrix and all
covariance eigenvalues. Matrices must be finite, symmetric and positive definite.

At this Asimov minimum, an independent analytic reference is available:

```text
objective = 2 * sum((prediction - data)^2 / data) + Gaussian penalties
covariance = inverse(2 * J^T * diag(1/data) * J + diag(1/priorSigma^2))
```

`J` is calculated directly while generating the input, independently of GUNDAM's
dial propagation and Hesse. Every backend must also match this reference; CPU
and CUDA agreeing with each other is not the sole validation criterion.

## Tolerances and reports

The comparison uses dimensionless covariance differences to handle small or
zero off-diagonal entries without dividing by those entries:

```text
abs(C_candidate[i,j] - C_reference[i,j]) / sqrt(C_reference[i,i] * C_reference[j,j])
```

| Quantity | Allowed difference |
|---|---|
| Best-fit parameter | `2e-4` times the reference posterior standard deviation |
| Posterior error | `2e-4` relative |
| Any covariance entry | `5e-4` in the units above |
| Any correlation entry | `5e-4` absolute |
| Covariance eigenvalue | `1e-3` relative |
| Covariance versus analytic reference | `1e-3` in posterior standard-deviation products |
| Total/statistical/penalty likelihood | `1e-7` absolute; final total must also be below `1e-6` |

The JSON reports contain parameter names/values/errors, full covariance and
correlation matrices, eigenvalues, the analytic covariance, likelihoods,
histograms, convergence diagnostics, tolerances and the maximum matrix
differences. Files are named:

```text
BackendFitCovariance_CPU_reference.json
BackendFitCovariance_CUDA_cachedResponses.json
BackendFitCovariance_CUDA_recomputeResponses.json
```

CTest writes them under `<build>/Testing/BackendFitCovariance/`. Running the
executable directly, including with the Slurm script, writes them in its current
working directory. A skipped test produces no fit report.

## Running and optional hardware

These are native CTest regular tests, separate from the scripts enumerated by
`tests/gundam-tests.sh`. They need GoogleTest and ROOT, but do not need the
GUNDAM Python interface, PyROOT or a Python virtual environment.

Configure and build in an allocation at the CC, preserving the CUDA compiler
and architecture settings already saved in the build cache:

```bash
cmake -S . -B /sps/t2k/ablanche/build/gundam \
  -DWITH_TESTS=ON -DWITH_GOOGLE_TEST=ON -DWITH_REGULAR_TEST=ON &&
cmake --build /sps/t2k/ablanche/build/gundam \
  --target gundamGTest_regularTest.exe -j4
```

With a GPU allocation:

```bash
GUNDAM_REQUIRE_CUDA_TESTS=1 ctest \
  --test-dir /sps/t2k/ablanche/build/gundam \
  -R '(BackendFitCovariance|CudaFitCovariance)' --output-on-failure
```

Alternatively, from the repository root, submit the executable in batch:

```bash
GUNDAM_REQUIRE_CUDA_TESTS=1 sbatch tests/cuda/CcLyonSmoke.sbatch \
  /sps/t2k/ablanche/build/gundam/tests gundamGTest_regularTest.exe
```

By default the tests skip successfully when the backend library or CUDA support
was not compiled, when CUDA device enumeration is unavailable, or when no device
is visible. Once a device is found, fit/configuration/numerical failures are
errors, not skips. `GUNDAM_REQUIRE_CUDA_TESTS=1` makes unavailable CUDA an error
too, preventing a CC validation job from silently skipping the comparison.

## Validation status

On the development Mac, the CPU/original fits and covariances pass, including
the analytic reference. The maximum covariance difference divided by posterior
standard-deviation products is about `2.7e-10`, with maximum absolute correlation
difference about `2.3e-10`. The CPU/original comparison takes less than a second.
The no-CUDA and no-backend skip paths, and the strict-CUDA failure path, have
also been checked locally. The three `BackendFitCovariance_*.json` reports
returned from the CC on 2026-10-24 all have `passed: true`. Both CUDA modes
agree with CPU to `1.93e-10` in covariance entries normalized by posterior
standard-deviation products and `1.14e-10` in absolute correlation difference.
All fits have status 0, covariance status 3 and 92 calls. Both modes returned
identical fit/covariance results. This small regression does not validate the
full production OA2024 covariance.
