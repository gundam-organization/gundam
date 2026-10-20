# MPS Hesse precision — October 20, 2026

## Diagnosis

The Upgrade CC 2026 Asimov comparison exposed a numerical differentiation
problem. MPS event weights and final histogram buffers use float, while Minuit2
previously used its default function precision. Hesse could therefore choose
steps too small for the numerical noise in the GPU results. A small difference
in histograms does not imply a small difference in their second derivatives.

This is separate from the previously corrected cancellation in the Barlow–Beeston
likelihood. That stable likelihood implementation was already present in all
files compared here.

A runtime probe of physical flux parameter #110 illustrates the problem. At the
prior, central second differences gave:

| Step / prior sigma | Reference curvature | MPS curvature |
|---|---:|---:|
| 0.1 | 76.909141 | 76.909151 |
| 0.01 | 76.909117 | 76.909202 |
| 0.001 | 76.909116 | 76.911786 |
| 0.0001 | 76.909116 | 76.921415 |
| 0.00001 | 76.909116 | 77.021558 |
| 0.000001 | 76.909116 | 80.212473 |

The reference and backend paths were evaluated at each point, with the reference
first to avoid reading stale host dial caches. The largest observed absolute
histogram-bin difference was approximately `6.75e-5`. These probes and inspection
of parameter uploads and device buffers support the precision diagnosis; they
do not establish a noise bound for other models.

## Implementation

- `BackendCapabilities::recommendedFunctionPrecision` lets each backend describe
  its recommended objective-function precision to numerical fitters. Zero leaves
  the fitter's default unchanged.
- MPS recommends `std::numeric_limits<float>::epsilon()`, exactly
  `1.1920928955078125e-7`. This is a conservative starting point for its current
  arithmetic, not a measured bound on accumulated likelihood error.
- `RootMinimizer` passes the recommendation to `SetPrecision()` at initialization.
  A configured `minimizerConfig.functionPrecision` takes precedence and must be
  finite and strictly positive. CPU and reference defaults remain unchanged.
- The recommendation or override is logged. The precision is exposed through
  `getFunctionPrecision()` and saved with the minimizer settings as
  `functionPrecision_TNamed`, retaining enough digits for small values.

The change affects numerical differentiation in both Migrad and Hesse. It adds
no backend buffers, changes no device kernels and introduces no new dependency.
It does not make the float calculation equivalent to CPU double precision.

## Real fit results

The working directory was `upgrade-cc-2026`, using:

```sh
gundamFitter -c ndUpCcInc.yaml \
  -of override/datasetHighlandv5.27.1_CCLyon.yaml override/backendMpsTest.yaml \
  -a -t 4
```

Two new Release fits included the complete post-fit Hesse calculation and wrote
to separate files under `/tmp/gundam-mps-hesse/`. User outputs and `comp.root`
were preserved. The reference was the existing CPU-original Asimov output with
the stabilized likelihood. The supplied MPS file provided the original result.
The model has 277,177 events, 4,302,852 event-dial references, 374 bins and 313
fitted parameters.

The table compares Hesse standard errors, excluding fixed parameters with zero
reference error. Each cell is the range of `MPS error / CPU-original error`.

| Parameter set | Original MPS | Automatic float precision | Explicit `1e-9` |
|---|---:|---:|---:|
| Flux 320 kA | 0.171933–0.739752 | 1.000975–1.005205 | 0.999680–1.004920 |
| Detector efficiencies | 0.868754–1.027966 | 0.999116–1.000926 | 0.994723–1.019551 |
| Detector migration | 0.048482–0.780974 | 0.998677–1.000262 | 0.996514–1.019907 |
| Cross section | 0.042447–1.289193 | 0.998327–1.010364 | 0.996418–1.016223 |
| Pion FSI | 0.967924–0.981674 | 0.999926–1.000333 | 0.999933–1.001180 |
| Binding energy | 0.521579–0.696353 | 1.001200–1.001927 | 1.001195–1.002727 |

With the automatic recommendation, the largest flux error difference is **0.521%**
and the largest difference across all sets is **1.036%**. The normalized-fit-space
covariance has a relative Frobenius difference of **0.618%** from the reference;
the largest absolute correlation-coefficient difference is **0.0356**. Originally
these were **11.82%** and **1.030**, respectively.

The `1e-9` sensitivity run has a larger maximum standard-error difference
(**1.991%**) and maximum correlation difference (**0.0569**), although its relative
covariance Frobenius difference is smaller (**0.331%**). Precision settings trade
numerical noise against finite-step truncation error; no single metric establishes
accuracy. This run does not justify using the more optimistic value by default.

| Measurement | Original MPS output | Automatic float precision | Explicit `1e-9` |
|---|---:|---:|---:|
| Hesse covariance status | 2 | 3 | 3 |
| Hesse time | 52.576 s | 52.890 s | 52.261 s |
| Hesse calls | — | 50,693 | 50,635 |
| Full process wall time | — | 128.079 s | 127.962 s |
| Peak process RSS | — | 5.373 GB | 5.364 GB |

These are single runs on a shared machine, not an isolated performance benchmark.
There is no observed large Hesse runtime penalty.

The new runs still report **Migrad status 1 and covariance status 2** before
post-fit Hesse, despite a convergence flag and an Asimov minimum of approximately
`2.48e-11`. Hesse subsequently returns success and covariance status 3. The raw
saved `hesseStatusCode` is 1 because ROOT combines the previous Migrad status with
the Hesse status; its hundreds digit is zero. The original MPS file had Migrad
status 0/covariance 3 followed by Hesse covariance 2. The change therefore improves
the final Hesse matrix but does not resolve all minimizer diagnostics.

## Validation and remaining work

- Release build succeeded.
- 106 C++ tests passed, excluding `*RootReaderCleanup*`, which already fails on a
  configuration missing a required `name` field in the earlier build.
- All 28 Python tests numbered 210 through 231 passed, including the array-dial
  input test, MPS dynamic binning, external weights and smoothing.
- New `231MpsHessePrecision.py` checks a weighted, single-bin Asimov model against
  analytic Barlow–Beeston curvature and the expected Hesse error, with a 0.5%
  tolerance. It checks the automatic recommendation, explicit override, saved
  precision and invalid inputs. It skips outside macOS or without backend support.
- As a negative control, explicitly declaring double epsilon on that model gave
  an error of `0.0429647` instead of `0.0432060`, failing the new Hesse assertion.

This is a mitigation for numerical differentiation, with residual differences
visible in the real covariance. It has not been validated on OA2024, non-Asimov
data or every backend feature combination. The next steps are to check multiple
step sizes and non-Asimov minima, investigate the Migrad covariance warning, and
measure whether increased precision in parameter differences, event-weight
products and histogram reductions is worthwhile. A CPU double Hesse at the GPU
best-fit point would also provide a useful accuracy reference or an explicit
fallback when tighter uncertainty agreement is required.
