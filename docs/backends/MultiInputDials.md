# Multi-input dials

CPU and MPS support `Surface` dials with the `Bilinear` and `Bicubic`
subtypes and two ordered parameter inputs. Existing configuration is used
unchanged, including per-input mirroring, extrapolation and response limits.
Surfaces can contribute to event weights or observable responses.

`BackendDialDescriptor::firstInput/inputCount` addresses an ordered range in
the common input table. Inputs carry a `DialInputSource`; `Parameter` is
implemented, while `EventVariable` is reserved and explicitly rejected during
backend preparation. No event-variable configuration or changes to the original
engine's `DialInputBuffer` are introduced by this extension. Supporting those
inputs will require an event-context evaluator, storage for auxiliary event
variables and a defined ordering relative to observable propagation. A shared
surface payload must then be distinguished from event-dependent response slots.

## Execution and caches

CPU retains its specialized scalar cache loops when the model has no
multi-input dials. Models containing surfaces select an additional compiled
loop specialization. Multi-input responses enter a separate candidate list and
are invalidated if any parameter changes. Each response is evaluated once even
if several inputs change, or the same parameter occupies several input slots.
Shared scalar-input groups remain available in mixed models.

MPS selects its executor in `MpsBackend::build`, during runtime model preparation.
The default `propagationMode: cachedResponses` uses the resident event-propagation
executor for scalar and surface dials alike. Surface models also use this executor
in `recomputeResponses` mode, with the response cache between evaluations disabled.
Their ordered
input ranges and references are uploaded in side buffers. The interpolation
loads two local scalar values, with no allocation or variable-sized temporary.
Dependency groups contain disjoint response slots and the set of all their
parameter dependencies, preventing duplicate GPU writes on simultaneous changes.
External-weight dials retain their generation-based invalidation even when their
producer declares multiple parameter dependencies.

In `recomputeResponses` mode, compatible scalar fixed-bin models use the
packed-weight executor. Its 24-byte spline descriptors, 16-byte norm/external
occurrences, kernels and payload layouts are unchanged. The common host input
descriptor remains 32 bytes on 64-bit platforms. The input-range side table is
only populated for resident models containing surfaces.

`SharedSurface.h` implements the same bilinear and Catmull-Rom interpolation and
edge continuation as the original surface dials. CPU uses doubles; resident MPS
uses compensated float pairs before publishing float weights and histograms.
Segment searches handle short and large axes without speculative out-of-array
coordinate reads. Payload shape and increasing finite coordinates are validated
while preparing the backend.

## Validation

`unitTest_BackendSurface.cpp` compares both surface subtypes to the original
`DialBase::evalResponse` on irregular axes. It covers scalar/surface mixtures,
weight/observable usage, sparse and dense updates, repeated points, reordered and
repeated parameter references, mirroring, extrapolation, response bounds, cache
options, rebuilds and invalid inputs. Analytic checks also cover 2-knot and
larger-than-16-knot axes.

For production inputs, `tests/benchmarks/227BackendComparison.py` loads the same
Asimov model in separate original/CPU/MPS processes and compares every measured
likelihood, bin content and bin error. Timing excludes input updates and explicit
histogram materialization. Local scenarios cover every effective fit coordinate;
all-coordinate changes use seeded bounded proposals. These are propagation
benchmarks, not full minimizer or Hesse timings.

```sh
env -u DYLD_LIBRARY_PATH \
  PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface" \
  python3.15 tests/benchmarks/227BackendComparison.py \
  --work-dir /Users/nadrino/Documents/Work/Output/results/gundam/upgrade-cc-2026 \
  --config ndUpCcInc.yaml --threads 4 \
  --override override/datasetHighlandv5.27.1_CCLyon.yaml override/enableEb2d.yaml \
  --output /tmp/gundam-2d-comparison
```

Select the interpreter matching the built module. On macOS, an installed GUNDAM
in `DYLD_LIBRARY_PATH` can redirect native library loading even when Python
reports the build-tree module. Use a matching native-library search path or
remove that variable for the benchmark invocation.

For before/after scalar performance measurements, use
`222CpuWeightSchedulingBenchmark.py` and `225MpsBackendProfiling.py` with the same
configuration, thread count, seed, call counts and trial counts on both builds.

## Production measurements (2026-10-22)

Release build, Apple M4 Max, four host threads, `upgrade-cc-2026/ndUpCcInc.yaml`
and `override/datasetHighlandv5.27.1_CCLyon.yaml`: 277,177 events and 374 bins.
With `override/enableEb2d.yaml`, the model contains 456 Bilinear dials and 314
effective fit coordinates. The production input exercises Bilinear; Bicubic is
covered by the synthetic native/CPU/MPS regression tests.

The comparison script checks 30 constant points, 314 single-coordinate points,
157 two-coordinate points and 30 all-coordinate points, including the Eb inputs.
Median evaluation times in milliseconds, excluding explicit materialization:

| Parameter changes | Original | CPU | MPS |
|---|---:|---:|---:|
| None | 1.848 | 0.841 | 0.549 |
| One coordinate | 1.844 | 0.848 | 0.274 |
| Two coordinates | 1.838 | 0.851 | 0.268 |
| All coordinates | 24.334 | 3.865 | 1.078 |

CPU's maximum absolute likelihood difference is `4.55e-13`; the corresponding
MPS difference is `6.40e-5`. Maximum bin-content/error differences are `5.80e-12`
and `1.81e-4`, respectively. MPS's maximum difference divided by
`max(1, abs(reference))` is `3.28e-7`. Every measured point passes the comparison.

For the scalar-only configuration, the existing CPU and MPS profiling scripts
were run before and after the change with three trials per scenario, 30 minimum
calls (extended for complete local-coordinate coverage), 10 all-coordinate calls,
four threads and default seeds. CPU's default Dynamic scheduling gave:

| Parameter changes | CPU before | CPU after | MPS before | MPS after |
|---|---:|---:|---:|---:|
| None | 0.842 | 0.833 | 0.713 | 0.701 |
| One coordinate | 0.857 | 0.841 | 0.708 | 0.715 |
| Two coordinates | 0.962 | 0.862 | 0.727 | 0.715 |
| All coordinates | 3.849 | 3.856 | 0.701 | 0.700 |

Times are milliseconds and subject to scheduler/clock variation; these
measurements show no material scalar-path regression. Fixed CPU scheduling was
also checked: changes ranged from -0.8% to +2.6%. All stored scalar-path
likelihoods and final histograms were identical before and after on both backends.
