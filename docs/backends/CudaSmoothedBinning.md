# CUDA smoothed binning

CUDA implements smoothed/fractional binning in native doubles, with host
math/validation tests and CPU/MPS/Python regressions passing locally. NVCC
compilation and execution of the new GPU path remain pending CC validation.
The previous five CUDA suites, including observable responses and strict
migration, passed at CC-IN2P3 on 2026-10-25 (1.53 s total).

## Semantics

Fractions use the same cumulative kernel and interval products as CPU/MPS.
A dimension may stay strict while another is smoothed. Sample geometry already
contains the inverse widths resolved from `width` or `widthFraction` by the
common builder; CUDA introduces no extra configuration.

An event retains its first matching hard bin and contributes independently to
every bin with a nonzero fractional response. The remaining fraction contributes
to the sample's outside bin. Sums use `weight * fraction`; squared sums use
`(weight * fraction)^2`, including outside contributions and signed weights.
Fixed, strict dynamic and smoothed samples can share one model.

The sum of fractions must not exceed one beyond the CPU roundoff tolerance:
`64 * epsilon(double) * max(1, nonzeroContributionCount)`. A tolerated excess
is subtracted from the first largest contribution. Larger overlaps or nonfinite
responses abort propagation and invalidate the token/caches. Empty bin selections
send the whole weight outside. Zero events have zero normal and outside sums.

## Storage and execution

The existing observable kernel updates variables and hard bins. For models with
smoothed samples, an additional count pass evaluates fractional responses and a
GPU prefix scan builds per-event offsets. A small extent/error readback completes
before allocation. Scratch buffers grow only when actual contribution counts
exceed their capacity; there is no events-by-bins allocation.

A second pass writes contributions in original event/bin declaration order.
The committed per-event list is retained for lazy `EventState` readback. A stable
GPU sort of a separate list groups contributions by global bin, including a slot
per sample for outside sums. Existing deterministic histogram reductions use a
fractional specialization; no floating-point atomics are introduced.

The strict-only path keeps its existing index and reduction specialization.
Weight-only updates reuse the resident fractional list/index in both propagation
modes. Relevant parameters, external/table generations, input revisions or host
edits trigger a state/index update. New variables, hard bins, offsets and
contributions are published after successful device execution. Errors/rebuilds
invalidate caches and retries rebuild the complete state.

The initial implementation scans each sample's bins to count/write fractions;
production performance has not been benchmarked. State updates require a count
synchronization and may temporarily retain committed, staging and sorted lists.
`contributionBufferBytes` reports those record buffers' allocated capacities;
`binContributionCount` reports active dynamic/smoothed event-state contributions.
Outside values are materialized with histograms; their bins are excluded from
the normal `BackendDeviceView` histogram byte range and statistical likelihood.

## Tests at the CC

After retrieving the changes, run in an allocated GPU session:

```bash
cmake --build /sps/t2k/ablanche/build/gundam \
  --target gundamGTest_cudaBackend.exe -j4 &&
ctest --test-dir /sps/t2k/ablanche/build/gundam \
  -R '^cuda-backend-' --no-tests=error --output-on-failure
```

This adds `cuda-backend-smoothed` to the previous five suites. It covers both
response modes/layouts against CPU, mixed sample types, 1D/2D fractions, gaps,
asymmetric widths, outside bins/squares, signed/zero weights, caps, migrations,
external updates, host edits, revisions, caches, errors/retries, rebuilds and
empty inputs. Independent quarter-fraction assertions distinguish squared
weighted fractions from fractions of squared weights.

The host `CudaBinning` tests check analytic halves/quarters, outside fractions,
roundoff normalization, invalid overlaps/widths and bounded contribution writes.
Locally, the 92 selected native regressions and both Python engine/producer tests
pass. CUDA test C++ and the CUDA-facing backend call signatures are syntax-checked
locally; GPU kernel compilation/execution requires the CC.

For the full engine, build `GUNDAM` with `WITH_PYTHON_INTERFACE=ON`, then use the
matching Python interpreter and the existing test dependencies:

```bash
repoDir="$PWD"
export PYTHONPATH="/sps/t2k/ablanche/build/gundam/src/PythonInterface${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p cuda-smoothed-tests
cd cuda-smoothed-tests
bash "$repoDir/tests/fast-tests/005PythonSetup.sh" "$repoDir/tests/fast-tests" &&
GUNDAM_TEST_CUDA=1 venv/bin/python "$repoDir/tests/fast-tests/230BackendObservableResponses.py" &&
GUNDAM_TEST_CUDA=1 venv/bin/python "$repoDir/tests/fast-tests/222VariableResponseDependencies.py"
```

The first test checks strict/smoothed composition, `widthFraction`, manual host
binning, materialization, likelihood and Asimov copies. The second checks external
observable producers and invalidation with both strict and smoothed bins. Both
CUDA response modes are required by this explicit opt-in. Without the opt-in,
the normal CPU/MPS tests remain usable without a CUDA toolkit or GPU.
