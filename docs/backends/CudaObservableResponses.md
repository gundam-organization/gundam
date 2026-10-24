# CUDA observable responses and strict binning

CUDA implements `applyToVariable` composition and strict dynamic binning.
NVCC compilation and all five CUDA CTest suites, including observables, passed
at CC-IN2P3 on 2026-10-25 (1.53 s total).
External/table **event weights** were already confirmed working at the CC.
Locally, 90 selected native regressions and the two Python engine/producer
tests passed (including the existing CPU/MPS cases).

## Execution and scope

Each event evaluates its target from the immutable reference or Override dial,
then applies Multiply dials in order, followed by Add dials in order. Values are
never accumulated from the preceding propagation. All supported dial types
share their cached response slots between weights and observable targets:
scalars, bilinear/bicubic surfaces, ExternalWeightDispatcher and Tabulated/Kriged.
Ordered surface inputs and external publication generations retain their semantics.

`EventStateCache` tracks relevant parameters, external/table generations and
input-buffer revisions. Host variable edits and `forceBinningUpdate` are also
accepted. Unrelated weight-only changes leave the observable state and histogram
index resident. Both response modes use this state cache; `recomputeResponses`
only forces dial-response evaluation, not needless observable recomposition.
External/table observables depend on the producer publication, not on the
producer's input descriptors.

The first CUDA kernel composes variables and searches strict bins. It preserves
bin declaration order for overlapping selections, lower-inclusive/upper-exclusive
edges and equality for zero-width selections. Fixed samples keep their original
bins. Events outside strict bins have no histogram contribution. A stable GPU
sort groups result slots by global bin in original event traversal order; binary
searches rebuild the histogram offsets. Existing FP64 histogram reductions are
reused without floating-point atomics. This is the initial strict executor;
no claim of production binning performance is made.

State changes are staged in separate variable/bin buffers and published after
a successful device pass. Invalid requests invalidate old tokens and both caches;
a retry rebuilds responses and the state/index. Event state is read lazily via
`EventState` and includes variables, sample-local bins, strict contributions and
revisions. Immutable metadata stays on the host. `BackendManager` uses those
revisions when materializing variables and host event lists.

`variableStageSeconds` currently measures the fused composition/strict-search
kernel and staging transfers; `binningStageSeconds` measures histogram-index
construction. Rebin, candidate and contribution counters are available.

Smoothed bins/fractional contributions are now implemented, pending CC
validation: see [smoothed binning](CudaSmoothedBinning.md).
`DialInputSource::EventVariable` remains rejected during model preparation. No CPU propagation fallback is used.
Resident backend replacement retains its existing fixed-model restriction.

## Device regression at the CC

After retrieving the code, run in an allocated GPU session:

```bash
cmake --build /sps/t2k/ablanche/build/gundam \
  --target gundamGTest_cudaBackend.exe GUNDAM -j4 &&
ctest --test-dir /sps/t2k/ablanche/build/gundam \
  -R '^cuda-backend-' --no-tests=error --output-on-failure
```

The new `cuda-backend-observables` suite checks both propagation modes and
weight layouts against CPU: mixed fixed/dynamic samples, reordered result slots,
Override/Multiply/Add composition, surfaces, external/table updates, weight-only
changes, host edits, revisions, overlapping/zero-width bins, outside events,
state readback, errors/retries, rebuilds and empty layouts. Independent analytic
values check the composition. Existing CUDA weight suites run alongside it.
Device suites are registered only in CUDA-enabled builds and require a GPU;
compilation itself does not launch them.

## Full engine and producer regression

`230BackendObservableResponses.py` compares native propagation and CPU through
the actual engine, including state materialization, histogram pointer lists,
weights, likelihood and Asimov copies. `222VariableResponseDependencies.py`
checks an external Python producer and explicit invalidation. Both keep their
default CPU/MPS tests usable without CUDA. `GUNDAM_TEST_CUDA=1` additionally
requires successful CUDA execution in both modes, using strict and smoothed bins.

From the repository root at the CC, with a matching Python interface build:

```bash
repoDir="$PWD"
export PYTHONPATH="/sps/t2k/ablanche/build/gundam/src/PythonInterface${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p cuda-observable-tests
cd cuda-observable-tests
bash "$repoDir/tests/fast-tests/005PythonSetup.sh" "$repoDir/tests/fast-tests" &&
GUNDAM_TEST_CUDA=1 venv/bin/python "$repoDir/tests/fast-tests/230BackendObservableResponses.py" &&
GUNDAM_TEST_CUDA=1 venv/bin/python "$repoDir/tests/fast-tests/222VariableResponseDependencies.py"
```

Use the Python version matching the compiled module when creating `venv`.
Without the explicit CUDA opt-in, absence of CUDA/toolkit/GPU is not an error.
