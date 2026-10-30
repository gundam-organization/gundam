# CUDA formula and composite dials

CUDA executes the common GUNDAM formula bytecode and flattened composite graphs
in native double precision, without ROOT evaluation or runtime compilation.
Configuration syntax is identical to CPU and MPS. Supported arithmetic and math
functions are those of `FormulaCompiler` (including exp/log/sqrt, trigonometric
functions, powers, min/max, erf, floor and ceil).

Event-formula inputs are values bound when building the engine layout. They are
uploaded as constants alongside parameter bindings. This preserves the existing
EventFormula behavior; a formula does not read observables modified by another
dial during the same propagation.

Formula programs are shared; root dials have disjoint device workspace. Composite
instructions preserve nested input mirrors, original spline bounds, response
limits and smoothing. Responses feed both event weights and observable transforms,
including dynamic and smoothed rebinning. Both `cachedResponses` and
`recomputeResponses` work with `eventMajor` and `warpInterleaved` layouts. Cache
dependency groups support arbitrary input counts and dispatch each dial once when
several inputs change together. No additional library is required.

## Build and validate at CCLyon

Build with the site's compiler/ROOT/Python environment and a CUDA toolkit:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DWITH_BACKENDS=ON -DWITH_CUDA_LIB=ON -DWITH_PYTHON_INTERFACE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-cuda -j4
```

`70` is an example for V100; select the architecture of the allocated GPU.
Set `CMAKE_CUDA_COMPILER` if `nvcc` is not on PATH. Run the following inside a GPU
allocation (e.g. the site's `srun -p gpu_v100_interactive --gpus=1 --ntasks=1 …`).
`deviceIndex=0` refers to the first GPU visible through `CUDA_VISIBLE_DEVICES`.

```bash
ctest --test-dir build-cuda -R '^cuda-backend-' --output-on-failure
export PYTHONPATH="$PWD/build-cuda/src/PythonInterface${PYTHONPATH:+:$PYTHONPATH}"
export GUNDAM_TEST_CUDA=1
cd tests
bash gundam-tests.sh -a -t fast-tests/235CudaFormulaCompositeDials.py -v
```

The C++ tests compare CUDA with CPU for every opcode, composite graph evaluation,
three-parameter invalidation, repeated evaluations and both modes/layouts. The
Python matrix compares with the original ROOT engine, including nested composites,
event-only formulas, event inputs, spline/surface leaves and observable migration.
Explicit CUDA selection fails when CUDA is unavailable; there is no CPU fallback.

## Production benchmark

Select the Python interpreter matching the compiled GUNDAM module (and its runtime
libraries). From the repository root, inside a GPU allocation:

```bash
export PYTHONPATH="$PWD/build-cuda/src/PythonInterface${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=/path/to/venv/bin/python \
  tests/benchmarks/231CudaFormulaCompositeBenchmark.sh \
  /path/to/upgrade-cc-2026 --output "$PWD/cuda-eb-benchmark"
```

The wrapper uses `ndUpCcInc.yaml`, the Highland/ebMigration/binResponse/backendCpuTest
overrides, and `override/onlyEb.yaml`. Each of five cases loads a separate Asimov
engine: CPU and both CUDA response modes with both event index layouts. Defaults
are four threads, three trials, eight warmup calls, 100 calls for constant/local
scenarios, and 40 global calls. Every eligible coordinate is covered in `local1`;
`local2` changes pairs and `global` changes all coordinates. Additional CLI options
override defaults, for example `--calls 20 --global-calls 10 --trials 1` for a smoke
run. Use `--cases CPU CUDA-cachedResponses-interleaved` for a shorter comparison.
The resolved fitter configuration is saved per case, with one native log per worker.

Outputs include:

- `report.md`: median/p95 evaluation and stage times, GPU stage ranking and speedups.
- `comparison.json`: likelihoods at every measured point, final histograms and
  outside bins for every trial, plus all event weights/Pmu and exact bin membership.
- `local1-coordinate-costs.json` per case: costs and response candidates per parameter.
- `settings.json`, `model.json`, `parameters.json`: environment, selected module,
  device/model metadata, workspace bytes and replay coordinates.
- Binary event snapshots: native-endian doubles with layout recorded in trial JSON;
  these are compared on the same host and retained for diagnosis.

Each worker disables automatic event materialization during timing. Explicit
readback, snapshots and comparisons occur afterwards. Snapshots use an additional
untimed evaluation at the same parameter point with automatic materialization. GPU durations come from CUDA
stream events; they overlap synchronous host evaluation time and must not be added
to it. Differences exceeding the default relative `1e-9`/absolute `1e-7` tolerances,
any bin-index mismatch, or absent rebinning in a changing scenario fail the run.
Warmup snapshots are also retained. These seeded workloads measure propagation,
not minimizer/Hesse behavior. GPU results need not be bit-identical to CPU libm.

CUDA compilation and GPU execution cannot be checked on a macOS machine without a
CUDA toolkit/device; run the device tests before interpreting benchmark timings.
