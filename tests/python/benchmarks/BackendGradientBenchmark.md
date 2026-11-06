# CPU and MPS analytic gradient benchmarks

These scripts measure `evalFitGradientAt` through the backend selected by
`BackendManager`, with automatic host materialization disabled. The common
fixture contains deterministic weighted events, a normalization parameter per
bin and two shared event Graph parameters. Data are Asimov and the likelihood
is Gaussian Barlow-Beeston Lite. Parameters use normalized fit coordinates.

The gradient timer includes parameter propagation, statistical likelihood,
penalty derivatives and Python call/result conversion. MPS results must be
available on the host when the callback returns. This measures the callback
used by the fitter, rather than an isolated GPU kernel. Initialization, ROOT
input generation, finite-difference checks and warmup are outside the timers.
Each pass traverses the same varying parameter points; value and gradient
measurement order alternates between trials. No minimization is performed.
There is no hardware-dependent performance pass/fail threshold.

With a matching GUNDAM build (`WITH_PYTHON_INTERFACE=ON`) on `PYTHONPATH` and
ROOT on `PATH`, run explicitly:

```bash
python tests/python/run.py --suite benchmarks --test 728BackendsCpuGradientBenchmark.py -- \
  --events 120000 --bins 48 --threads 4 --points 16 --trials 5 \
  --output /tmp/cpu-gradient
python tests/python/run.py --suite benchmarks --test 747BackendsMpsGradientBenchmark.py -- \
  --events 120000 --bins 48 --threads 4 --points 16 --trials 5 \
  --output /tmp/mps-gradient
```

Output directories must be new. MPS skips on non-macOS systems; on macOS an
unavailable MPS backend is an error, so CPU fallback cannot produce a misleading
MPS timing. The CPU and MPS scripts use identical workloads and options.

Each directory contains the ROOT fixture, `config.json` and `summary.json`.
The summary records raw trial timings per call, medians, gradient/value ratio,
checksums, finite-difference checks, options, platform and loaded module path.
An interrupted timing pass leaves `complete: false`. Compare raw gradient
medians between backends or revisions under the same workload and build mode.
Record the source revision and build options alongside results.

For scaling studies, vary `--events` (e.g. 12000, 120000, 1200000), `--bins`
(e.g. 12, 48, 192), and CPU `--threads` (e.g. 1, 4). Each invocation generates
one fixture. Increasing bins also increases the parameter count and changes the
per-bin occupancy. `--repeats` increases measurement duration without changing
the set of points. `--check-components` controls the number of evenly sampled
finite-difference components; both Graph parameters are always checked too.
Set it to at least `bins + 2` to check all components at both validation points.
MPS uses a larger difference step and looser tolerances for FP32 event weights;
exact tolerances and step are recorded in the report. These synthetic checks
cover this fixture, not every supported dial type.
