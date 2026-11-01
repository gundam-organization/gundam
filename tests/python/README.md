# Python tests

These tests run directly with Python, independently of `tests/gundam-tests.sh`.
They use the interpreter you launch; they do not create an environment, install
packages or switch interpreters automatically.

- `fast/`: regression tests, including Python interface CMake configuration tests.
- `regular/`: longer validation tests and plots.
- `benchmarks/`: performance measurements and their supporting scripts.

The directory identifies the test suite; the first digit identifies the primary
behavior being checked:

| Range | Category | What is checked |
|---|---|---|
| **0XX** | Build and installation | CMake configuration, Python interface builds, imports and installation compatibility |
| **1XX** | Configuration | Field resolution, validation, defaults and invalid configurations |
| **2XX** | API and diagnostics | Exposed methods, object access, observable states and monitoring |
| **3XX** | Model loading and content | Events, trees, samples, variables, binning and histograms |
| **4XX** | Dials and responses | Construction, event associations, interpolation, composition and propagation |
| **5XX** | Numerical and statistical results | Likelihoods, gradients, minimization, covariance and pseudo-experiments |
| **6XX** | Execution and lifecycle | Cache invalidation, workers, reuse, loading termination and synchronization |
| **7XX** | Backends | Contracts and results specific to CPU, MPS or CUDA backends |
| **8XX** | Reserved | Available for future categories |
| **9XX** | Reserved | Available for future categories |

Numbers are independent between suites: backend regression tests and benchmarks
both start at `700`, for example `fast/700BackendCpuPropagation.py` and
a backend benchmark numbered `700` in `benchmarks/`. Backend-specific scripts use the
`7XXBackend<Subject>.py` naming convention.

Choose a category by the test's primary purpose. A general feature test may also
exercise a backend without belonging to `7XX`. Model-content checks verify the
loaded objects and data; numerical-result checks verify the calculated values.
Leave gaps when assigning new numbers, and keep a test's number when moving it
between `fast` and `regular`, provided it does not collide with another test.
Numbers do not imply execution dependencies: every script must run independently.

Build GUNDAM with `-DWITH_PYTHON_INTERFACE=ON`, then expose the matching module
through `PYTHONPATH` (or source the installed GUNDAM `setup.sh`). Install the
Python dependencies into your chosen environment:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r tests/python/requirements.txt
export PYTHONPATH="$PWD/cmake-build-release/src/PythonInterface${PYTHONPATH:+:$PYTHONPATH}"
```

From the repository root:

```bash
python tests/python/run.py --list
python tests/python/run.py --output /tmp/gundam-python-fast
python tests/python/run.py --suite regular --output /tmp/gundam-python-regular
python tests/python/run.py --test 510IgnoreZeroPredictionAtPrior.py
python tests/python/run.py --suite benchmarks --test 350BinningResponseBenchmark.py -- --help
```

The runner starts each script in a separate directory with a `test.log`, using
the same Python interpreter. It runs every selected script and exits nonzero if
any script fails. Without `--output`, it creates and preserves a temporary output
directory. An explicit output directory must not already exist.

Scripts can also be launched directly from a writable working directory:

```bash
python /path/to/gundam/tests/python/fast/510IgnoreZeroPredictionAtPrior.py
python /path/to/gundam/tests/python/fast/010PythonInterfaces.py
```

Some tests compile small native libraries and require a C++ compiler. The two
gradient checks `570EvalFitGradientCheck.py` and
`580EvalFitGradientSplineCheck.py` also require the ROOT executable: they prepare
their own inputs using the shared fixtures in `tests/fast-tests`, without running
the standard test suite. Benchmark scripts accept their own arguments; use `--help` to inspect them.

`fast/010PythonInterfaces.py` exercises CMake configuration without compiling
GUNDAM. It requires CMake 3.18 or newer, a C++ compiler, pybind11 and Python
development files; it does not need an installed GUNDAM module.
