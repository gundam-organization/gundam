# Python tests

These tests run directly with Python, independently of `tests/gundam-tests.sh`.
They use the interpreter you launch; they do not create an environment, install
packages or switch interpreters automatically.

- `fast/`: regression tests, including Python interface CMake configuration tests.
- `regular/`: longer validation tests and plots.
- `benchmarks/`: performance measurements and their supporting scripts.

Use `NNN<Category><Subject>.py` for test scripts. The directory identifies the
test suite; the first digit and fixed category prefix identify the primary
behavior being checked:

| Range | Prefix | Category | What is checked |
|---|---|---|---|
| **0XX** | `Build` | Build and installation | CMake configuration, Python interface builds, imports and installation compatibility |
| **1XX** | `Config` | Configuration | Field resolution, validation, defaults and invalid configurations |
| **2XX** | `Api` | API and diagnostics | Exposed methods, object access, observable states and monitoring |
| **3XX** | `Model` | Model loading and content | Events, trees, samples, variables, binning and histograms |
| **4XX** | `Dials` | Dials and responses | Construction, event associations, interpolation, composition and propagation |
| **5XX** | `Numerics` | Numerical and statistical results | Likelihoods, gradients, minimization, covariance and pseudo-experiments |
| **6XX** | `Runtime` | Execution and lifecycle | Cache invalidation, workers, reuse, loading termination and synchronization |
| **7XX** | `Backends` | Backends | Contracts and results specific to CPU, MPS or CUDA backends |
| **8XX** | `Compatibility` | Configuration and input backward compatibility | Historical configuration keys and input conventions retain their results and expected deprecation warnings |
| **9XX** | — | Reserved | Available for future categories |

Numbers are independent between suites: backend regression tests and benchmarks
both start at `700`, for example `fast/700BackendsCpuPropagation.py` and
a backend benchmark numbered `700` in `benchmarks/`. Backend-specific scripts use the
`7XXBackends<Subject>.py` naming convention.

Choose a category by the test's primary purpose. A general feature test may also
exercise a backend without belonging to `7XX`. Model-content checks verify the
loaded objects and data; numerical-result checks verify the calculated values.
Leave gaps when assigning new numbers, and keep a test's number when moving it
between `fast` and `regular`, provided it does not collide with another test.
Numbers do not imply execution dependencies: every script must run independently.
Use PascalCase for the category and subject (`Api`, `Cpu`, `Mps`, `Cuda`). Avoid
repeating the category in the subject: use `470DialsComposite.py`, rather than
`470DialsCompositeDials.py`. Benchmark wrappers retain the same category prefix
with their appropriate `.sh`, `.sbatch` or `.mm` extension.

Feature tests exercise the current configuration and input formats. Backward
compatibility checks live in separate `8XX` scripts; they compare historical
syntax with current syntax or a known numerical reference, and verify any
required deprecation warnings. A new input representation is a feature test;
support for an older representation or an old configuration key is a backward
compatibility check.

| Compatibility script | Historical behavior covered |
|---|---|
| `810CompatibilityEvalFromLib.py` | `variablesTransform` versus `variableDict/evalFromLib` |
| `820CompatibilityFriendTrees.py` | Individual friend paths, mixed declarations and disabled friends |
| `830CompatibilityDialBranchData.py` | `dialLeafName` and `treeExpression` versus `dialBranchData` |
| `840CompatibilityBinnedDials.py` | Implicit `Graph`/`Spline` branches and deprecation warnings |
| `850CompatibilityDialOptions.py` | `dialConfig` versus `options` for Formula, Composite and compiled dials |

These scripts reuse input builders and assertion helpers without running another
test's `main()`. They generate their own inputs in their own working directory.
Shared compiled-variable-transform helpers live in `helpers/EvalFromLib.py`;
the corresponding feature check is `210ApiEvalFromLib.py`.


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
python tests/python/run.py --test 510NumericsIgnoreZeroPredictionAtPrior.py
python tests/python/run.py --suite benchmarks --test 350ModelBinningResponseBenchmark.py -- --help
```

The runner starts each script in a separate directory with a `test.log`, using
the same Python interpreter. It runs every selected script and exits nonzero if
any script fails. Without `--output`, it creates and preserves a temporary output
directory. An explicit output directory must not already exist.

Scripts can also be launched directly from a writable working directory:

```bash
python /path/to/gundam/tests/python/fast/510NumericsIgnoreZeroPredictionAtPrior.py
python /path/to/gundam/tests/python/fast/010BuildPythonInterfaces.py
```

Some tests compile small native libraries and require a C++ compiler. The two
gradient checks `570NumericsFitGradient.py` and
`580NumericsFitGradientSpline.py` also require the ROOT executable: they prepare
their own inputs using the shared fixtures in `tests/fast-tests`, without running
the standard test suite. Benchmark scripts accept their own arguments; use `--help` to inspect them.

`fast/010BuildPythonInterfaces.py` exercises CMake configuration without compiling
GUNDAM. It requires CMake 3.18 or newer, a C++ compiler, pybind11 and Python
development files; it does not need an installed GUNDAM module.
