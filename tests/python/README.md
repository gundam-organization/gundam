# Python tests

## Prerequisites

Build GUNDAM with `-DWITH_PYTHON_INTERFACE=ON`, then expose the matching module
through `PYTHONPATH` (or source the installed GUNDAM `setup.sh`).

## Running with CTest and CI

With `WITH_TESTS=ON` and `WITH_PYTHON_INTERFACE=ON`, `make test` (or `ctest`)
includes the Python fast suite when `WITH_FAST_TEST=ON`, and the regular suite
when `WITH_REGULAR_TEST=ON`. Benchmarks are run manually. With the Python interface
disabled, no Python tests are registered. Each script is registered separately as
`gundam-python-<suite>-<script>`, with its own result and duration.

CTest uses the last interpreter that successfully configured a Python interface,
and loads the module from the build directory.

By default, a CTest setup fixture
creates `tests/python-venv` inside the build directory using that interpreter
and installs `requirements.txt`. The environment inherits its system packages
and is reused until the requirements or selected interpreter change. The first
`make test` needs network access if packages are not already available locally.
No packages are installed into the system Python.

To use a prepared environment without automatic installation, configure with
`-DWITH_PYTHON_TEST_VENV=OFF` and
`-DGUNDAM_PYTHON_VERSIONS=/absolute/path/to/venv/bin/python`.
Install the requirements into that environment before running tests.

The GitHub Docker CI prepares its venv, installs `requirements.txt` and fetches
C++ dependency sources before configuring GUNDAM. CMake uses those local sources
with `FETCHCONTENT_FULLY_DISCONNECTED=ON` and disables automatic test venv
preparation with `WITH_PYTHON_TEST_VENV=OFF`. The existing `make test` step then runs
the Python tests too, and Python failures fail the CI check.

## How to run the tests manually?

Install the Python dependencies into your chosen environment:

```bash
cd $GUNDAM_REPO_DIR
python3 -m venv --system-site-packages tests/python/venv
source tests/python/venv/bin/activate
python -m pip install -r tests/python/requirements.txt
```

From the repository root:

```bash
python tests/python/run.py # run fasts tests
```

Scripts can also be launched directly from a writable working directory:

```bash
python tests/python/fast/510NumericsIgnoreZeroPredictionAtPrior.py
```


Command line options:

```bash
-v, --verbose # display test output live and keep it in test.log
--list # show tests file names
--output <folder-path> # place where logger files are writen. By default it will be in a temp folder
--test <python-test-filename> # runs a specific test
--suite <name-of-the-suite> # runs the tests of the selected suite [fast, regular, benchmarks]
```

For live output:

```bash
python tests/python/run.py -v
```

Sub command options can be passed. Useful for benchmarks:
```bash
python tests/python/run.py --suite benchmarks --test 761BackendsCudaWeightBenchmark.py -- --help
```


## How tests are organized?

These tests run directly with Python, independently of `tests/gundam-tests.sh`.
They use the interpreter you launch; they do not create an environment, install
packages or switch interpreters automatically.

- `fast/`: runtime regression tests using the Python interface.
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

Backend-specific scripts use `7XXBackends<Subcategory><Subject>.py`, with the
exact subcategory spelling `Common`, `Multiple`, `Cpu`, `Mps` or `Cuda`:

| Range | Subcategory | Scope |
|---|---|---|
| **700–709** | `Common` | Backend-neutral checks with no implementation specialization; one selected backend per invocation |
| **710–719** | `Multiple` | Several target backends, such as MPS and CUDA, or backend switching contracts |
| **720–739** | `Cpu` | CPU implementations, execution and numerical behavior |
| **740–759** | `Mps` | Apple Metal implementations and numerical behavior, including CPU reference checks |
| **760–799** | `Cuda` | NVIDIA CUDA implementations, profiling and tooling, including CPU reference checks |

Within each suite, assign consecutive identifiers to the existing tests in each
subcategory. The ranges leave room for additions while keeping each group
contiguous when sorted by filename. CPU is always available in a backend build:
using it as a reference does not make a test `Multiple`. CPU/MPS comparisons
belong to `Mps`; CPU/CUDA comparisons belong to `Cuda`. Apply the same rule to
benchmark wrappers and the regression tests for those wrappers.

Use `Multiple` when the test covers several target implementations, such as MPS
and CUDA (including opt-in cases), or tests switching across backend types.
`Common` is reserved for backend-neutral tests without implementation-specific
behavior; a generic test accepting one `--backend` selection belongs there.

Numbers are independent between suites: for example,
`fast/720BackendsCpuPropagation.py` and `benchmarks/720BackendsCpuBenchmark.py`
share a number. Both suites start `Common` at `700` and `Multiple` at `710`.

Choose a category by the test's primary purpose. A general feature test may also
exercise a backend without belonging to `7XX`. Model-content checks verify the
loaded objects and data; numerical-result checks verify the calculated values.
Leave gaps when assigning new numbers, and keep a test's number when moving it
between `fast` and `regular`, provided it does not collide with another test.
Numbers do not imply execution dependencies: every script must run independently.
Use PascalCase for the category and subject (`Api`, `FitGradient`), keeping
backend subcategories capitalized as `Common`, `Multiple`, `Cpu`, `Mps` and `Cuda`. Avoid
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

Some tests compile small native libraries and require a C++ compiler. The two
gradient checks `570NumericsFitGradient.py` and
`580NumericsFitGradientSpline.py` also require the ROOT executable: they prepare
their own inputs using the shared fixtures in `tests/fast-tests`, without running
the standard test suite. GPU tests retain their existing device requirements and
opt-in environment variables. Benchmark commands and prerequisites are documented
in [benchmarks/README.md](benchmarks/README.md).

When validating a specific build on macOS, `PYTHONPATH` alone does not guarantee
that its native libraries are loaded. `DYLD_LIBRARY_PATH` can override the Python
extension itself or its dependencies with an installed GUNDAM version. Use the
matching Python interpreter and remove library search overrides for an isolated
check (replace `/path/to/build` below):

```bash
env -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  PYTHONPATH=/path/to/build/src/PythonInterface \
  python tests/python/run.py --output /tmp/gundam-python-isolated
```

Set `DYLD_PRINT_LIBRARIES=1` to inspect the actual native library paths on macOS.
Checking only `GUNDAM.__file__` is insufficient to establish their provenance.
A script that exits successfully after skipping unsupported features does not
validate compatibility with those features.

Fast tests `110`, `410` and `420` keep representative cases. Their matching
`*Full.py` tests in `regular/` preserve the full scenario matrices. Run them with
`python tests/python/run.py --suite regular --test <name>Full.py` or enable
`WITH_REGULAR_TEST` for CTest.

CTest labels Python scripts with `python` and `python-fast` or `python-regular`.
The `fast`, `regular`, `extended` and `slow` labels identify the selected shell
suite run by `gundam-tests`; broader shell suites include the faster suites.
Label totals overlap: `python` includes both Python suites and environment setup.

Fast tests `560` use representative cases or small deterministic inputs.
Their matching `*Full.py` tests in `regular/` retain the full validation.

Fast tests `570` and `580` use representative cases or small deterministic inputs.
Their matching `*Full.py` tests in `regular/` retain the full validation.

Fast tests `711`, `726` and `743` keep representative backend scenarios; their
matching `*Full.py` tests in `regular/` retain the complete validation.
