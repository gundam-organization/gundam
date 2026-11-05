# Unit tests written using GoogleTest

This directory contains the source files for unit tests written using
the GoogleTest library.  The GoogleTest library will need to have been
externally installed on the system running the tests.  If it is found
by cmake, then these tests will be added to `make test` and run under
the control of ctest.


## Categories

CTest names start with a functional category. Parameterized test names keep
GoogleTest's backend and parameter suffixes within that category.

| Category | Coverage | CTest label |
| --- | --- | --- |
| Numerics | Interpolation, splines and monotonicity | `unit-numerics` |
| Dials | Definitions, loading, sharing, responses and formulas | `unit-dials` |
| DataLoading | ROOT readers, friend files and resource lifetime | `unit-dataloading` |
| Likelihood | Statistical probability functions | `unit-likelihood` |
| Minimizer | Algorithms, configuration and exception handling | `unit-minimizer` |
| Backends | Layouts, caches, CPU/MPS/CUDA propagation, gradients and comparisons | `unit-backends` |

All these tests also carry the `unit` label. For example:

```sh
ctest --test-dir cmake-build-debug --output-on-failure -L '^unit$'
ctest --test-dir cmake-build-debug --output-on-failure -L '^unit-dials$'
ctest --test-dir cmake-build-debug -R '^Numerics/'
```

Fast unit sources are grouped by category in `tests/CMakeLists.txt`, each with
its own executable. They are always compiled; `WITH_FAST_TEST` controls their
CTest registration. Regular backend and legacy cache tests use the same category
prefixes and labels, retaining their suite and host/device labels.

Logger tests belong to `submodules/simple-cpp-logger/tests` and are disabled in
the GUNDAM build. Their header-only and compiled variants check runtime logging
behavior and shared state across translation units in the Logger's own build.
