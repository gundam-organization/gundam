# Validations for that should be run before a push or pull-request

This directory contains scripts that will be run as part of the
gundam-tests.sh regular testings.  These tests are not run as part of
continuous integration, but should be run locally before making a
push/pull-request.  These tests are checking critical features and the
fit precision.  They could be run during continuous integration, but
are placed here to limit the total CI time.

Each script should finish in a "few" seconds, and all of the scripts
should finish in a couple minutes.  See the gundam-tests.sh script for
documentation on the file naming convention.

This file should exist so that git will create the regular-tests
subdirectory

Compiled regular tests are also available through CTest when configured with
`WITH_REGULAR_TEST=ON`. In particular, `gundamGTest_regularTest.exe` runs a small
fit and compares fitted values and the full Hesse covariance for original/CPU
and CPU/CUDA (both propagation modes). These native tests do not require the
Python interface and are not enumerated by `gundam-tests.sh`.

```bash
cmake --build <build-directory> --target gundamGTest_regularTest.exe -j3
ctest --test-dir <build-directory> -R '(BackendFitCovariance|CudaFitCovariance)' --output-on-failure
```

CUDA tests skip successfully if the required backend or device is unavailable.
Set `GUNDAM_REQUIRE_CUDA_TESTS=1` in a GPU job to require their execution.
See [fit and covariance regression](../../docs/backends/BackendFitCovariance.md)
for configuration, Slurm commands, comparison tolerances and JSON reports.
