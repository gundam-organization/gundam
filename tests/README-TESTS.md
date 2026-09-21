# Testing and validation for GUNDAM

The tests directory contains tools to run a series of tests on the
installed instance of GUNDAM.  The gundam executables, and ROOT must
be in the path.  See the gundam-tests.sh script for more detailed
documentation.

## Python tests

Tests written in python need packages that are not part of the standard
library.  They are listed, with the test that needs them, in
`fast-tests/005PythonSetup-requirements.txt`.

Those packages are installed into a python virtual environment that
`fast-tests/005PythonSetup.sh` builds at the start of every run.  The
environment is created in the run directory, which is the output directory
that `gundam-tests.sh` makes for the run, so it is thrown away with the rest
of the output and nothing is cached from one run to the next.  It is created
with `--system-site-packages`, so a package that is already installed on the
machine is used as it is and nothing is downloaded.

A python test finds the environment through
`fast-tests/005PythonSetup-utils.py`, which is part of the setup test and
looks for `venv` in the directory the test is run in, then re-runs the test
with the interpreter it finds there.  A test never builds the environment
itself.
This means a python test fails when it is run on its own, either with
`gundam-tests.sh -t` or directly from a shell, because the setup script did
not run for it.  Run the whole set of fast tests instead.

A python test also needs the GUNDAM python interface, which is not a pip
package.  GUNDAM must be built with `-D WITH_PYTHON_INTERFACE=ON`, and
`$PYTHONPATH` must point at the installed library, which the `setup.sh` in
the install directory does.  `fast-tests/212EvalFromLibCompatibility.py`
additionally needs a `g++` compiler, since it builds a library while it runs.

To add a package, put it in `fast-tests/005PythonSetup-requirements.txt`
with a comment saying which test needs it.  The name is also imported as a
smoke test by `fast-tests/005PythonSetup.sh`, so it has to be the module name
as well.
