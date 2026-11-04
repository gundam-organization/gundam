# Testing and validation for GUNDAM

The tests directory contains tools to run a series of tests on the
installed instance of GUNDAM.  The GUNDAM executables and ROOT must
be in the path.

## Unit tests for the libraries (run using GoogleTest)

The tests in the GTests directory are a set of unit tests run using
GoogleTest.  This is where the library routines should be tested.  Most of
the interfaces they cover are not visible to users.

## Tests for the user interfaces (the executables and python interfaces)

The user visible interfaces are directly tested by running GUNDAM as a
stand-alone fitter, or through the python interface, and then verifying the
outputs.  These tests are run by `gundam-tests.sh`, which must be run from
the tests directory that contains it.  It runs nothing unless the apply
option (`-a`) is given:

```bash
cd ./tests
bash gundam-tests.sh -a
```

Run the script without `-a` to see the options and the list of tests it
would run.  The output of a run goes into a new directory, by default
`./output.YYYY-MM-DD-hhmmss`, which must not already exist (the default
name has one second resolution, so two runs started in the same second
collide).

### Testing levels

The tests in `fast-tests` are always run.  Tests in the other directories
are run only when the applicable option is given.  The levels run in order
of increasing cost, so a later level can use the output of an earlier one.

- `fast-tests/`: always run, and used during continuous integration.
- `regular-tests/`: quick tests that are not used for continuous
  integration, but that should be run locally before a push or a pull
  request.  Run when `-r` is given.  (Plan to get a drink of water while
  these tests run.)
- `extended-tests/`: slower tests, run when `-e` is given.  Each of them
  should finish in well under 30 seconds, and all of them together in less
  than a few minutes.  (Plan to take a coffee break while these tests run.)
- `slow-tests/`: long validation tests, run only when `-s` is given, after
  every other test has finished.  (Plan to work on something else while
  these tests run.)

### How a test is run

A standard test script can be any executable file except a Python script, and is
generally written in bash. Python tests use a separate runner. Each standard
test is run in the output directory with the command line

```bash
cd <output> && <script> <directory>
```

where `<output>` is the directory the script is run in, `<script>` is the
full path of the test script, and `<directory>` is the full path of the
directory that contains it.  A test finds its own configuration files
through `<directory>`, so they are kept beside the script.

`gundam-tests.sh` runs every executable file in the selected test
directories whose name starts with a digit, and prints the list of them
before it starts.  Within a directory they run in lexical order by name, so
`001MyName` runs before `002MyName` and the naming convention below
controls the order.

A test passes when it exits with a zero status and its log does not end
with a failure message.  Tests that are expected to fail are listed in the
`EXPECTED_FAILURES` file by their path relative to the tests directory.
The framework has its own example of one, `fast-tests/090ExpectedFailure.sh`.

### Script naming convention

- `000-099`: reserved for `gundam-tests.sh`.  This is where job headers
  and similar things are generated, and where the run environment is set
  up.  A script here builds something that the later tests need, so a test
  that is run on its own with `-t` will not have it.
- `100-199`: scripts which do not require input, including any script
  that generates input data for the later tests.
- `200-299`: scripts which generate GUNDAM output files.  These mostly
  apply fits.
- `800-899`: scripts which produce summary files.
- `900-998`: scripts which look at summary files and check the results.
- `999`: reserved for `gundam-tests.sh`, where job completion information
  is generated.

As an example, a script that runs a GUNDAM fit taking a binning file and a
configuration file might be named like this:

```
fast-tests/
  200RunGUNDAM.sh          -- The script
  200RunGUNDAM-config.yaml -- The configuration file
  200RunGUNDAM-binning.txt -- The binning file
  200RunGUNDAM-utils.py    -- Python utilities used by the script
```

The output file should be named `200RunGUNDAM.root`, or something similar
as needed.

### Support files

A file named `<NNNName>-<role>.<ext>` is support material for the test
`<NNNName>`, and is never a test itself.  Such a file is found by the
search for test scripts, and is skipped only because it cannot be
executed, so it must never be given the executable bit.  A python file of
this kind holds utilities that the test imports, and has no `#!` line.

### Python tests

Python tests live separately in [`python/`](python/README.md), organized into
`fast`, `regular` and `benchmarks`. They are excluded from `gundam-tests.sh`.
CTest includes these suites when `WITH_TESTS=ON` and `WITH_PYTHON_INTERFACE=ON`,
using the last interpreter selected for an interface build. `WITH_FAST_TEST`
and `WITH_REGULAR_TEST` control the corresponding Python suites. Dependencies
must be installed in that interpreter beforehand.

For manual execution, use the Python interpreter matching your GUNDAM build:

```bash
python python/run.py --list
python python/run.py --output /tmp/gundam-python-fast
python python/run.py --suite regular --output /tmp/gundam-python-regular
```

Install dependencies from `python/requirements.txt` into your chosen environment
and expose the GUNDAM interface through `PYTHONPATH`. No Python environment is
created by the standard shell tests. See [python/README.md](python/README.md) for
single-test execution, prerequisites and benchmarks.
