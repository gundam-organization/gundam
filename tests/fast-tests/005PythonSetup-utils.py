"""Find the python environment that the GUNDAM python tests run inside.

This is part of 005PythonSetup.sh, which is the test that builds the
environment.  It is kept beside that script and named after it, because the
two have to agree about where the environment is.  The hyphen says it is
support material for that test rather than a test itself, in the same way as
005PythonSetup-requirements.txt.  It must not be executable: gundam-tests.sh
finds candidate tests with "find -name [0-9]*", so this file is found by that
search and is skipped only because it cannot be executed.

The python tests need packages (see 005PythonSetup-requirements.txt) that
are not part of
the standard library.  Those packages are installed into a virtual
environment that 005PythonSetup.sh builds at the start of every test run.
The environment lives in the run directory (the output directory that
gundam-tests.sh creates and runs every test in), so it is discarded with the
rest of the run and nothing is cached between runs.

This module does not create anything.  A test that is run on its own, either
with "gundam-tests.sh -t" or directly from the command line, will not have
had 005PythonSetup.sh run for it, and it is supposed to fail.

A python test uses this by re-executing itself inside the environment before
it imports anything that is not part of the standard library.  The module
name starts with a digit and holds a hyphen, neither of which an import
statement can express, so it is imported by name:

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.dont_write_bytecode = True
    from importlib import import_module

    import_module("005PythonSetup-utils").requireVenv(__file__)

Bytecode writing is turned off so that importing this leaves no __pycache__
behind in the test directory: a test run must not change anything outside
its own output directory.

A python test in another test directory would insert the path of this
directory instead, e.g. Path(__file__).resolve().parents[1] / "fast-tests".
"""

import os
import sys
from pathlib import Path

# The name of the virtual environment directory, relative to the directory
# that the test is run in.  005PythonSetup.sh must agree with this.
VENV_DIR_NAME = "venv"


def venvPython(runDir_=None):
    """Return the interpreter of the run directory virtual environment.

    Returns None when the virtual environment does not exist.  The run
    directory defaults to the current working directory since gundam-tests.sh
    always runs a test in the output directory.
    """
    runDir = Path(runDir_) if runDir_ is not None else Path.cwd()
    python = runDir / VENV_DIR_NAME / "bin" / "python"
    return python if python.exists() else None


def insideVenv():
    """Return true when the running interpreter is inside a virtual environment."""
    return sys.prefix != sys.base_prefix


def requireVenv(script_=None):
    """Re-execute the calling script inside the test virtual environment.

    Does nothing when the interpreter is already inside a virtual environment,
    so a test that re-runs itself through sys.executable stays where it is.
    Otherwise the script is re-executed with the interpreter of the run
    directory virtual environment.  The environment is inherited, so $PYTHONPATH
    still finds the GUNDAM python interface.

    Exits with a failure when the virtual environment does not exist.  Building
    it is the job of fast-tests/005PythonSetup.sh, not of a test.
    """
    if insideVenv():
        return

    python = venvPython()
    if python is None:
        runDir = Path.cwd()
        print("FAIL: no python environment in the run directory " + str(runDir))
        print("FAIL: it is built by fast-tests/005PythonSetup.sh, which did not run")
        sys.stdout.flush()
        sys.exit(1)

    script = os.path.abspath(script_ if script_ is not None else sys.argv[0])
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(python), [str(python), script] + sys.argv[1:])
