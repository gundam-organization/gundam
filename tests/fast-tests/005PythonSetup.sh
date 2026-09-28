#!/bin/bash
#
# Build the python environment that the python tests run inside.
#
# The python tests need packages that are not part of the standard library.
# They are listed in 005PythonSetup-requirements.txt, beside this script,
# and are installed into a virtual environment that is created here, in the
# run directory, so that it is thrown away with the rest of the output and
# nothing is cached between runs.  The environment is created with
# --system-site-packages, so a package that is already installed on the
# machine is reused instead of being downloaded.
#
# The tests find this environment with 005PythonSetup-utils.py, which is
# part of this test and looks for "venv" in the directory the test is run in.
# The VENV_DIR_NAME there and the VENV set below must agree.
#
# A python test that is run on its own does not get this script run for it,
# and is supposed to fail.

# Get the directory containing the script from the command line
# parameters (avoids bash trickery).  Use the current directory as the
# default.
DIR=.
if [ ${#1} -gt 0 ]; then
    DIR=${1}
fi

# The environment is built in the directory the test is run in.  The
# packages it holds are part of this test, so they are listed beside it.
BASE=005PythonSetup
VENV=${PWD}/venv
REQUIREMENTS=${DIR}/${BASE}-requirements.txt

echo Building the python test environment in ${VENV}

# Make sure that python is available.
if ! which python3; then
    echo FAIL: Executable not found for python3
    exit 1
fi
python3 --version

# Create the virtual environment.  This needs the venv module, which some
# distributions package separately (e.g. the python3-venv package on ubuntu).
if [ ! -d ${VENV} ]; then
    if ! python3 -m venv --system-site-packages ${VENV}; then
        echo FAIL: Could not create a virtual environment in ${VENV}
        echo FAIL: The python venv module may need to be installed
        exit 1
    fi
fi

PYTHON=${VENV}/bin/python
if [ ! -x ${PYTHON} ]; then
    echo FAIL: No interpreter in the virtual environment: ${PYTHON}
    exit 1
fi

# Install the requirements.  Packages that are already installed on the
# machine are visible in the environment, so this does not need the network
# unless something is actually missing.
if [ ! -f ${REQUIREMENTS} ]; then
    echo FAIL: Requirements file not found: ${REQUIREMENTS}
    exit 1
fi

echo REQUIRED PYTHON PACKAGES
cat ${REQUIREMENTS}

if ! ${PYTHON} -m pip install -r ${REQUIREMENTS}; then
    echo FAIL: Could not install the packages in ${REQUIREMENTS}
    exit 1
fi

# Check that every requirement can actually be imported, and record the
# version the tests will be using.  The requirement names are also the module
# names (see the comments in the requirements file), so strip any comment and
# version specifier and import what is left.
MISSING=""
echo PYTHON TEST PACKAGES
for PACKAGE in $(sed -e 's/#.*$//' -e 's/[<>=!~;[].*$//' -e 's/[[:space:]]//g' ${REQUIREMENTS}); do
    if [ ${#PACKAGE} -eq 0 ]; then
        continue
    fi
    if ! ${PYTHON} -c "
import ${PACKAGE}
print('${PACKAGE}', getattr(${PACKAGE}, '__version__', 'unknown version'),
      getattr(${PACKAGE}, '__file__', 'unknown location'))
"; then
        MISSING="${MISSING} ${PACKAGE}"
    fi
done

if [ ${#MISSING} -gt 0 ]; then
    echo FAIL: Required packages cannot be imported:${MISSING}
    exit 1
fi

# Record what the tests will be running with.
echo PYTHON TEST INTERPRETER
${PYTHON} -c "import sys; print(sys.executable, sys.version)"

echo SUCCESS: The python test environment is ready

# End of the script
