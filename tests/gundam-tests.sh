#!/bin/bash
#
# Run tests on gundam.  Test scripts that are kept in the fast-tests
# subdirectory will always be run.  Any test scripts that take a lot
# of time and are for more detailed validation should be kept in
# slow-tests.  The slow-tests will only be run if the "-s" option is
# provided.  The apply option ("-a") must be added to actually run the
# scripts.
#
# This needs to be run in the tests subdirectory (which contains this
# script).  Any tests that are expected to fail should be listed in
# the EXPECTED_FAILURES file (by file name relative to the tests
# directory) where there is an example called
# "fast-tests/090ExpectedFailure.sh" that is part of the testing
# framework.
#
# Validation scripts can be any executable file, but are generally
# written in bash or python.  They are run in a separate execution
# directory with command line 
#
# cd <output> && <script> <directory>
#
# Where <output> is directory where the script is run, <script> is the
# full path of the test script, and <directory> is the full path of
# the directory containing the test script.  Any necessary
# configuration files should be saved in the same directory as the
# script.
#
# The gundam-tests.sh script will run all of the executable scripts in
# the script directories (i.e. fast-tests and/or slow-tests) that
# start with a digit.  The list of scripts to be run are printed
# before they start to run.
#
# The validation scripts are run in lexical order based on the script
# name.  This means that script "001MyName" is run before "002MyName",
# and allows users to specify the script order. The following
# convention is suggested for script naming.
#
#    000-099 -- Reserved for gundam-tests.sh.  This is where job
#               headers and similar things can be generated.
#
#    100-199 -- Scripts which don't require input.  This includes any
#               scripts generating input data that can be used by the
#               later tests.
#
#    200-299 -- Scripts which generate gundam output files.
#
#    800-899 -- Scripts which produce summary trees.
#
#    900-998 -- Scripts looking at summary trees and checking results
#
#    999 -- Reserved for gundam-tests.sh.  This is where job
#               completion information is generated.
#
# NAMING CONVENTION EXAMPLE: This is how the naming convention works
# in practice.  This is how a script that runs a GUNDAM fit that takes
# a binning and configuration file might be named.
#
#   fast-test/
#     200RunGUNDAM.sh          -- The script
#     200RunGUNDAM-config.yaml -- The configuration file
#     200RunGUNDAM-binning.txt -- The binning file.
#
#   The output file should be named 200RunGUNDAM.root (or similar as
#   needed).

echo 'USAGE: gundam-tests.sh [-s] [-a] [output-directory]'
echo '    -s               : Run the slow tests'
echo '    -a               : Apply the tests (no tests are run without this)'
echo '    output-directory : The name of the output directory.  The default'
echo '                       value is \"./output.YYYY-MM-DD-hhmm\"'
echo ' See gundam-tests.sh for more usage documentation.'
TESTS="fast-tests"

# Handle any input arguments
TEMP=$(getopt -o 'abs' -n "$0" -- "$@")
if [ $? -ne 0 ]; then
    echo "Error ..."
    exit 1
fi
eval set -- "$TEMP"
unset TEMP
while true; do
    case "$1" in
	'-s')
	    TESTS="${TESTS} slow-tests"
	    shift
	    continue;;
        '-a')
            APPLY="yes"
            shift
            continue;;
	'--')
	    shift
	    break;
    esac
done

echo
echo Requesting tests in ${TESTS}

# Find the name of the output directory.  It might have been provided
# on the command line.
OUTPUT_DIR="output.$(date +%Y-%m-%d-%H%M)"  # A default name for the output
if [ ${#1} -gt 0 ]; then
    # A name was provided on the command line.
    OUTPUT_DIR=${1}
fi

echo Output will be in ${OUTPUT_DIR}

# Make sure the output directory does not exist.
if [ -x ${OUTPUT_DIR} ]; then
    echo ERROR: Output directory already exists ${OUTPUT_DIR}
    exit 1
fi

echo Running in ${PWD}
if [ ! -x ./gundam-tests.sh ]; then
    echo ERROR: Must be run from the directory containing gundam-tests.sh
    exit 1
fi

for i in ${TESTS}; do
    if [ -x ${PWD}/${i} ]; then
        echo Testing directory found: $i
        for j in $(find ${i} -name "[0-9]*" -type f | grep -v "~" | sort); do
            if [ -x ${j} ]; then
                echo '   Will run:' $j
            fi
        done
    fi
done

if [ ! -f EXPECTED_FAILURES ]; then
    echo ERROR: EXPECTED_FAILURES file must exist, but it can be empty.
    exit 1
fi

if [ ${APPLY}x != "yesx" ]; then
    echo
    echo WARNING: Add the -a option to run the test.
    exit 1
fi


###################################################################
#
# Start the actual testing.
#
###################################################################

# Make sure the output directory has been created
mkdir -p ${OUTPUT_DIR}

# Make sure the output directory was correctly created (i.e. it exists)
if [ ! -x ${OUTPUT_DIR} ]; then
    echo OUTPUT DIRECTORY WAS NOT CREATED: ${OUTPUT_DIR}
    exit 1
fi

# Find and run the jobs in lexical order.
FAILURES=""
EXPECTED=""
for i in $(find ${TESTS} -name "[0-9]*" -type f | grep -v "~" | sort); do
    JOB=${PWD}/${i}
    # Only run files that are executable
    if [ ! -x ${JOB} ]; then
        continue;
    fi
    # SUCCESS is false by default.
    SUCCESS="no"
    # Get the full path to the script.  This is passed to the script
    # so the script can easily find any input files.
    DIR=$(dirname ${JOB})
    # The name of the output log file
    LOG=$(basename ${JOB}).log
    # Run the script in the output directory.
    echo "(cd $OUTPUT_DIR && ${JOB} ${DIR})"
    if (cd $OUTPUT_DIR && ${JOB} ${DIR} >& ${LOG}); then
        # The job exited with success, but look for a fail messsage
        if (tail -5 ${OUTPUT_DIR}/${LOG} | grep FAIL >> /dev/null); then
            echo JOB FAILURE: ${i}
        else
            echo JOB SUCCESS: ${i}
            SUCCESS="yes"
        fi
    else
        echo JOB FAILURE: ${i}
    fi
    if [ ${SUCCESS} = "yes" ]; then
        # The job succeeded, make sure it's not in EXPECTED_FAILURES
        if (grep $i EXPECTED_FAILURES >> /dev/null); then
            cat ${OUTPUT_DIR}/${LOG}
            echo JOB FAILURE: Expected $i to fail
            FAILURES="${FAILURES} unexpected-success:\"${JOB}\""
        fi
    else
        # The job failed, check if it was expected
        if (grep $i EXPECTED_FAILURES >> /dev/null); then
            cat ${OUTPUT_DIR}/${LOG}
            echo JOB SUCCESS: Failure was expected for $i
            EXPECTED="${EXPECTED} \"${JOB}\""
        else
            cat ${OUTPUT_DIR}/${LOG}
            FAILURES="${FAILURES} unexpected-failure:\"${JOB}\""
        fi
    fi
done

if [ ${#EXPECTED} -gt 0 ]; then
    echo
    echo Expected Failures:
    for i in ${EXPECTED}; do
        echo EXPECTED: $i
    done
fi

if [ ${#FAILURES} -gt 0 ]; then
    echo
    echo Failed Jobs:
    for i in ${FAILURES}; do
        echo FAILED: $i
    done
    echo
    echo FAIL: Tests failed
    exit 1
else
    echo
    echo SUCCESS: Tests succeeded
fi
# End of the script
