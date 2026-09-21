#!/bin/bash
#
# Run the GUNDAM validation tests.
#
# This must be run from the tests directory that contains it, and it runs
# nothing unless the apply option ("-a") is given.
#
# See README-TESTS.md in this directory for the testing levels, how a test
# script is run, the script naming convention, the support file convention,
# and the environment that the python tests need.

echo 'USAGE: gundam-tests.sh [-f] [-r] [-e] [-s] [-v] [-a] [output-directory]'
echo '    -c               : Force use of terminfo colors for output'
echo '    -f               : Only run the fast tests [default]'
echo '    -r               : Run fast and regular tests'
echo '    -e               : Run fast, regular and extended tests'
echo '    -s               : Run all tests including the slow tests'
echo '    -t <test-path>   : Run only one test script, e.g. fast-tests/200CovarianceFit.sh'
echo '                       (the 000-099 setup scripts are not run, so a test'
echo '                       needing them will fail)'
echo '    -v               : Print test logs live while also saving them to the log files'
echo '    -a               : Apply the tests (no tests are run without this)'
echo '    output-directory : The name of the output directory.  The default'
echo '                       value is \"./output.YYYY-MM-DD-hhmmss\"'
echo ' See README-TESTS.md for more documentation.'

# The default tests to be run.
TESTS="fast-tests"

# Handle any input arguments
while getopts ":acfvrest:" opt; do
    case "${opt}" in
        c)
            USE_COLORS="yes"
            ;;
        v)
            VERBOSE_LOGS="yes"
            ;;
        f)
            TESTS="fast-tests"
            ;;
        r)
            TESTS="fast-tests regular-tests"
            ;;
        e)
            TESTS="fast-tests regular-tests extended-tests"
            ;;
        s)
            TESTS="fast-tests regular-tests extended-tests slow-tests"
            ;;
        t)
            SINGLE_TEST="${OPTARG}"
            ;;
        a)
            APPLY="yes"
            ;;
        :)
            echo "Missing argument for -${OPTARG}"
            exit 1
            ;;
        \?)
            echo "Unknown option: -${OPTARG}"
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

echo
echo Requesting tests in ${TESTS}

# Result when this script has a problem.
RESULT_ERROR="ERROR:"
RESULT_WARNING="WARNING:"

# Result for a particular sub job.
RESULT_JOB_FAILURE="JOB FAILURE:"
RESULT_JOB_SUCCESS="JOB SUCCESS:"

# Result for the test.  These report the result to the testing harness.
RESULT_FAILURE="FAIL:"
RESULT_SUCCESS="SUCCESS:"

# Add colors to the results (on terminals only)
if [ -t 1 -o ${USE_COLORS}x == "yesx" ]; then
    TERMINFO_INIT=$(tput init)
    TERMINFO_RED=$(tput setaf 1)
    TERMINFO_YELLOW=$(tput setaf 3)
    TERMINFO_GREEN=$(tput setaf 2)
    RESULT_ERROR=${TERMINFO_RED}${RESULT_ERROR}${TERMINFO_INIT}
    RESULT_WARNING=${TERMINFO_YELLOW}${RESULT_ERROR}${TERMINFO_INIT}
    RESULT_JOB_FAILURE=${TERMINFO_RED}${RESULT_JOB_FAILURE}${TERMINFO_INIT}
    RESULT_JOB_SUCCESS=${TERMINFO_GREEN}${RESULT_JOB_SUCCESS}${TERMINFO_INIT}
    RESULT_FAILURE=${TERMINFO_RED}${RESULT_FAILURE}${TERMINFO_INIT}
    RESULT_SUCCESS=${TERMINFO_GREEN}${RESULT_SUCCESS}${TERMINFO_INIT}
fi

# Find the name of the output directory.  It might have been provided
# on the command line.
OUTPUT_DIR="output.$(date +%Y-%m-%d-%H%M%S)"  # A default name for the output
if [ ${#1} -gt 0 ]; then
    # A name was provided on the command line.
    OUTPUT_DIR=${1}
fi

echo Output will be in ${OUTPUT_DIR}

# Make sure the output directory does not exist.
if [ -x ${OUTPUT_DIR} ]; then
    echo -e ${RESULT_ERROR} Output directory already exists ${OUTPUT_DIR}
    exit 1
fi

echo Running in ${PWD}
if [ ! -x ./gundam-tests.sh ]; then
    echo -e ${RESULT_ERROR} Must be run from the directory containing gundam-tests.sh
    exit 1
fi

is_test_runnable() {
    local test_path=$1
    if [ -x "${test_path}" ]; then
        return 0
    fi
    return 1
}

SINGLE_TEST_FOUND="no"

for i in ${TESTS}; do
    if [ -x ${PWD}/${i} ]; then
        echo Testing directory found: $i
        for j in $(find ${i} -name "[0-9]*" -type f | grep -v "~" | sort); do
            if ! is_test_runnable "${j}"; then
                continue
            fi
            if [ -n "${SINGLE_TEST}" ] && [ "${j}" != "${SINGLE_TEST}" ]; then
                continue
            fi
            SINGLE_TEST_FOUND="yes"
            echo '   Will run:' $j
        done
    fi
done

if [ -n "${SINGLE_TEST}" ] && [ "${SINGLE_TEST_FOUND}" != "yes" ]; then
    echo
    echo -e ${RESULT_ERROR} Requested test not found or not runnable: ${SINGLE_TEST}
    exit 1
fi

if [ ! -f EXPECTED_FAILURES ]; then
    echo -e ${RESULT_ERROR} EXPECTED_FAILURES file must exist, but it can be empty.
    exit 1
fi

if [ ${APPLY}x != "yesx" ]; then
    echo
    echo -e ${RESULT_ERROR} Tests not run. Add the -a option to run the test.
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
    echo -e ${RESULT_ERROR} OUTPUT DIRECTORY WAS NOT CREATED: ${OUTPUT_DIR}
    exit 1
fi

# Find and run the jobs in lexical order.
FAILURES=""
EXPECTED=""
for d in ${TESTS}; do
    if [ ! -x ${PWD}/${d} ]; then
        echo -e ${RESULT_WARNING} TESTING DIRECTORY ${d} DOES NOT EXIST
        continue;
    fi
    for i in $(find ${d} -name "[0-9]*" -type f | grep -v "~" | sort); do
        if [ -n "${SINGLE_TEST}" ] && [ "${i}" != "${SINGLE_TEST}" ]; then
            continue
        fi
        JOB=${PWD}/${i}
        if ! is_test_runnable "${JOB}"; then
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
        if [ "${VERBOSE_LOGS}" = "yes" ]; then
            (
                cd $OUTPUT_DIR &&
                ${JOB} ${DIR} 2>&1 | tee ${LOG}
                exit ${PIPESTATUS[0]}
            )
            JOB_STATUS=$?
        elif (cd $OUTPUT_DIR && ${JOB} ${DIR} >& ${LOG}); then
            JOB_STATUS=0
        else
            JOB_STATUS=$?
        fi
        if [ ${JOB_STATUS} -eq 0 ]; then
            # The job exited with success, but look for a fail messsage
            if (tail -5 ${OUTPUT_DIR}/${LOG} | grep FAIL >> /dev/null); then
                echo -e ${RESULT_JOB_FAILURE} ${i}
            elif (tail -10 ${OUTPUT_DIR}/${LOG} | grep "Execution.*aborted" >> /dev/null); then
                echo -e ${RESULT_JOB_FAILURE} ${i}
            else
                echo -e ${RESULT_JOB_SUCCESS} ${i}
                SUCCESS="yes"
            fi
        else
            echo -e ${RESULT_JOB_FAILURE} ${i}
        fi
        if [ ${SUCCESS} = "yes" ]; then
            # The job succeeded, make sure it's not in EXPECTED_FAILURES
            if (grep -F $i EXPECTED_FAILURES >> /dev/null); then
                cat ${OUTPUT_DIR}/${LOG}
                echo -e ${RESULT_JOB_FAILURE} Expected $i to fail
                FAILURES="${FAILURES} unexpected-success:\"${JOB}\""
            fi
        else
            # The job failed, check if it was expected
            if (grep -F $i EXPECTED_FAILURES >> /dev/null); then
                cat ${OUTPUT_DIR}/${LOG}
                echo -e ${RESULT_JOB_SUCCESS} Failure was expected for $i
                EXPECTED="${EXPECTED} \"${JOB}\""
            else
                cat ${OUTPUT_DIR}/${LOG}
                FAILURES="${FAILURES} unexpected-failure:\"${JOB}\""
            fi
        fi
    done
done

if [ ${#EXPECTED} -gt 0 ]; then
    echo
    echo Expected Failures:
    for i in ${EXPECTED}; do
        echo EXPECTED FAILURE: $i
    done
fi

if [ ${#FAILURES} -gt 0 ]; then
    echo
    echo Failed Jobs:
    for i in ${FAILURES}; do
        echo UNEXPECTED FAILURE: $i
    done
    echo
    echo -e ${RESULT_FAILURE} Tests failed
    exit 1
else
    echo
    echo -e ${RESULT_SUCCESS} Tests succeeded
fi
# End of the script
