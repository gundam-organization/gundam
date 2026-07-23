#!/bin/bash
#
# Run tests on gundam.  Test scripts that are kept in the fast-tests
# subdirectory will always be run.  Any test scripts that take a lot
# of time and are for more detailed validation should be kept in
# slow-tests.  Tests that are not part of "fast-tests" will only be
# run when the applicable options are set.  The apply option ("-a")
# must be added to actually run the scripts.
#
# The testing levels are:
#
#    fast-tests/ -- Always run and used during continuous integration.
#
#    regular-tests/ -- Quick tests that are not used for CI, but
#       should be run locally before a push/pull-request Run when "-r"
#       is provided.  They are run after and can use results from the
#       fast-tests.  (Plan to get a dring of water while these tests run).
#
#    extended-tests/ -- Slower tests that are run when "-e" is
#       provided.  These tests should finish in well under 30 seconds,
#       and all of the tests should take less than a few minutes.
#       They are run after and can use results from the fast and
#       regular tests. (Plan to take a coffee break while these tests
#       run).
#
#    slow-tests/ -- Long validation tests.  Only run with "-s" is
#       provided.  These tests are run after all other tests are
#       finished. (Plan to work on something else while these tests
#       run).
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
# before they start to run.  All of the fast-tests are run before all
# of the slow-tests (i.e. slow-tests can use output from fast-tests)
#
# The validation scripts are run in the order of increasing speed, so
# fast-tests are run before slow-tests.  Tests in a particular
# category (e.g. fast-tests) are run in lexical order based on the
# script name.  This means that script "001MyName" is run before
# "002MyName", so users have controll of the script order. The
# following convention is suggested for script naming.
#
#    000-099 -- Reserved for gundam-tests.sh.  This is where job
#               headers and similar things can be generated.
#
#    100-199 -- Scripts which don't require input.  This includes any
#               scripts generating input data that can be used by the
#               later tests.
#
#    200-299 -- Scripts which generate gundam output files.  These
#               scripts mostly apply fits.
#
#    800-899 -- Scripts which produce summary files.
#
#    900-998 -- Scripts looking at summary files and checking results
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

echo 'USAGE: gundam-tests.sh [-f] [-r] [-e] [-s] [-v] [-a] [output-directory]'
echo '    -c               : Force use of terminfo colors for output'
echo '    -f               : Only run the fast tests [default]'
echo '    -r               : Run fast and regular tests'
echo '    -e               : Run fast, regular and extended tests'
echo '    -s               : Run all tests including the slow tests'
echo '    -t <test-path>   : Run only one test script, e.g. fast-tests/210IgnoreZeroPredictionAtPrior.py'
echo '    -v               : Print test logs live while also saving them to the log files'
echo '    -a               : Apply the tests (no tests are run without this)'
echo '    output-directory : The name of the output directory.  The default'
echo '                       value is \"./output.YYYY-MM-DD-hhmm\"'
echo ' See gundam-tests.sh for more usage documentation.'

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
OUTPUT_DIR="output.$(date +%Y-%m-%d-%H%M)"  # A default name for the output
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

for i in ${TESTS}; do
    if [ -x ${PWD}/${i} ]; then
        echo Testing directory found: $i
        for j in $(find ${i} -name "[0-9]*" -type f | grep -v "~" | sort); do
            if [ -x ${j} ]; then
                if [ -n "${SINGLE_TEST}" ] && [ "${j}" != "${SINGLE_TEST}" ]; then
                    continue
                fi
                echo '   Will run:' $j
            fi
        done
    fi
done

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

PYTHON_TEST_VENV="${PWD}/venv"
PYTHON_TEST_VENV_PYTHON=""
PYTHON_TEST_VENV_READY="no"

resolve_python_test_interpreter() {
    local job=$1

    local shebang
    shebang=$(head -n 1 "${job}")

    local python_cmd=""
    if [[ "${shebang}" == "#!/usr/bin/env "* ]]; then
        python_cmd=${shebang#\#!/usr/bin/env }
    elif [[ "${shebang}" == "#!"* ]]; then
        python_cmd=${shebang#\#!}
    else
        python_cmd=python3
    fi

    local resolved_python=""
    if [[ "${python_cmd}" == /* ]]; then
        resolved_python=${python_cmd}
    else
        resolved_python=$(command -v "${python_cmd}")
    fi

    if [ ! -x "${resolved_python}" ]; then
        echo "FAIL: Python interpreter not found for ${job}: ${python_cmd}"
        return 1
    fi

    echo "${resolved_python}"
}

collect_python_test_dependencies() {
    local deps=""
    local job=""
    local dep_line=""
    for d in ${TESTS}; do
        if [ ! -x "${PWD}/${d}" ]; then
            continue
        fi
        for job in $(find "${d}" -name "[0-9]*.py" -type f | grep -v "~" | sort); do
            if [ ! -x "${job}" ]; then
                continue
            fi
            dep_line=$(grep -E '^# GUNDAM_TEST_PYTHON_DEPENDENCIES:' "${job}" || true)
            if [ -n "${dep_line}" ]; then
                deps="${deps} ${dep_line#\# GUNDAM_TEST_PYTHON_DEPENDENCIES: }"
            fi
        done
    done
    echo "${deps}" | xargs -n 1 | sort -u | xargs
}

prepare_python_test_venv() {
    local job=$1
    local log=$2

    if [ "${PYTHON_TEST_VENV_READY}" = "yes" ]; then
        local resolved_python=""
        resolved_python=$(resolve_python_test_interpreter "${job}") || {
            echo "$resolved_python" > "${PWD}/${OUTPUT_DIR}/${log}"
            return 1
        }
        if [ "${resolved_python}" != "${PYTHON_TEST_VENV_PYTHON}" ]; then
            echo "FAIL: Python interpreter mismatch for ${job}: ${resolved_python} != ${PYTHON_TEST_VENV_PYTHON}" > "${PWD}/${OUTPUT_DIR}/${log}"
            return 1
        fi
        return 0
    fi

    PYTHON_TEST_VENV_PYTHON=$(resolve_python_test_interpreter "${job}") || {
        echo "${PYTHON_TEST_VENV_PYTHON}" > "${PWD}/${OUTPUT_DIR}/${log}"
        return 1
    }

    "${PYTHON_TEST_VENV_PYTHON}" -m venv "${PYTHON_TEST_VENV}" > "${PWD}/${OUTPUT_DIR}/${log}" 2>&1 || return 1

    local deps=""
    deps=$(collect_python_test_dependencies)
    if [ -n "${deps}" ]; then
        "${PYTHON_TEST_VENV}/bin/python" -m pip install ${deps} >> "${PWD}/${OUTPUT_DIR}/${log}" 2>&1 || return 1
    fi

    PYTHON_TEST_VENV_READY="yes"
    return 0
}

run_python_test() {
    local job=$1
    local dir=$2
    local log=$3

    prepare_python_test_venv "${job}" "${log}" || return 1
    if [ "${VERBOSE_LOGS}" = "yes" ]; then
        (
            cd "${OUTPUT_DIR}" &&
            "${PYTHON_TEST_VENV}/bin/python" "${job}" "${dir}" 2>&1 | tee -a "${log}"
            exit ${PIPESTATUS[0]}
        )
    else
        (cd "${OUTPUT_DIR}" && "${PYTHON_TEST_VENV}/bin/python" "${job}" "${dir}" >> "${log}" 2>&1)
    fi
}

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
        if [[ "${JOB}" == *.py ]]; then
            echo "python-venv:${JOB} ${DIR}"
            : > "${OUTPUT_DIR}/${LOG}"
            run_python_test "${JOB}" "${DIR}" "${LOG}"
            JOB_STATUS=$?
        else
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
