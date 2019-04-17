#!/bin/bash

TRUTH_TREE=""
READ_LIMITS=""
NORM_DIALS=""

while getopts ":hTLNd:i:n:b:o:" opt; do
    case ${opt} in
        h )
            echo "USAGE: "
            echo "-h: Display this help message."
            echo "-i: Input list of Highland files (.txt)"
            echo "-d: Input dials file (.txt)"
            echo "-b: Input binning file (.txt)"
            echo "-o: Output validation file"
            echo "-n: Number of dials steps"
            echo "-T: Use truth tree"
            echo "-L: Read errors as upper/lower limits"
            echo "-N: Normalize dial values for splines"
            exit 0
            ;;
        i )
            HIGHLAND_INPUT=${OPTARG}
            ;;
        d )
            FILE_DIALS=${OPTARG}
            ;;
        b )
            FILE_BINNING=${OPTARG}
            ;;
        o )
            FILE_VALIDATE=${OPTARG}
            ;;
        n )
            NUM_STEPS=${OPTARG}
            ;;
        T )
            TRUTH_TREE="-T"
            ;;
        L )
            READ_LIMITS="-L"
            ;;
        N )
            NORM_DIALS="-N"
            ;;
        \? )
            echo -e "\033[31mInvalid argument: $OPTARG\033[0m"
            exit 1
            ;;
    esac
done

echo -e "\033[96mMaking splines...\033[0m"

WEIGHTS_SUFFIX="weights.root"
SPLINES_SUFFIX="splines.root"

rm -v ${FILE_VALIDATE}

for dial in $(sed '/^#/d' ${FILE_DIALS} | awk '{print $1}')
do
    echo -e "\033[93mGenerating splines for ${dial}\033[0m"
    FILE_WEIGHTS=$(echo "${dial}_${WEIGHTS_SUFFIX}" | awk '{print tolower($0)}')
    FILE_SPLINES=$(echo "${dial}_${SPLINES_SUFFIX}" | awk '{print tolower($0)}')
    ./xsllh_response_functions.exe -i ${HIGHLAND_INPUT} -o ${FILE_WEIGHTS} -d ${FILE_DIALS} -r ${dial} -n ${NUM_STEPS} ${READ_LIMITS} ${TRUTH_TREE}
    ./xsllh_generate_splines.exe -i ${FILE_WEIGHTS} -o ${FILE_SPLINES} -b ${FILE_BINNING} -d ${FILE_DIALS} -r ${dial} -n ${NUM_STEPS} ${READ_LIMITS} ${NORM_DIALS}
    ./xsllh_validate_splines.exe -i ${FILE_WEIGHTS} -o ${FILE_VALIDATE} -b ${FILE_BINNING} -s ${FILE_SPLINES} -d ${FILE_DIALS} -r ${dial} -n ${NUM_STEPS} -U ${READ_LIMITS} ${NORM_DIALS}
done
echo -e "\033[96mFinished.\033[0m"
