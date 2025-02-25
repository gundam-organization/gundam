#!/bin/bash

# Set the base name for this test (should match the script name)
BASE=200TabulatedAsimov

# Get the directory containing the script from the command line
# parameters (avoids bash trickery).  Use the current directory as the
# default.
DIR=.
if [ ${#1} -gt 0 ]; then
    DIR=${1}
fi

# Make sure that gundam has been setup.
if ! which gundamFitter; then
    echo FAIL: Executable not found for gundamFitter
    exit 1
fi

# Set the expected locations for the config and output files.
export CONFIG_DIR=${DIR}
export DATA_DIR=${PWD}

CONFIG_FILE=${CONFIG_DIR}/${BASE}-config.yaml
OUTPUT_FILE=${DATA_DIR}/${BASE}.root
SHARED_LIBRARY=${DATA_DIR}/${BASE}.so

echo ${OUTPUT_FILE}
echo ${CONFIG_FILE}

if [ -f ${SHARED_LIBRARY} ]; then
    rm ${SHARED_LIBRARY}
fi
if ! $(root-config --cxx) $(root-config --cflags) \
                     -fPIC --shared -rdynamic \
                     -o ${SHARED_LIBRARY} \
                     ${CONFIG_DIR}/${BASE}.C; then
   echo Error building shared library
   exit 1
fi

gundamFitter --cpu -t 1 -s 10000 -c ${CONFIG_FILE} -o ${OUTPUT_FILE}

# End of the script
