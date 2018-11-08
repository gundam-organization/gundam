#!/bin/bash
FITROOT="/mnt/home/cuddandr/work/nd280Software/v12r15/xsLLhFitter"
RUNPATH="${SCRATCH}/xsllhFit"

if [ ! -d $RUNPATH ]; then
    echo "Creating directory: $RUNPATH"
    mkdir $RUNPATH
else
    rm $RUNPATH/*
fi

if [ -z ${1} ]; then
    echo "No config file specified. Exiting."
    exit 64
fi

THREADS=$(grep -i "num_threads" ${1} | sed 's/[^0-9]*//g')
CONFIG=$(basename ${1})

sed -i "/#SBATCH --cpus-per-task/c\#SBATCH --cpus-per-task=${THREADS}" fit_job.sb
sed -i "/xsllhFit -j/c\xsllhFit -j ${CONFIG} &> fit_output.log" fit_job.sb

echo "Config File: ${1}"
echo "Threads: ${THREADS}"

cp ${1} $RUNPATH
cd $RUNPATH

sbatch ${FITROOT}/scripts/fit_job.sb
