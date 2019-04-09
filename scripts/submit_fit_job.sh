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

if [ -z ${2} ]; then
    NUMJOBS=0
else
    NUMJOBS=$((${2}-1))
fi

THREADS=$(grep -i "num_threads" ${1} | sed 's/[^0-9]*//g')
CONFIG=$(basename ${1})

sed -i "/#SBATCH --cpus-per-task/c\#SBATCH --cpus-per-task=${THREADS}" ${FITROOT}/scripts/fit_job.sb
sed -i "/#SBATCH --array/c\#SBATCH --array=0-${NUMJOBS}" ${FITROOT}/scripts/fit_job.sb
sed -i "/FITCONFIG=/c\FITCONFIG=\"${CONFIG}\"" ${FITROOT}/scripts/fit_job.sb

echo "Config File: ${1}"
echo "Threads: ${THREADS}"
echo "Num Jobs: $(($NUMJOBS+1))"

cp ${1} $RUNPATH
cd $RUNPATH

sbatch ${FITROOT}/scripts/fit_job.sb
