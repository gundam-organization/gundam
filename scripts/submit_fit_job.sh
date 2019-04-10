#!/bin/bash
FITROOT="/mnt/home/cuddandr/work/nd280Software/v12r15/xsLLhFitter"
RUNPATH="${SCRATCH}/xsllhFit"

if [ ! -d $RUNPATH ]; then
    echo "Creating directory: $RUNPATH"
    mkdir $RUNPATH
#else
#    rm $RUNPATH/*
fi

if [ -z ${1} ]; then
    echo "No fit config file specified. Exiting."
    exit 64
fi

if [ -z ${2} ]; then
    echo "No errprop config file specified. Exiting."
    exit 64
fi

if [ -z ${3} ]; then
    NUMJOBS=0
else
    NUMJOBS=$((${3}-1))
fi

if [ -z ${4} ]; then
    SEEDOFFSET=$RANDOM
else
    SEEDOFFSET=${4}
fi

THREADS=$(grep -i "num_threads" ${1} | sed 's/[^0-9]*//g')
FITCONFIG=$(basename ${1})
ERRCONFIG=$(basename ${2})

sed -i "/#SBATCH --cpus-per-task/c\#SBATCH --cpus-per-task=${THREADS}" ${FITROOT}/scripts/fit_job.sb
sed -i "/#SBATCH --array/c\#SBATCH --array=0-${NUMJOBS}" ${FITROOT}/scripts/fit_job.sb
sed -i "/FITCONFIG=/c\FITCONFIG=\"${FITCONFIG}\"" ${FITROOT}/scripts/fit_job.sb
sed -i "/ERRCONFIG=/c\ERRCONFIG=\"${ERRCONFIG}\"" ${FITROOT}/scripts/fit_job.sb
sed -i "/OFFSET=/c\OFFSET=${SEEDOFFSET}" ${FITROOT}/scripts/fit_job.sb

echo "Fit Config File: ${1}"
echo "Err Config File: ${2}"
echo "Threads: ${THREADS}"
echo "Num Jobs: $(($NUMJOBS+1))"
echo "Seed offset: ${SEEDOFFSET}"

cp ${1} $RUNPATH
cp ${2} $RUNPATH
cd $RUNPATH

sbatch ${FITROOT}/scripts/fit_job.sb
