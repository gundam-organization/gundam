#!/bin/sh
source /home/dolan/.bashrc
runnd280
sourcefitter
if [ "${8}" == "dat" ]
then
	echo "Running is dat mode"
	CCQEFit_daT -i $1 -f $2 -o $3 -s ${RANDOM} -r $4 -R $5 -m 2 -M 2 -n $6 -N $7
elif [ "${8}" == "dphit" ]
then
	echo "Running is dphit mode"
	CCQEFit_dphiT -i $1 -f $2 -o $3 -s ${RANDOM} -r $4 -R $5 -m 2 -M 2 -n $6 -N $7
else
	echo "Running is dpt mode"
    CCQEFit -i $1 -f $2 -o $3 -s ${RANDOM} -r $4 -R $5 -m 2 -M 2 -n $6 -N $7
fi
# Use as:
# for i in 0.01 0.1 1 3 5 7 9 15 20 30 50; do qsub doLcurveStudy.sh -F "inputMC fakeData outputfile_reg${i}.root $i"; done;
# e.g for i in 0.01 0.1 1 3 5 7 9 15 20 30 50; do qsub doLcurveStudy.sh -F "/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutAir5_2DV2.root /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/GenieAir2_2DV2.root $PWD/lcurveStudyOut_reg${i}.root $i /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dptbinning2DPS.txt"; done;
# e.g:
# for r in 0.01 0.05 0.1 1 3 6 15; 
#   do for R in 0.01 0.05 0.1 1 3 6 15;
#     do qsub doLcurveStudy.sh -F "/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutAir5_2DV2.root /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/GenieAir2_2DV2.root 
#                                  $PWD/lcurveStudyOut_regr${r}r${R}R.root $r $R 58.11 331.6 /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dptbinning2DPS_shorter_inclusive.txt dpt"; done; done;