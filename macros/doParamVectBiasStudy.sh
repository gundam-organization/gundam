#!/bin/sh
source /home/dolan/.bashrc
runnd280
sourcefitter
CCQEFit -i $1 -v $2 -o $3 -s ${RANDOM} -r $4 -R $5 -M 2 -S 1 -m 9
# Use as:
# # e.g for i in `seq 0 250`; do qsub doParamVectBiasStudy.sh -F "/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutAir5_2DV2.root $PWD/../lcurveStudyOut_reg0.8.root $PWD/biasStudyOut_${i}_r1R1.root 1.0 1.0"; done;