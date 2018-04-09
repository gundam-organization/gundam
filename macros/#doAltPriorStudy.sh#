#!/bin/sh
source /home/dolan/.bashrc
runnd280
sourcefitter
CCQEFit -i $1 -f $1 -m 10 -o $2 -s ${RANDOM} -M 2 -S 1 
# Use as:
# # e.g for i in `seq 0 250`; do qsub doAltPriorStudy.sh -F "/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutAir5_2DV2.root  $PWD/biasStudyAsiOut_${i}_reg0p001.root"; done;