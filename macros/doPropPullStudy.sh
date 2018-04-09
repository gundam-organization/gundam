#!/bin/sh
source /home/dolan/.bashrc
runnd280
sourcefitter
cd $1
mkdir $2
#fixedci=${9: -0}
#firstfixed=${{10}: -99}
#binning=${{11}: -$XSLLHFITER/inputs/dptbinning2DPS_shortest_inclusive.txt}
PropError -o ${2}/propErrorOut_${3}.root -s ${RANDOM} -t 500 -f ${4} -p ${5} -r ${6} -n ${7} -N ${8} -F ${9} -G ${10} -b ${11}

# Use as:
#  i=0; for file in $( ls studyOut_*.root ); do qsub doPropPullStdy.sh -F "$PWD propErrorOutDir $i $file 1 irreleventFile.root  331.6 331.6 1 8 $XSLLHFITER/inputs/dptbinning2DPS_shortest_inclusive.txt"; let i=i+1; done - for syst
#  i=0; for file in $( ls studyOut_*.root ); do qsub doPropPullStdy.sh -F "$PWD propErrorOutDir $i $file 2 irreleventFile.root  331.6 372.67 1 8 $XSLLHFITER/inputs/dptbinning2DPS_shortest_inclusive.txt"; let i=i+1; done - for stats only asimov
#  i=0; for file in $( ls studyOut_*.root ); do qsub doPropPullStdy.sh -F "$PWD propErrorOutDir $i $file 3 nomFile.root 331.6 372.67 1 8 $XSLLHFITER/inputs/dptbinning2DPS_shortest_inclusive.txt"; let i=i+1; done - for stats only fake data
# N.B irreleventFile.root has to be a real file

#i=0; for file in $( ls *AsiOut_10*.root ); do qsub doPropPullStudy.sh -F "$PWD propErrorOutDir_short $i $file 3 $PWD/../asiNomOut_noCCQEpar_oneOOPSbin_dataStats.root 331.6 331.6 1 9 /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dptbinning2DPS_shortest_inclusive.txt"; let i=i+1; done