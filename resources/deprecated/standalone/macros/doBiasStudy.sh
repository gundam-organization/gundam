#!/bin/sh
time
#run example:
#for i in `seq 0 250`; do qsub -l cput=01:50:00  ./doBiasStudy.sh -F "dat $i $PWD /data/t2k/dolan/fitting/Nov16_runByScript/dat/rdp/7_1reg/quickFtX_fitOut.root 331.6 7.0"; done

source /home/dolan/.bashrc
runnd280
sourcefitter
var=$1
iterCount=$2
workdir=$3
prevFitResult=$4
prevFitResultPOT=$5
reg1=$6
reg2=1.0

echo "Startting doBiasStudy.sh"
echo "Variable is"
echo ${var}

if [ "${var}" = "dpt" ]
then
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutAir5_2DV2.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutWater2_2D_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutWater2_2D_No2p2h_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/GenieAir2_2DV2_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/p6K_rdp_allRuns_FHC_v1.root"
elif [ "${var}" = "dphit" ]
then
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dphitInputs/NeutAir5_2D.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dphitInputs/NeutWater5_2D_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dphitInputs/NeutWater5_2D_2p2hless_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dphitInputs/GenieAir2_2D_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dphitInputs/p6K_rdp_allRuns_FHC_v1.root"
elif [ "${var}" = "dat" ]
then
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/datInputs/NeutAir5_2D.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/datInputs/NeutWater2_2D_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/datInputs/NeutWater2_2D_2p2hless_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/datInputs/GenieAir2_2D_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/datInputs/p6K_rdp_allRuns_FHC_v1.root"
elif [ "${var}" = "muOnly" ]
then
  sourcefitter_muOnlyChi2
  neutMC="/data/t2k/dolan/software/xsLLhFitter/muOnlyChi2/inputs/NeutAirV5.root"
else
  echo "Unknown vairable"
  echo ${var}
  echo "Or no command line options specified."
  echo "Please run as: "
  echo "./doBiasStudy.sh variable iterationCount WorkingDir reffFitResult reffFitResultPOT reg1Strength"
  echo "exiting ..."
  exit
fi

neutPOT="331.6"
neutFDPOT="58.11"
neutNo2p2hFDPOT="58.11"
genieFDPOT="58.11"
rdpDataPOT="66.8151"

echo "inputs: "
echo ${neutMC}
echo ${neutFD}
echo ${neutNo2p2hFD}
echo ${genieFD}
echo ${rdpData}

echo "var is:"
echo ${var}
echo "iterCount is:"
echo ${iterCount}
echo "workdir is:"
echo ${workdir}
echo "prevFitResult is:"
echo ${prevFitResult}
echo "prevFitResult POT is:"
echo ${prevFitResultPOT}
echo "reg1 is:"
echo ${reg1}
echo "reg2 is:"
echo ${reg2}


cd ${workdir}
pwd

if [ "${var}" == "dpt" ]
then
  echo "Running in dpt mode"
  mkdir dpt
  cd dpt
  CCQEFit -i ${neutMC} -v ${prevFitResult} -o "biasStudyFitOut_${iterCount}.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -n ${neutPOT} -M 2 -S 1 -m 9
  mkdir propErrorDir
  PropError -o propErrorDir/"biasStudyPropErrOut_${iterCount}.root" -s ${RANDOM} -t 5000 -f "biasStudyFitOut_${iterCount}.root" -p 3 -r ${prevFitResult} -n ${neutPOT} -N ${prevFitResultPOT} -F 1 -G 8
  cd propErrorDir
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/propPullStudy.C .
  sleep $((RANDOM%10))
  if [ `qstat -u dolan | grep doBiasStudy | wc -l` == '1' ]
  then
    root -b -l -q propPullStudy.C
  fi
elif [ "${var}" == "dphit" ]
then
  echo "Running in dphit mode"
  mkdir dphit
  cd dphit
  CCQEFit_dphiT -i ${neutMC} -v ${prevFitResult} -o "biasStudyFitOut_${iterCount}.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -n ${neutPOT} -M 2 -S 1 -m 9
  mkdir propErrorDir
  PropError_dphiT -o propErrorDir/"biasStudyPropErrOut_${iterCount}.root" -s ${RANDOM} -t 5000 -f "biasStudyFitOut_${iterCount}.root" -p 3 -r ${prevFitResult} -n ${neutPOT} -N ${prevFitResultPOT} 
  cd propErrorDir
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/propPullStudy.C .
  sleep $((RANDOM%10))
  if [ `qstat -u dolan | grep doBiasStudy | wc -l` == '1' ]
  then
    root -b -l -q propPullStudy.C
  fi
elif [ "${var}" == "dat" ]
then
  echo "Running in dalphat mode"
  mkdir dat
  cd dat
  CCQEFit_dphiT -i ${neutMC} -v ${prevFitResult} -o "biasStudyFitOut_${iterCount}.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -n ${neutPOT} -M 2 -S 1 -m 9
  mkdir propErrorDir
  PropError_daT -o propErrorDir/"biasStudyPropErrOut_${iterCount}.root" -s ${RANDOM} -t 5000 -f "biasStudyFitOut_${iterCount}.root" -p 3 -r ${prevFitResult} -n ${neutPOT} -N ${prevFitResultPOT} 
  cd propErrorDir
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/propPullStudy.C .
  sleep $((RANDOM%10))
  if [ `qstat -u dolan | grep doBiasStudy | wc -l` == '1' ]
  then
    root -b -l -q propPullStudy.C
  fi
elif [ "${var}" == "muOnly" ]
then
  echo "Running in muOnly mode"
  mkdir muOnly
  cd muOnly
  CCQEFit -i ${neutMC} -v ${prevFitResult} -o "biasStudyFitOut_${iterCount}.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -n ${neutPOT} -M 2 -S 1 -m 9 -c 0 
  mkdir propErrorDir
  PropError -o propErrorDir/"biasStudyPropErrOut_${iterCount}.root" -s ${RANDOM} -t 5000 -f "biasStudyFitOut_${iterCount}.root" -p 3 -r ${prevFitResult} -n ${neutPOT} -N ${prevFitResultPOT} -c 0 -z 1.95996 -Z 2.5758
  cd propErrorDir
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/propPullStudy.C .
else
	echo "unrecognised variable"
fi

echo "finished"
time


# Use as:
# # e.g for i in `seq 0 250`; do qsub doAsimovBiasStudy.sh -F "/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutAir5_2DV2.root  $PWD/biasStudyAsiOut_${i}_reg0p001.root"; done;