#!/bin/sh
time
#run example:
#for i in `seq 0 250`; do qsub -l cput=01:50:00  ./doBiasStudy.sh -F "dat $i $PWD 7.0"; done

source /home/dolan/.bashrc
runnd280
sourcefitter
var=$1
iterCount=$2
workdir=$3
reg1=$4
reg2=1.0

echo "Startting doBiasStudy.sh"
echo "Variable is"
echo ${var}

if [ "${var}" = "dpt" ]
then
  neutMC_all="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/NeutAirV2_dpt_allStats.root"
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/NeutAirV2_dpt_dataStats.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/NeutWater_dpt_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/Neut6DV2_POTFix_dpt_no2p2h_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/GenieAirV2_dpt_dataStats.root"
  nuWroFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/NuWro_dpt_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/p6K_rdp_allRuns_FHC_v1.root"
  neutBaskFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/Neut6DV2_POTFix_dpt_dataStats.root"
elif [ "${var}" = "dphit" ]
then
  neutMC_all="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/NeutAirV2_dphit_allStats.root"
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/NeutAirV2_dphit_dataStats.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/NeutWater_dphit_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/Neut6DV2_POTFix_dphit_no2p2h_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/GenieAirV2_dphit_dataStats.root"
  nuWroFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/NuWro_dphit_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dphitInputs/p6K_rdp_allRuns_FHC_v1.root"
  neutBaskFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/Neut6DV2_POTFix_dphit_dataStats.root"
elif [ "${var}" = "dat" ]
then
  neutMC_all="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/NeutAirV2_dat_allStats.root"
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/NeutAirV2_dat_dataStats.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/NeutWater_dat_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/Neut6DV2_POTFix_dat_no2p2h_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/GenieAirV2_dat_dataStats.root"
  nuWroFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/NuWro_dat_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/datInputs/p6K_rdp_allRuns_FHC_v1.root"
  neutBaskFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/Neut6DV2_POTFix_dat_dataStats.root"
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

#New POTs:
neutPOT_all="331.9" # Data/InputSim POT ratio is: 0.20
neutPOT="66.8151" # Data/InputSim POT ratio is: 0.20
neutFDPOT="66.8151" #Ratio is 0.19, total POT is 350
#neutNo2p2hFDPOT="66.8151" #Ratio is 0.19, total POT is 350
genieFDPOT="66.8151" # Ratio is 0.18, total POT is 372.5
nuWroFDPOT="66.8151" # Ratio os 0.167 total POT is 40.1
rdpDataPOT="66.8151"
#neutBaskFDPOT="72.828"
neutBaskFDPOT="63.03" #56.78
#neutBaskFDPOT="56.78"
neutNo2p2hFDPOT="63.03"

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
  CCQEFit -i ${neutMC} -o "systStudyFitOut_${iterCount}.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -n ${neutPOT} -M 2 -S 1 -m 5
  mkdir propErrorDir
  PropError -o propErrorDir/"systStudyPropErrOut_${iterCount}.root" -s ${RANDOM} -t 5000 -f "systStudyFitOut_${iterCount}.root" -p 1 -n ${neutPOT} -F 1 -G 8
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
  CCQEFit_dphiT -i ${neutMC} -o "systStudyFitOut_${iterCount}.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -n ${neutPOT} -M 2 -S 1 -m 5
  mkdir propErrorDir
  PropError_dphiT -o propErrorDir/"systStudyPropErrOut_${iterCount}.root" -s ${RANDOM} -t 5000 -f "systStudyFitOut_${iterCount}.root" -p 1 -n ${neutPOT}  
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
  CCQEFit_dphiT -i ${neutMC} -o "systStudyFitOut_${iterCount}.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -n ${neutPOT} -M 2 -S 1 -m 5
  mkdir propErrorDir
  PropError_daT -o propErrorDir/"systStudyPropErrOut_${iterCount}.root" -s ${RANDOM} -t 5000 -f "systStudyFitOut_${iterCount}.root" -p 1 -n ${neutPOT}  
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
  CCQEFit -i ${neutMC} -o "systStudyFitOut_${iterCount}.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -n ${neutPOT} -M 2 -S 1 -m 5 -c 0 
  mkdir propErrorDir
  PropError -o propErrorDir/"systStudyPropErrOut_${iterCount}.root" -s ${RANDOM} -t 5000 -f "systStudyFitOut_${iterCount}.root" -p 1 -n ${neutPOT} -c 0 -z 1.95996 -Z 2.5758
  cd propErrorDir
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/propPullStudy.C .
else
	echo "unrecognised variable"
fi

echo "finished"
time


# Use as:
# # e.g for i in `seq 0 250`; do qsub doAsimovBiasStudy.sh -F "/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutAir5_2DV2.root  $PWD/biasStudyAsiOut_${i}_reg0p001.root"; done;