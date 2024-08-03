#!/bin/sh

#Example of how to run:
# for i in `seq 0 4`; do qsub -l cput=00:50:50 getIndiSystEr.sh -F "dphit $i $PWD"; done;


source /home/dolan/.bashrc
runnd280
sourcefitter

var=$1
systConf=$2
useCR=$3
whereami=$4

varToFit=${var}

cd ${whereami}

echo "Running in test mode:"
echo "var is"
echo $1
echo "syst configuraton is:"
echo $2
echo "Using sidebands?:"
echo $3
echo "working dir is:"
echo $4

echo "This program allows the easy running of the fitter only including certain systmatic parameters"

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
elif [ "${var}" = "dptFeb17" ]
then
  varToFit="dpt"
  echo "Running feb17 refits"
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/NeutAirV2_dpt_allStats.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/NeutWater_dpt_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/Neut6DV2_POTFix_dpt_no2p2h_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/GenieAirV2_dpt_dataStats.root"
  nuWroFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/NuWro_dpt_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/p6K_rdp_allRuns_FHC_v1.root"
  neutBaskFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/Neut6DV2_POTFix_dpt_dataStats.root"
elif [ "${var}" = "dphitFeb17" ]
then
  echo "Running feb17 refits"
  varToFit="dphit"
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/NeutAirV2_dphit_allStats.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/NeutWater_dphit_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/Neut6DV2_POTFix_dphit_no2p2h_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/GenieAirV2_dphit_dataStats.root"
  nuWroFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/NuWro_dphit_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/dphitInputs/p6K_rdp_allRuns_FHC_v1.root"
  neutBaskFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/Neut6DV2_POTFix_dphit_dataStats.root"
elif [ "${var}" = "datFeb17" ]
then
  echo "Running feb17 refits"
  varToFit="dat"
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/NeutAirV2_dat_allStats.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/NeutWater_dat_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/Neut6DV2_POTFix_dat_no2p2h_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/GenieAirV2_dat_dataStats.root"
  nuWroFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/NuWro_dat_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/datInputs/p6K_rdp_allRuns_FHC_v1.root"
  neutBaskFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/Neut6DV2_POTFix_dat_dataStats.root"
else
  echo "Unknown vairable"
  echo ${var}
  echo "exiting ..."
  exit
fi

#New POTs:
neutPOT="331.9" # Data/InputSim POT ratio is: 0.20
neutFDPOT="66.8151" #Ratio is 0.19, total POT is 350
genieFDPOT="66.8151" # Ratio is 0.18, total POT is 372.5
nuWroFDPOT="66.8151" # Ratio os 0.167 total POT is 40.1
rdpDataPOT="66.8151"
neutBaskFDPOT="63.03" #56.78
neutNo2p2hFDPOT="63.03"


#If statement not working for some reason
if [ "${var}" = "dptFeb17" ] || [ "${var}" = "dphitFeb17" ] || [ "${var}" = "datFeb17" ]
then
  echo "Getting feb17 refits POT"
  neutPOT="331.9" # Data/InputSim POT ratio is: 0.20
  neutFDPOT="66.8151" #Ratio is 0.19, total POT is 350
  genieFDPOT="66.8151" # Ratio is 0.18, total POT is 372.5
  nuWroFDPOT="66.8151" # Ratio os 0.167 total POT is 40.1
  rdpDataPOT="66.8151"
  neutBaskFDPOT="63.03" 
  neutNo2p2hFDPOT="63.03" 
fi


reg1=3
reg2=1

var=${varToFit}

if [ "${var}" == "test" ]
then
	echo "Running in test mode:"
	echo "var is"
	echo $1
	echo "syst configuraton is:"
	echo $2
  echo "working dir is:"
  echo $3
elif [ "${var}" == "dphit" ]
then
	echo "Running is dphit mode"
	CCQEFit_dphiT   -y ${systConf} -i ${neutFD} -f ${neutFD} -o "dphit_systmode${systConf}_fitOut.root" -s 1337 -r ${reg1} -R ${reg2} -m 2 -M 2 -N ${neutFDPOT} -n ${neutFDPOT}  -c ${useCR} &> ${var}_systmode${systConf}_fitoutput.log 
    PropError_dphiT -y ${systConf} -i ${neutFD} -d ${neutFD} -f "dphit_systmode${systConf}_fitOut.root" -s 1337 -N ${neutFDPOT} -n ${neutFDPOT} -o "dphit_systmode${systConf}_propOut.root" -t 5000  &-c ${useCR} > ${var}_systmode${systConf}_propoutput.log
	echo "Finished"
elif [ "${var}" == "dat" ]
then
	echo "Running is dat mode"
	CCQEFit_daT   -y ${systConf} -i ${neutFD} -f ${neutFD} -o "dat_systmode${systConf}_fitOut.root" -s 1337 -r ${reg1} -R ${reg2} -m 2 -M 2 -N ${neutFDPOT} -n ${neutFDPOT}  -c ${useCR} &> ${var}_systmode${systConf}_fitoutput.log 
    PropError_daT -y ${systConf} -i ${neutFD} -d ${neutFD} -f "dat_systmode${systConf}_fitOut.root" -s 1337 -N ${neutFDPOT} -n ${neutFDPOT} -o "dat_systmode${systConf}_propOut.root" -t 5000  -c ${useCR} &> ${var}_systmode${systConf}_propoutput.log
	echo "Finished"
else
	echo "Running in dpt mode"
	CCQEFit   -y ${systConf} -i ${neutFD} -f ${neutFD} -o "dpt_systmode${systConf}_fitOut.root" -s 1337 -r ${reg1} -R ${reg2} -m 2 -M 2 -N ${neutFDPOT} -n ${neutFDPOT}  -c ${useCR} &> ${var}_systmode${systConf}_fitoutput.log 
	PropError -y ${systConf} -i ${neutFD} -d ${neutFD} -f "dpt_systmode${systConf}_fitOut.root" -s 1337 -N ${neutFDPOT} -n ${neutFDPOT} -o "dpt_systmode${systConf}_propOut.root" -t 5000 -F 1 -G 8  -c ${useCR} &> ${var}_systmode${systConf}_propoutput.log
	echo "Finished"
fi

if [ "${var}" == "test" ]
then
  echo "Running in test mode:"
  echo "var is"
  echo $1
  echo "syst configuraton is:"
  echo $2
  echo "working dir is:"
  echo $3
elif [ "${var}" == "dphit" ]
then
  echo "Running is dphit mode"
  CCQEFit_dphiT   -y ${systConf} -i ${neutMC} -f ${neutMC} -o "dphit_systmode${systConf}_fitOut_allStats.root" -s 1337 -r ${reg1} -R ${reg2} -m 2 -M 2 -N ${neutPOT} -n ${neutPOT}  -c ${useCR} &> ${var}_systmode${systConf}_fitoutput.log 
    PropError_dphiT -y ${systConf} -i ${neutMC} -d ${neutMC} -f "dphit_systmode${systConf}_fitOut_allStats.root" -s 1337 -N ${neutPOT} -n ${neutPOT} -o "dphit_systmode${systConf}_propOut_allStats.root" -t 5000  &-c ${useCR} > ${var}_systmode${systConf}_propoutput.log
  echo "Finished"
elif [ "${var}" == "dat" ]
then
  echo "Running is dat mode"
  CCQEFit_daT   -y ${systConf} -i ${neutMC} -f ${neutMC} -o "dat_systmode${systConf}_fitOut_allStats.root" -s 1337 -r ${reg1} -R ${reg2} -m 2 -M 2 -N ${neutPOT} -n ${neutPOT}  -c ${useCR} &> ${var}_systmode${systConf}_fitoutput.log 
    PropError_daT -y ${systConf} -i ${neutMC} -d ${neutMC} -f "dat_systmode${systConf}_fitOut_allStats.root" -s 1337 -N ${neutPOT} -n ${neutPOT} -o "dat_systmode${systConf}_propOut_allStats.root" -t 5000  -c ${useCR} &> ${var}_systmode${systConf}_propoutput.log
  echo "Finished"
else
  echo "Running in dpt mode"
  CCQEFit   -y ${systConf} -i ${neutMC} -f ${neutMC} -o "dpt_systmode${systConf}_fitOut_allStats.root" -s 1337 -r ${reg1} -R ${reg2} -m 2 -M 2 -N ${neutPOT} -n ${neutPOT}  -c ${useCR} &> ${var}_systmode${systConf}_fitoutput.log 
  PropError -y ${systConf} -i ${neutMC} -d ${neutMC} -f "dpt_systmode${systConf}_fitOut_allStats.root" -s 1337 -N ${neutPOT} -n ${neutPOT} -o "dpt_systmode${systConf}_propOut_allStats.root" -t 5000 -F 1 -G 8  -c ${useCR} &> ${var}_systmode${systConf}_propoutput.log
  echo "Finished"
fi