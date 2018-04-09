#!/bin/sh
time
source /home/dolan/.bashrc
runnd280
sourcefitter
var=$1
runmode=$2
workdir=$3

varToFit=${var}


echo "Startting runQuickFits.sh"
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
elif [ "${var}" = "dptFeb17" ]
then
  varToFit="dpt"
  echo "Running feb17 refits"
  neutMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/NeutAirV2_dpt_allStats.root"
  neutFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/NeutWater_dpt_dataStats.root"
  neutNo2p2hFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/Neut6DV2_POTFix_dpt_no2p2h_dataStats.root"
  genieFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/GenieAirV2_dpt_dataStats.root"
  genieMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dpt/GenieAirV2_dpt_allStats.root"
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
  genieMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dphit/GenieAirV2_dphit_allStats.root"
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
  genieMC="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/GenieAirV2_dat_allStats.root"
  nuWroFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/NuWro_dat_dataStats.root"
  rdpData="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/datInputs/p6K_rdp_allRuns_FHC_v1.root"
  neutBaskFD="/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/newFitsFeb17/dat/Neut6DV2_POTFix_dat_dataStats.root"
else
  echo "Unknown vairable"
  echo ${var}
  echo "exiting ..."
  exit
fi

#Old POTs (pre Feb17)
#neutPOT="331.6"
#neutFDPOT="58.11"
#neutNo2p2hFDPOT="58.11"
#genieFDPOT="58.11"
#rdpDataPOT="66.8151"
#nuWroFDPOT="66.8151" # Ratio os 0.167 total POT is 40.1
#neutBaskFDPOT="66.8151"
#
#
#asimovPOT_ratio="1.0"
#neutPOT_ratio="0.20"
#neutFDPOT_ratio="0.19"
#neutNo2p2hFDPOT_ratio="0.19"
#genieFDPOT_ratio="0.19"
#nuWroFDPOT_ratio="0.19"
#rdpDataPOT_ratio="0.19"
#neutBaskFDPOT_ratio="0.19"

#New POTs:
neutPOT="331.9" # Data/InputSim POT ratio is: 0.20
neutFDPOT="66.8151" #Ratio is 0.19, total POT is 350
#neutNo2p2hFDPOT="66.8151" #Ratio is 0.19, total POT is 350
genieFDPOT="66.8151" # Ratio is 0.18, total POT is 372.5
nuWroFDPOT="66.8151" # Ratio os 0.167 total POT is 40.1
rdpDataPOT="66.8151"
#neutBaskFDPOT="72.828"
neutBaskFDPOT="63.03" #56.78
#neutBaskFDPOT="56.78"
neutNo2p2hFDPOT="63.03"
asimovPOT_ratio="1.0"



neutPOT_ratio="0.20" # Data/InputSim POT ratio is: 0.20
neutFDPOT_ratio="0.19" #Ratio is 0.19, total POT is 350
neutNo2p2hFDPOT_ratio="0.19" #Ratio is 0.19, total POT is 350
genieFDPOT_ratio="0.19" # Ratio is 0.18, total POT is 372.5
nuWroFDPOT_ratio="0.19" # Ratio os 0.167 total POT is 40.1
rdpDataPOT_ratio="0.19"
neutBaskFDPOT_ratio="0.19"


#If statement not working for some reason
if [ "${var}" = "dptFeb17" ] || [ "${var}" = "dphitFeb17" ] || [ "${var}" = "datFeb17" ]
then
  echo "Getting feb17 refits POT"
  neutPOT="331.9" # Data/InputSim POT ratio is: 0.20
  neutFDPOT="66.8151" #Ratio is 0.19, total POT is 350
  #neutNo2p2hFDPOT="66.8151" #Ratio is 0.19, total POT is 350
  genieFDPOT="66.8151" # Ratio is 0.18, total POT is 372.5
  nuWroFDPOT="66.8151" # Ratio os 0.167 total POT is 40.1
  rdpDataPOT="66.8151"
  #neutBaskFDPOT="72.828"
  neutBaskFDPOT="63.03" 
  #neutBaskFDPOT="56.78"
  neutNo2p2hFDPOT="63.03" 

  neutPOT_ratio="0.20" # Data/InputSim POT ratio is: 0.20
  neutFDPOT_ratio="0.19" #Ratio is 0.19, total POT is 350
  neutNo2p2hFDPOT_ratio="0.19" #Ratio is 0.19, total POT is 350
  genieFDPOT_ratio="0.19" # Ratio is 0.18, total POT is 372.5
  nuWroFDPOT_ratio="0.19" # Ratio os 0.167 total POT is 40.1
  rdpDataPOT_ratio="0.19"
  neutBaskFDPOT_ratio="0.19"
fi

echo "inputs: "
echo ${neutMC}
echo ${neutFD}
echo ${neutNo2p2hFD}
echo ${genieFD}
echo ${rdpData}
echo ${nuWroFD}
echo ${neutBaskFD}

echo "var is:"
echo ${varToFit}
echo "runmode is:"
echo ${runmode}
echo "workdir is:"
echo ${workdir}

cd ${workdir}
pwd

COUNTER=1
RunMode="111"

#Run as: for i in `seq 1 25`; do qsub ./runQuickFits.sh -F "dpt $i $PWD"; done


#************************************************
#                      RDP L Curve:
#************************************************


if (( "${runmode}" == "rdplcurve" ))
then
  mkdir rdp
  cd rdp
  mkdir lCurve_fine
  cd lCurve_fine
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/doLcurveStudy.sh .
  echo "Running rdp fits to make high res l curve:"
  echo "qsub doLcurveStudy.sh -F ${neutMC} ${rdpData} $PWD/lcurveStudyOut_regr${r}r${R}R.root $r $R ${rdpDataPOT} ${neutPOT} ${varToFit}"
  for r in 0.01 0.05 0.1 1 2 3 4 5 8 10 12 15 20 30 50; do
  #for r in 1; do
    for R in 1; do
      qsub -l cput=01:59:59 doLcurveStudy.sh -F "${neutMC} ${rdpData} $PWD/lcurveStudyOut_regr${r}r${R}R.root $r $R ${rdpDataPOT} ${neutPOT} ${varToFit}";
    done; 
  done;
  cd ${workdir}
  time
  exit
fi


#************************************************
#                      ASIMOV:
#************************************************

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov
  cd asimov
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with no regularisation:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov
  cd asimov
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 1r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log 
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov
  cd asimov
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 3r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log 
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov
  cd asimov
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 5r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log 
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov
  cd asimov
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 7r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov
  cd asimov
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 9r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov_noSB
  cd asimov_noSB
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with no regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log 
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov_noSB
  cd asimov_noSB
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 1r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log 
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov_noSB
  cd asimov_noSB
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 3r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov_noSB
  cd asimov_noSB
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 5r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov_noSB
  cd asimov_noSB
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 7r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir asimov_noSB
  cd asimov_noSB
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running Asimov Fits with 9r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutFD} ${neutFD} ${neutFDPOT} ${neutFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutFD}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${asimovPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

#************************************************
#                      neutFD:
#************************************************

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD
  cd neutFD
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with no regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD
  cd neutFD
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 1r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD
  cd neutFD
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 3r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD
  cd neutFD
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 5r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD
  cd neutFD
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 7r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD
  cd neutFD
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 9r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD_noSB
  cd neutFD_noSB
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with no regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD_noSB
  cd neutFD_noSB
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 1r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD_noSB
  cd neutFD_noSB
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 3r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD_noSB
  cd neutFD_noSB
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 5r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD_noSB
  cd neutFD_noSB
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 7r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutFD_noSB
  cd neutFD_noSB
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutFD Fits with 9r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutFD} ${neutPOT} ${neutFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
    time
fi

let COUNTER=COUNTER+1
echo $COUNTER

#************************************************
#                      GENIE:
#************************************************


if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie
  cd genie
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with no regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie
  cd genie
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 1r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie
  cd genie
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 3r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie
  cd genie
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 5r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie
  cd genie
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 7r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie
  cd genie
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 9r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie_noSB
  cd genie_noSB
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with no regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie_noSB
  cd genie_noSB
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 1r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie_noSB
  cd genie_noSB
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 3r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie_noSB
  cd genie_noSB
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 5r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie_noSB
  cd genie_noSB
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 7r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir genie_noSB
  cd genie_noSB
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running genie Fits with 9r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${genieFD} ${neutPOT} ${genieFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${genieFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${genieFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER




#************************************************
#                      RDP:
#************************************************


if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp
  cd rdp
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with no regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "0.00001" "0.00001" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp
  cd rdp
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 1r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"1.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "1.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp
  cd rdp
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 3r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"3.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "3.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp
  cd rdp
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 5r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"5.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "5.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp
  cd rdp
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 7r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"7.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "7.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp
  cd rdp
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 9r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"9.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "9.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_noSB
  cd rdp_noSB
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with no regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "0.00001" "0.00001" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_noSB
  cd rdp_noSB
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 1r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"1.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "1.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_noSB
  cd rdp_noSB
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 3r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"3.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "3.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_noSB
  cd rdp_noSB
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 5r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"5.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "5.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_noSB
  cd rdp_noSB
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 7r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"7.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "7.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_noSB
  cd rdp_noSB
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 9r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"9.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "9.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER


#************************************************
#                      NEUT no 2p2h:
#************************************************
 


if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h
  cd neutNo2p2h
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with no regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h
  cd neutNo2p2h
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 1r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h
  cd neutNo2p2h
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 3r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h
  cd neutNo2p2h
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 5r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h
  cd neutNo2p2h
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 7r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h
  cd neutNo2p2h
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 9r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h_noSB
  cd neutNo2p2h_noSB
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with no regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h_noSB
  cd neutNo2p2h_noSB
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 1r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h_noSB
  cd neutNo2p2h_noSB
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 3r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h_noSB
  cd neutNo2p2h_noSB
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 5r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h_noSB
  cd neutNo2p2h_noSB
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 7r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutNo2p2h_noSB
  cd neutNo2p2h_noSB
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutNo2p2h fits with 9r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutNo2p2hFD} ${neutPOT} ${neutNo2p2hFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutNo2p2hFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutNo2p2hFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER


#************************************************
#                      NuWro:
#************************************************
 


if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro
  cd nuWro
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with no regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWro2p2hFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro
  cd nuWro
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 1r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro
  cd nuWro
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 3r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro
  cd nuWro
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 5r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro
  cd nuWro
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 7r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro
  cd nuWro
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 9r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro_noSB
  cd nuWro_noSB
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with no regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro_noSB
  cd nuWro_noSB
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 1r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro_noSB
  cd nuWro_noSB
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 3r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro_noSB
  cd nuWro_noSB
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 5r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro_noSB
  cd nuWro_noSB
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 7r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir nuWro_noSB
  cd nuWro_noSB
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running nuWro fits with 9r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${nuWroFD} ${neutPOT} ${nuWroFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${nuWroFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${nuWroFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER



#************************************************
#                      NEUT6D:
#************************************************
 


if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask
  cd neutBask
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with no regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask
  cd neutBask
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 1r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask
  cd neutBask
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 3r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask
  cd neutBask
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 5r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask
  cd neutBask
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 7r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask
  cd neutBask
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 9r1R regularisation:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"\" \"\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "" "" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask_noSB
  cd neutBask_noSB
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with no regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "0.00001" "0.00001" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask_noSB
  cd neutBask_noSB
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 1r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"1.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "1.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask_noSB
  cd neutBask_noSB
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 3r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"3.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "3.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask_noSB
  cd neutBask_noSB
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 5r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"5.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "5.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask_noSB
  cd neutBask_noSB
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 7r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"7.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "7.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir neutBask_noSB
  cd neutBask_noSB
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running neutBask fits with 9r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} \"9.0\" \"1.0\" false ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 1 &> qftxOut.log &"
  ./quickFitToXsec.sh ${neutMC} ${neutBaskFD} ${neutPOT} ${neutBaskFDPOT} "9.0" "1.0" "false" ${varToFit} ${RunMode} "-c 0" "-c 0" 1 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${neutMC}", "${neutBaskFD}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${neutBaskFD}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER




#************************************************
#                      RDP, GENIE PRIOR:
#************************************************


if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior
  cd rdp_geniePrior
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with no regularisation:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "0.00001" "0.00001" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior
  cd rdp_geniePrior
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 1r1R regularisation:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"1.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "1.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior
  cd rdp_geniePrior
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 3r1R regularisation:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"3.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "3.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior
  cd rdp_geniePrior
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 5r1R regularisation:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"5.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "5.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior
  cd rdp_geniePrior
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 7r1R regularisation:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"7.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "7.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior
  cd rdp_geniePrior
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 9r1R regularisation:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"9.0\" \"1.0\" true ${varToFit} ${RunMode} \"\" \"\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "9.0" "1.0" "true" ${varToFit} ${RunMode} "" "" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior_noSB
  cd rdp_geniePrior_noSB
  mkdir noreg
  cd noreg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with no regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "0.00001" "0.00001" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior_noSB
  cd rdp_geniePrior_noSB
  mkdir 1reg
  cd 1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 1r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"1.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "1.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior_noSB
  cd rdp_geniePrior_noSB
  mkdir 3_1reg
  cd 3_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 3r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"3.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "3.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior_noSB
  cd rdp_geniePrior_noSB
  mkdir 5_1reg
  cd 5_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 5r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"5.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "5.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior_noSB
  cd rdp_geniePrior_noSB
  mkdir 7_1reg
  cd 7_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 7r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"7.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "7.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER

if (( "${runmode}" == "${COUNTER}" ))
then
  mkdir rdp_geniePrior_noSB
  cd rdp_geniePrior_noSB
  mkdir 9_1reg
  cd 9_1reg
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/quickFitToXsec.sh .
  echo "Running rdp fits with 9r1R regularisation and no sideband:"
  echo "./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} \"9.0\" \"1.0\" true ${varToFit} ${RunMode} \"-c 0\" \"-c 0\" 0 &> qftxOut.log &"
  ./quickFitToXsec.sh ${genieMC} ${rdpData} ${neutPOT} ${rdpDataPOT} "9.0" "1.0" "true" ${varToFit} ${RunMode} "-c 0" "-c 0" 0 &> qftxOut.log
  mkdir plotReco
  cd plotReco
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/plotReco.cc .
root -b -l <<EOF
.L plotReco.cc
plotReco("${genieMC}", "${rdpData}", "plotRecoOut_prefit.root", 0, ${neutPOT_ratio}, "${varToFit}")
plotReco("../quickFtX_fitOut.root", "${rdpData}", "plotRecoOut_postfit.root", 0, 1.0, "${varToFit}")
.q
EOF
  cd ${workdir}
  time
fi

let COUNTER=COUNTER+1
echo $COUNTER