#!/bin/sh

#Example of how to run:
# ./quickFitToXsec.sh /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutAir5_2DV2.root /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/inputs/NeutAir5_2DV2.root 331.6 331.6 false dpt 111 "-c 0" "-c 0"

source /home/dolan/.bashrc
runnd280
sourcefitter

inputMC=$1
data=$2
MCPOT=$3
dataPOT=$4
reg1=$5
reg2=$6
isRealData=$7
var=$8
runmode=$9
useInputEff=${12}

echo "Running quick fit to xsec"
potRatio=$(bc <<< "scale=5;$dataPOT/$MCPOT")
echo "Calculated POT ratio:"
echo ${potRatio}

echo "Input MC location is"
echo $1
echo "Input data location is:"
echo $2
echo "MC POT is"
echo $3
echo "Data POT is"
echo $4
echo "reg1 is"
echo $5
echo "reg 2 is"
echo $6
echo "isRealData is"
echo $7
echo "var is"
echo $8
echo "runmode is"
echo $9
echo "fit optarg is"
echo ${10}
echo "prop optarg is"
echo ${11}
echo "use input eff is"
echo ${12}

if [ "${var}" == "test" ]
then
	echo "Running in test mode:"
	echo "Input MC location is"
	echo $1
	echo "Input data location is:"
	echo $2
	echo "MC POT is"
	echo $3
	echo "Data POT is"
	echo $4
	echo "reg1 is"
	echo $5
	echo "reg 2 is"
	echo $6
	echo "isRealData is"
	echo $7
	echo "var is"
	echo $8
	echo "runmode is"
	echo $9
	echo "fit optarg is"
	echo ${10}
  echo "prop optarg is"
  echo ${11}
  echo "use input eff is"
  echo ${12}

elif [ "${var}" == "dphit" ]
then
	echo "Running is dphit mode"
	if (( "${runmode}" == "111" )) || (( "${runmode}" == "100" )) 
	then
	  echo "Running fitter, see fitoutput.log for details"
      CCQEFit_dphiT -i ${inputMC} -f ${data} -o "quickFtX_fitOut.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -m 2 -M 2 -n ${dataPOT} -N ${MCPOT} ${10} &> fitoutput.log 
  	  if  (( "${runmode}" == "100" ))
  	  then
  	    echo "Fit only mode finished"
  	    exit
  	  fi
    fi
   	if (("${runmode}" == "111")) || (("${runmode}" == "011")) 
	then 
   	  echo "Running propogator, see propoutput.log for details"
      PropError_dphiT -i ${inputMC} -d ${data} -f "quickFtX_fitOut.root" -s ${RANDOM} -n ${dataPOT} -N ${MCPOT} -o "quickFtX_propErrOut.root" -t 5000 ${11} &> propoutput.log
    fi
    ln -s $XSLLHFITTER/macros/calcXSecWithErrors2.cc .
   	echo "Running xsec builder, see xsecout.log for details"
root -b <<EOF > xsecout.log
cout << "running calcXSecWithErrors2" << endl;
.L calcXSecWithErrors2.cc+
calcXsecWithErrors2("quickFtX_propErrOut.root", "${inputMC}", "quickFtX_fitOut.root", "${data}", "quickFtX_xsecOut_temp.root", ${potRatio}, $(bc <<< "scale=5;$MCPOT/100"), ${isRealData}, "${var}", ${useInputEff})
EOF
  #So we don't overwrite a histo that needs to be read by other jobs
  sleep 10
  mv quickFtX_xsecOut_temp.root quickFtX_xsecOut.root
	echo "Finished"
elif [ "${var}" == "dat" ]
then
	echo "Running is dat mode"
	if (( "${runmode}" == "111" )) || (( "${runmode}" == "100" )) 
	then
	  echo "Running fitter, see fitoutput.log for details"
      CCQEFit_daT -i ${inputMC} -f ${data} -o "quickFtX_fitOut.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -m 2 -M 2 -n ${dataPOT} -N ${MCPOT} ${10} &> fitoutput.log 
      if  (( "${runmode}" == "100" ))
      then
        echo "Fit only mode finished"
        exit
      fi
    fi
   	if (("${runmode}" == "111")) || (("${runmode}" == "011")) 
	then 
   	  echo "Running propogator, see propoutput.log for details"
      PropError_daT -i ${inputMC} -d ${data} -f "quickFtX_fitOut.root" -s ${RANDOM} -n ${dataPOT} -N ${MCPOT} -o "quickFtX_propErrOut.root" -t 5000 ${11} &> propoutput.log
    fi
    ln -s $XSLLHFITTER/macros/calcXSecWithErrors2.cc .
   	echo "Running xsec builder, see xsecout.log for details"
root -b <<EOF > xsecout.log
cout << "running calcXSecWithErrors2" << endl;
.L calcXSecWithErrors2.cc+
calcXsecWithErrors2("quickFtX_propErrOut.root", "${inputMC}", "quickFtX_fitOut.root", "${data}", "quickFtX_xsecOut_temp.root", ${potRatio}, $(bc <<< "scale=5;$MCPOT/100"), ${isRealData}, "${var}", ${useInputEff})
EOF
  #So we don't overwrite a histo that needs to be read by other jobs
  sleep 10
  mv quickFtX_xsecOut_temp.root quickFtX_xsecOut.root
  echo "Finished"
else
	echo "Running in dpt mode"
	if (( "${runmode}" == "111" )) || (( "${runmode}" == "100" )) 
	then
	  echo "Running fitter, see fitoutput.log for details"
      CCQEFit -i ${inputMC} -f ${data} -o "quickFtX_fitOut.root" -s ${RANDOM} -r ${reg1} -R ${reg2} -m 2 -M 2 -n ${dataPOT} -N ${MCPOT} ${10} &> fitoutput.log 
      if  (( "${runmode}" == "100" ))
      then
        echo "Fit only mode finished"
        exit
      fi
    fi
   	if (("${runmode}" == "111")) || (("${runmode}" == "011")) 
	then 
   	  echo "Running propogator, see propoutput.log for details"
      PropError -i ${inputMC} -d ${data} -f "quickFtX_fitOut.root" -s ${RANDOM} -n ${dataPOT} -N ${MCPOT} -o "quickFtX_propErrOut.root" -t 5000 -F 1 -G 8 ${11} &> propoutput.log
    fi
    ln -s $XSLLHFITTER/macros/calcXSecWithErrors2.cc .
   	echo "Running xsec builder, see xsecout.log for details"
root -b <<EOF > xsecout.log
cout << "running calcXSecWithErrors2" << endl;
.L calcXSecWithErrors2.cc+
calcXsecWithErrors2("quickFtX_propErrOut.root", "${inputMC}", "quickFtX_fitOut.root", "${data}", "quickFtX_xsecOut_temp.root", ${potRatio}, $(bc <<< "scale=5;$MCPOT/100"), ${isRealData}, "${var}", ${useInputEff})
EOF
	#So we don't overwrite a histo that needs to be read by other jobs
  sleep 10
  mv quickFtX_xsecOut_temp.root quickFtX_xsecOut.root
  echo "Finished"
fi
