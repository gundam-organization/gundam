#!/bin/sh
#source /home/dolan/.bashrc
#runxstoolmar16v1r5
time
mt="/data/t2k/dolan/xsToolBasedMEC/CC0PiAnl/microtrees/mar16/job_NeutAirAllSystV5_out/allMerged.root"
wd="/data/t2k/dolan/xsToolBasedMEC/CC0PiAnl/weights/outdir/cc0piv27Mar16NeutAirV2/"
var=$2


if [ $1 == 'test' ]; then
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolXsSig_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots	
mkdir test
cd test
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing test mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NXSec_MaCCQE\"       ,0.95,0.14,${topo})" > /dev/null
EOF
qsub xsToolXsSig_T${topo}R${reac}.sh
	done
fi


if [ $1 == 'xsToolXsSig' ]; then
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolXsSig_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots	
mkdir xsToolXsSig
cd xsToolXsSig
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing xsToolXsSig mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NXSec_MaCCQE\"       ,0.95,0.14,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2014a_pF_C12\"   ,1.03,0.15,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWGMEC_Norm_C12\"   ,0.27,0.73,${topo})" > /dev/null
# Ignore Oxygen Params for the moment
#root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2014a_pF_O16\"   ,1.21,0.45,${topo})" > /dev/null
#root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWGMEC_Norm_O16\"   ,1.21,0.45,${topo})" > /dev/null
#root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2014a_Eb_O16\"   ,1.21,0.45,${topo})" > /dev/null
EOF
qsub xsToolXsSig_T${topo}R${reac}.sh
	done
fi

if [ $1 == 'xsToolXsBkg' ]; then
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolXsBkg_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots  	 
mkdir xsToolXsBkg
cd xsToolXsBkg
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing xsToolXsBkg mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NXSec_CA5RES\"       ,1.00,0.12,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NXSec_MaNFFRES\"     ,1.00,0.16,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NXSec_BgSclRES\"     ,1.00,0.15,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_ccnueE0\"  ,1.00,0.02,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_dismpishp\",1.00,0.40,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_cccohE0\"  ,1.00,1.00,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_nccohE0\"  ,1.00,0.30,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_ncotherE0\",1.00,0.30,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2014a_Eb_C12\"   ,1.00,0.36,${topo})" > /dev/null
EOF
qsub xsToolXsBkg_T${topo}R${reac}.sh
	done
fi

if [ $1 == 'xsToolXsBkg1' ]; then
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolXsBkg_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots  	 
mkdir xsToolXsBkg
cd xsToolXsBkg
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing xsToolXsBkg mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NXSec_CA5RES\"       ,1.00,0.12,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NXSec_MaNFFRES\"     ,1.00,0.16,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NXSec_BgSclRES\"     ,1.00,0.15,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_ccnueE0\"  ,1.00,0.02,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_dismpishp\",1.00,0.40,${topo})" > /dev/null
EOF
qsub xsToolXsBkg_T${topo}R${reac}.sh
	done
fi

if [ $1 == 'xsToolXsBkg2' ]; then
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolXsBkg_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots  	 
mkdir xsToolXsBkg
cd xsToolXsBkg
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing xsToolXsBkg mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_cccohE0\"  ,1.00,1.00,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_nccohE0\"  ,1.00,0.30,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2012a_ncotherE0\",1.00,0.30,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NIWG2014a_Eb_C12\"   ,1.00,0.36,${topo})" > /dev/null
EOF
qsub xsToolXsBkg_T${topo}R${reac}.sh
	done
fi


if [ $1 == 'xsToolFsiSig' ]; then
	mt="/data/t2k/dolan/MECProcessing/CC0Pi/mar15HL2/job_Neut6DAirV1_out/allMerged.root"
	wd="/data/t2k/dolan/fitting/modelWeights/protonFSI/v1r5_xsTools/outdir/"
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolFsiSig_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolwprotonfsi
ulimit -n 4096
cd /data/t2k/dolan/systPlots  
mkdir xsToolFsiSig
cd xsToolFsiSig  	
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing xsToolXsSig mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NINuke_MFP_N\" ,1.00,0.05,${topo})" > /dev/null
EOF
qsub -l cput=00:30:00 xsToolFsiSig_T${topo}R${reac}.sh
	done
fi

if [ $1 == 'xsToolFsiBkg' ]; then
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolFsiBkg_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots  	
mkdir xsToolFsiBkg
cd xsToolFsiBkg  	
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing xsToolXsBkg mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrInelHigh_pi\",1.00,0.33,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrInelLow_pi\" ,1.00,0.41,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrPiProd_pi\"  ,1.00,0.50,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrAbs_pi\"     ,1.00,0.41,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrCExLow_pi\"  ,1.00,0.57,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrCExHigh_pi\" ,1.00,0.28,${topo})" > /dev/null
EOF
qsub xsToolFsiBkg_T${topo}R${reac}.sh
	done
fi

if [ $1 == 'xsToolFsiBkg1' ]; then
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolFsiBkg_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots  	
mkdir xsToolFsiBkg
cd xsToolFsiBkg  	
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing xsToolXsBkg mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrInelHigh_pi\",1.00,0.33,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrInelLow_pi\" ,1.00,0.41,${topo})" > /dev/null
EOF
qsub xsToolFsiBkg_T${topo}R${reac}.sh
	done
fi

if [ $1 == 'xsToolFsiBkg2' ]; then
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolFsiBkg_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots  	
mkdir xsToolFsiBkg
cd xsToolFsiBkg  	
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing xsToolXsBkg mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrPiProd_pi\"  ,1.00,0.50,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrAbs_pi\"     ,1.00,0.41,${topo})" > /dev/null
EOF
qsub xsToolFsiBkg_T${topo}R${reac}.sh
	done
fi

if [ $1 == 'xsToolFsiBkg3' ]; then
	mkdir job_${1}_${var}
	cp systPlots_${var}.C ./job_${1}_${var}
	cd job_${1}_${var}
	#for topo in 1 2 3 5 6 7 8 9; do
	#  for reac in 0 1 2 3 4 5 7; do
	for topo in 1 2 3 5 6 7; do
cat <<EOF > xsToolFsiBkg_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots  	
mkdir xsToolFsiBkg
cd xsToolFsiBkg  	
mkdir output
ln -s ../systPlots_${var}.C .
echo "Processing xsToolXsBkg mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrCExLow_pi\"  ,1.00,0.57,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"${mt}\",\"${wd}\",\"NCasc_FrCExHigh_pi\" ,1.00,0.28,${topo})" > /dev/null
EOF
qsub xsToolFsiBkg_T${topo}R${reac}.sh
	done
fi

if [ $1 == 'QE' ]; then
	for topo in 1 2 3 5 6 7 8 9; do
cat <<EOF > QE_T${topo}R${reac}.sh 	
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots  	
mkdir job_${1}
cd job_${1}  
echo "Processing QE mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
                                 	      \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
                                 	      \"NXSec_MaCCQE\",1.21,0.45,${topo})" > /dev/null
EOF
qsub MARES_T${topo}R${reac}.sh
	done
fi

if [ $1 == 'MARES' ]; then
	for topo in 1 2 3 5 6 7 8 9; do
cat <<EOF > MARES_T${topo}R${reac}.sh 
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots
mkdir job_${1}
cd job_${1}  
echo "Processing RES mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
                                 \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
                                 \"NXSec_MaNFFRES\",1.41,0.21,${topo})" > /dev/null
EOF
qsub MARES_T${topo}R${reac}.sh
	done
fi


if [ $1 == 'FSI' ]; then
	for topo in 1 2 3 5 6 7 8 9; do
cat <<EOF > FSI_T${topo}R${reac}.sh 
#!/bin/sh
source /home/dolan/.bashrc
runxstoolmar16v1r5
ulimit -n 4096
cd /data/t2k/dolan/systPlots
mkdir job_${1}
cd job_${1}  
echo "Processing FSI mode on topo ${topo} and reac ${reac}"
root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
                                 \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
                                 \"NCasc_FrInelHigh_pi\",1.0,0.338,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
                                 \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
                                 \"NCasc_FrInelLow_pi\",1.0,0.412,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
                                 \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
                                 \"NCasc_FrPiProd_pi\",1.0,0.500,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
                                 \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
                                 \"NCasc_FrAbs_pi\",1.0,0.412,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
                                 \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
                                 \"NCasc_FrCExLow_pi\",1.0,0.567,${topo})" > /dev/null
root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
                                 \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
                                 \"NCasc_FrCExHigh_pi\",1.0,0.278,${topo})" > /dev/null
EOF
qsub MARES_T${topo}R${reac}.sh
	done
fi


if [ $1 == 'RES' ]; then
	for topo in 1 2 3 5 6 7 8 9; do
	    echo "Processing RES mode on topo ${topo} and reac ${reac}"
	    root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
	                                     \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
	                                     \"NXSec_MaRES\",1.41,0.21,${topo})" > /dev/null
	    root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
	                                     \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
	                                     \"NIWG2012a_cc1piE0\",1.0,0.43,${topo})" > /dev/null
	    root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
	                                     \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
	                                     \"NIWG2012a_cc1piE1\",1.0,0.40,${topo})" > /dev/null
	    root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
	                                     \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
	                                     \"NIWG2012a_dismpishp\",1.0,0.40,${topo})" > /dev/null
	    root -b -q -l "systPlots_${var}.C(\"../microtrees/mar16/job_NeutAirAllSystV4_out/allMerged.root\",\
	                                     \"../weights/outdir/cc0piv27Mar16NeutAirV2/\",\
	                                     \"NIWG2012a_nc1pi0E0\",1.0,0.40,${topo})" > /dev/null
	done
fi

time

#xsTools dials:

# GROUP flux
#   DIAL JEnu2013a_nd5numu0
#   DIAL JEnu2013a_nd5numu1
#   DIAL JEnu2013a_nd5numu2
#   DIAL JEnu2013a_nd5numu3
#   DIAL JEnu2013a_nd5numu4
#   DIAL JEnu2013a_nd5numu5
#   DIAL JEnu2013a_nd5numu6
#   DIAL JEnu2013a_nd5numu7
#   DIAL JEnu2013a_nd5numu8
#   DIAL JEnu2013a_nd5numu9
#   DIAL JEnu2013a_nd5numu10
#   DIAL JEnu2013a_nd5numub0
#   DIAL JEnu2013a_nd5numub1
#   DIAL JEnu2013a_nd5numub2
#   DIAL JEnu2013a_nd5numub3
#   DIAL JEnu2013a_nd5numub4
#   DIAL JEnu2013a_nd5nue0
#   DIAL JEnu2013a_nd5nue1
#   DIAL JEnu2013a_nd5nue2
#   DIAL JEnu2013a_nd5nue3
#   DIAL JEnu2013a_nd5nue4
#   DIAL JEnu2013a_nd5nue5
#   DIAL JEnu2013a_nd5nue6
#   DIAL JEnu2013a_nd5nueb0
#   DIAL JEnu2013a_nd5nueb1
# END GROUP
# GROUP fsi
#   DIAL NCasc_FrInelLow_pi
#   DIAL NCasc_FrInelHigh_pi
#   DIAL NCasc_FrPiProd_pi
#   DIAL NCasc_FrAbs_pi
#   DIAL NCasc_FrCExLow_pi
#   DIAL NCasc_FrCExHigh_pi
# END GROUP
# GROUP xs
#   DIAL NXSec_MaCCQE
#   DIAL NIWG2014a_pF_C12
#   DIAL NIWGMEC_Norm_C12
#   DIAL NIWG2014a_Eb_C12
#   DIAL NIWG2014a_pF_O16
#   DIAL NIWGMEC_Norm_O16
#   DIAL NIWG2014a_Eb_O16
#   DIAL NXSec_CA5RES
#   DIAL NXSec_MaNFFRES
#   DIAL NXSec_BgSclRES
#   DIAL NIWG2012a_ccnueE0
#   DIAL NIWG2012a_dismpishp
#   DIAL NIWG2012a_cccohE0
#   DIAL NIWG2012a_nccohE0
#   DIAL NIWG2012a_ncotherE0
# END GROUP

# Proton FSI in NEUT 6D:
#NINuke_MFP_N

#BANFF File index:

#First params are Flux, starts with NDNuModeNumu0 (i.e xstool first flux param)
#Should read first 24 params to get flux covar matrix

#Param 100 is first flux param, they then read: (with param values)
 # TObjString = FSI_INEL_LO_E 100 0
 # TObjString = FSI_INEL_HI_E 101 0
 # TObjString = FSI_PI_PROD 102 0
 # TObjString = FSI_PI_ABS 103 0
 # TObjString = FSI_CEX_LO_E 104 0
 # TObjString = FSI_CEX_HI_E 105 0
 # TObjString = MAQE 106 0.958333
 # TObjString = pF_C 107 1.02765
 # TObjString = MEC_C 108 0.27
 # TObjString = EB_C 109 1
 # TObjString = pF_O 110 1
 # TObjString = MEC_O 111 1
 # TObjString = EB_O 112 1
 # TObjString = CA5 113 1
 # TObjString = MANFFRES 114 1
 # TObjString = BgRES 115 1
 # TObjString = CCNUE_0 116 1
 # TObjString = DISMPISHP 117 0
 # TObjString = CCCOH_C_0 118 1 
 # TObjString = CCCOH_O_0 119 1
 # TObjString = NCCOH_0 120 1
 # TObjString = NCOTHER_0 121 1

 #Cov Mat, if element not listed then it's 0:

 #Pion FSI

#       |       0    |       1    |       2    |       3    |       4    |      5    |
# ------------------------------------------------------------------------------------
#    0 |       0.17   -0.002778           0     0.02273       0.005		0
#    1 |  -0.002778      0.1142     -0.1667   -0.001263   -0.002083		-0.09259
#    2 |          0     -0.1667        0.25  -5.204e-18           0		0.1389	
#    3 |    0.02273   -0.001263  -5.204e-18      0.1694   -0.002273		-3.469e-18
#    4 |      0.005   -0.002083           0   -0.002273      0.3213		1.735e-18
#    5 |          0    -0.09259      0.1389  -3.469e-18   1.735e-18		0.07716


#QE + MEC + Nuc:

#      |       5    |       6    |       7    |       8    |       9    |
# ----------------------------------------------------------------------
#    6 |          0     0.02141   -0.004853     0.05314           0
#    7 |          0   -0.004853     0.02044    0.004588           0
#    8 |          0     0.05314    0.004588       0.537           0
#    9 |          0           0           0           0      0.1296
#   10 |          0           0           0           0           0
#   11 |          0           0           0      0.3099           0



#      |      10    |      11    |      12    |      13    |      14    |
# ----------------------------------------------------------------------
#    8 |          0      0.3099           0           0           0
#    9 |          0           0           0           0           0
#   10 |    0.01902           0           0           0           0
#   11 |          0       0.537           0           0           0

#RES + Other

#   12 |          0           0      0.1111           0           0
#   13 |          0           0           0     0.01412           0
#   14 |          0           0           0           0     0.02493


#        |      15    |      16    |      17    |      18    |      19    |
# ----------------------------------------------------------------------

#   15 |    0.02367           0           0           0           0
#   16 |          0      0.0004           0           0           0
#   17 |          0           0        0.16           0           0
#   18 |          0           0           0           1           0
#   19 |          0           0           0           0           1

#      |      20    |      21    |
# ----------------------------------------------------------------------
#   20 |       0.09           0
#   21 |          0        0.09
