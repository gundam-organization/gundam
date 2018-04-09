#!/bin/sh
source /home/dolan/.bashrc
runnd280
sourcefitter

echo "Run as:"
echo "makeFitterInputs.sh /path/to/InputMicroTree.root outputFileTag dataToSimPotRatio $PWD"
echo "If not running on real data, disregard the realDataFormat outputs"
echo "If running on real data everything other than the realDataFormat outputs may fail"
echo "Running ..."

inputMT=$1
outTag=$2
potRatio=$3
here=$4

cd ${here}
ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/cc0piNpPaper/treeConvert_paper.cc .

root <<EOF
.L treeConvert_paper.cc+
//MC format
treeConvert_paper("${inputMT}", "default", "truth", "./${outTag}_dpt_allStats.root", 1, 0, 0, 0, 0, "recDpT", "trueDpT", "trueDpT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_paper("${inputMT}", "default", "truth", "./${outTag}_dpt_dataStats.root", 1, 0, 0, ${3}, 0, "recDpT", "trueDpT", "trueDpT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_paper("${inputMT}", "default", "truth", "./${outTag}_dphit_allStats.root", 1, 0, 0, 0, 0, "recDphiT", "trueDphiT", "trueDphiT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_paper("${inputMT}", "default", "truth", "./${outTag}_dphit_dataStats.root", 1, 0, 0, ${3}, 0, "recDphiT", "trueDphiT", "trueDphiT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_paper("${inputMT}", "default", "truth", "./${outTag}_dat_allStats.root", 1, 0, 0, 0, 0, "recDalphaT", "trueDalphaT", "trueDalphaT", "recDpT", "trueDpT", "trueDpT", false, true, true)
treeConvert_paper("${inputMT}", "default", "truth", "./${outTag}_dat_dataStats.root", 1, 0, 0, ${3}, 0, "recDalphaT", "trueDalphaT", "trueDalphaT", "recDpT", "trueDpT", "trueDpT", false, true, true)
.q
EOF

echo "finished"

#Run as: ./makeFitterInputs_paper.sh /data/t2k/dolan/MECProcessing/CC0Pi/feb17HL2/job_NeutAiNoSystV2_out/allMerged.root NeutAirV2 0.2 $PWD &>NeutAirInputMaker.log & 
