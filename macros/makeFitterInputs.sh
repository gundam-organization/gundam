#!/bin/sh
#source /home/dolan/.bashrc
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
ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/treeConvert_jan17hl2.cc .

root <<EOF
.L treeConvert_jan17hl2.cc+
//Real data format
treeConvert_jan17hl2("${inputMT}", "default", "default", "./${outTag}_dpt_realDataFormat.root", 1, 0, 0, 0, 0, "recDpT", "trueDpT", "trueDpT", "recDalphaT", "trueDalphaT", "trueDalphaT", true, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "default", "./${outTag}_dphit_realDataFormat.root", 1, 0, 0, 0, 0, "recDphiT", "trueDphiT", "trueDphiT", "recDalphaT", "trueDalphaT", "trueDalphaT", true, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "default", "./${outTag}_dat_realDataFormat.root", 1, 0, 0, 0, 0, "recDalphaT", "trueDalphaT", "trueDalphaT", "recDpT", "trueDpT", "trueDpT", true, true, true)
//MC format
treeConvert_jan17hl2("${inputMT}", "default", "truth", "./${outTag}_dpt_allStats.root", 1, 0, 0, 0, 0, "recDpT", "trueDpT", "trueDpT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "truth", "./${outTag}_dpt_dataStats.root", 1, 0, 0, ${3}, 0, "recDpT", "trueDpT", "trueDpT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "truth", "./${outTag}_dpt_no2p2h_dataStats.root", 1, 0, 0, ${3}, 1, "recDpT", "trueDpT", "trueDpT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "truth", "./${outTag}_dphit_allStats.root", 1, 0, 0, 0, 0, "recDphiT", "trueDphiT", "trueDphiT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "truth", "./${outTag}_dphit_dataStats.root", 1, 0, 0, ${3}, 0, "recDphiT", "trueDphiT", "trueDphiT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "truth", "./${outTag}_dphit_no2p2h_dataStats.root", 1, 0, 0, ${3}, 1, "recDphiT", "trueDphiT", "trueDphiT", "recDalphaT", "trueDalphaT", "trueDalphaT", false, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "truth", "./${outTag}_dat_allStats.root", 1, 0, 0, 0, 0, "recDalphaT", "trueDalphaT", "trueDalphaT", "recDpT", "trueDpT", "trueDpT", false, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "truth", "./${outTag}_dat_dataStats.root", 1, 0, 0, ${3}, 0, "recDalphaT", "trueDalphaT", "trueDalphaT", "recDpT", "trueDpT", "trueDpT", false, true, true)
treeConvert_jan17hl2("${inputMT}", "default", "truth", "./${outTag}_dat_no2p2h_dataStats.root", 1, 0, 0, ${3}, 1, "recDalphaT", "trueDalphaT", "trueDalphaT", "recDpT", "trueDpT", "trueDpT", false, true, true)
.q
EOF

echo "finished"
