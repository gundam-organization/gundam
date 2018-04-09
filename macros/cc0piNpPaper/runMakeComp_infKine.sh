#!/bin/sh
nw="/data/t2k/dolan/fitting/feb17_refit/summaryPlots/oct17Nuis/infk/nuwro/"
nu="/data/t2k/dolan/fitting/feb17_refit/summaryPlots/oct17Nuis/infk/neut/"
ge="/data/t2k/dolan/fitting/feb17_refit/summaryPlots/oct17Nuis/infk/genie/"
gi="/data/t2k/dolan/fitting/feb17_refit/summaryPlots/oct17Nuis/infk/gibuu/"

cd /data/t2k/dolan/fitting/feb17_refit/summaryPlots/infKineComp/p/
ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/cc0piNpPaper/multiComp_infKine.cc .
root -b -l <<EOF
.L multiComp_infKine.cc+
multiComp_infKine_4comp("${nw}nuwro11_SF_multiDif_out.root", "${nw}nuwro11_SF_no2p2h_multiDif_out.root", "${nu}neut_540_lyon_infk_out.root", "${nw}nuwro11_LFGRPA_multiDif_out.root", "NuWro11q SF w/ 2p2h_{N}", "NuWro11q SF w/o 2p2h", "NEUT 5.4.0 LFG_{N}+RPA  w/ 2p2h_{N}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "genComp", "p")
multiComp_infKine_4comp("${ge}genier2124_defaultpMEC_nuisv1r0p1_multidif_out.root", "${nw}nuwro11_RFGRPA_multiDif_out.root", "${nu}neut5322_RFG_RPARW_out.root", "${nu}neut5322_RFG_RPARW_out.root", "GENIE 2.12.4 BRRFG w/ 2p2h_{E}", "NuWro11q RFG+RPA w/ 2p2h_{N}", "NEUT 5.3.2.2 RFG+RPA w/ 2p2h_{N}", "NEUT 5.3.2.2 RFG+RPA w/ 2p2h_{N}", "rfgComp", "p", true)
multiComp_infKine_4comp("${nu}neut5322_SF_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_no2p2hRW_multidif_out.root", "${nu}neut5322_SF_ExtraFSI_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_NoFSI_MA1p0RW_multidif_out.root", "NEUT 5.3.2.2 SF w/ 2p2h_{N}", "NEUT 5.3.2.2 SF w/o 2p2h", "NEUT 5.3.2.2 SF 2 times FSI w/ 2p2h_{N}", "NEUT 5.3.2.2 SF no FSI w/ 2p2h_{N}", "FSIComp", "p")
multiComp_infKine_4comp("${nu}neut_540_lyon_infk_out.root", "${gi}gibuu_211117_out.root", "${nw}nuwro11_LFGRPA_multiDif_out.root", "${nw}nuwro11_LFGRPA_multiDif_out.root", "NEUT 5.4.0 LFG_{N}+RPA  w/ 2p2h_{N}", "GiBUU 2016 w/ 2p2h_{G}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "lfgComp", "p", true)
multiComp_infKine_4comp("${nu}neut5322_SF_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_no2p2hRW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_onlyRES_multidif_out.root","${nu}neut5322_SF_MA1p0RW_onlyRES_multidif_out.root", "NEUT 5.3.2.2 SF CCQE+2p2h_{N}+RES+Other", "NEUT 5.3.2.2 SF CCQE+RES", "NEUT 5.3.2.2 SF RES", "NEUT 5.3.2.2 SF RES", "resComp", "p", true)
EOF

cd /data/t2k/dolan/fitting/feb17_refit/summaryPlots/infKineComp/a/
ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/cc0piNpPaper/multiComp_infKine.cc .
root -b -l <<EOF
.L multiComp_infKine.cc+
multiComp_infKine_4comp("${nw}nuwro11_SF_multiDif_out.root", "${nw}nuwro11_SF_no2p2h_multiDif_out.root", "${nu}neut_540_lyon_infk_out.root", "${nw}nuwro11_LFGRPA_multiDif_out.root", "NuWro11q SF w/ 2p2h_{N}", "NuWro11q SF w/o 2p2h", "NEUT 5.4.0 LFG_{N}+RPA  w/ 2p2h_{N}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "genComp", "a")
multiComp_infKine_4comp("${ge}genier2124_defaultpMEC_nuisv1r0p1_multidif_out.root", "${nw}nuwro11_RFGRPA_multiDif_out.root", "${nu}neut5322_RFG_RPARW_out.root", "${nu}neut5322_RFG_RPARW_out.root", "GENIE 2.12.4 BRRFG w/ 2p2h_{E}", "NuWro11q RFG+RPA w/ 2p2h_{N}", "NEUT 5.3.2.2 RFG+RPA w/ 2p2h_{N}", "NEUT 5.3.2.2 RFG+RPA w/ 2p2h_{N}", "rfgComp", "a", true)
multiComp_infKine_4comp("${nu}neut5322_SF_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_no2p2hRW_multidif_out.root", "${nu}neut5322_SF_ExtraFSI_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_NoFSI_MA1p0RW_multidif_out.root", "NEUT 5.3.2.2 SF w/ 2p2h_{N}", "NEUT 5.3.2.2 SF w/o 2p2h", "NEUT 5.3.2.2 SF 2 times FSI w/ 2p2h_{N}", "NEUT 5.3.2.2 SF no FSI w/ 2p2h_{N}", "FSIComp", "a")
multiComp_infKine_4comp("${nu}neut_540_lyon_infk_out.root", "${gi}gibuu_211117_out.root", "${nw}nuwro11_LFGRPA_multiDif_out.root", "${nw}nuwro11_LFGRPA_multiDif_out.root", "NEUT 5.4.0 LFG_{N}+RPA  w/ 2p2h_{N}", "GiBUU 2016 w/ 2p2h_{G}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "lfgComp", "a", true)
multiComp_infKine_4comp("${nu}neut5322_SF_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_no2p2hRW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_onlyRES_multidif_out.root","${nu}neut5322_SF_MA1p0RW_onlyRES_multidif_out.root", "NEUT 5.3.2.2 SF CCQE+2p2h_{N}+RES+Other", "NEUT 5.3.2.2 SF CCQE+RES", "NEUT 5.3.2.2 SF RES", "NEUT 5.3.2.2 SF RES", "resComp", "a", true)
EOF

cd /data/t2k/dolan/fitting/feb17_refit/summaryPlots/infKineComp/ip/
ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/cc0piNpPaper/multiComp_infKine.cc .
root -b -l <<EOF
.L multiComp_infKine.cc+
multiComp_infKine_4comp("${nw}nuwro11_SF_multiDif_out.root", "${nw}nuwro11_SF_no2p2h_multiDif_out.root", "${nu}neut_540_lyon_infk_out.root", "${nw}nuwro11_LFGRPA_multiDif_out.root", "NuWro11q SF w/ 2p2h_{N}", "NuWro11q SF w/o 2p2h", "NEUT 5.4.0 LFG_{N}+RPA  w/ 2p2h_{N}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "genComp", "ip")
multiComp_infKine_4comp("${ge}genier2124_defaultpMEC_nuisv1r0p1_multidif_out.root", "${nw}nuwro11_RFGRPA_multiDif_out.root", "${nu}neut5322_RFG_RPARW_out.root", "${nu}neut5322_RFG_RPARW_out.root", "GENIE 2.12.4 BRRFG w/ 2p2h_{E}", "NuWro11q RFG+RPA w/ 2p2h_{N}", "NEUT 5.3.2.2 RFG+RPA w/ 2p2h_{N}", "NEUT 5.3.2.2 RFG+RPA w/ 2p2h_{N}", "rfgComp", "ip", true)
multiComp_infKine_4comp("${nu}neut5322_SF_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_no2p2hRW_multidif_out.root", "${nu}neut5322_SF_ExtraFSI_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_NoFSI_MA1p0RW_multidif_out.root", "NEUT 5.3.2.2 SF w/ 2p2h_{N}", "NEUT 5.3.2.2 SF w/o 2p2h", "NEUT 5.3.2.2 SF 2 times FSI w/ 2p2h_{N}", "NEUT 5.3.2.2 SF no FSI w/ 2p2h_{N}", "FSIComp", "ip")
multiComp_infKine_4comp("${nu}neut_540_lyon_infk_out.root", "${gi}gibuu_211117_out.root", "${nw}nuwro11_LFGRPA_multiDif_out.root", "${nw}nuwro11_LFGRPA_multiDif_out.root", "NEUT 5.4.0 LFG_{N}+RPA  w/ 2p2h_{N}", "GiBUU 2016 w/ 2p2h_{G}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "lfgComp", "ip", true)
multiComp_infKine_4comp("${nu}neut5322_SF_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_no2p2hRW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_onlyRES_multidif_out.root","${nu}neut5322_SF_MA1p0RW_onlyRES_multidif_out.root", "NEUT 5.3.2.2 SF CCQE+2p2h_{N}+RES+Other", "NEUT 5.3.2.2 SF CCQE+RES", "NEUT 5.3.2.2 SF RES", "NEUT 5.3.2.2 SF RES", "resComp", "ip", true)
EOF

cd /data/t2k/dolan/fitting/feb17_refit/summaryPlots/infKineComp
exit
