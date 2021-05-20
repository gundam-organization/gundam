#!/bin/sh
for dir in $(ls oct17Nuis/multDif/); do
  #if [[ "${dir}" != "gibuu" ]]; then
  #  continue 
  #fi
  echo "Using files in dir: ${dir}"
  cd oct17Nuis/multDif/${dir};
  for file in $(ls *.root) ; do
    echo "Using file: ${file}"
    cd $1
root -b -l <<EOF
.L makeComp_multiDif.cc+
makeComp_multiDif("oct17Nuis/multDif/${dir}/${file}", "/data/t2k/dolan/software/nuisance/oct17/nuisance/data/T2K/CC0pi/multidif_3D_pcoscos.root", "/data/t2k/dolan/fitting/feb17_refit/summaryPlots/multDifCompTest/Np_${file}", "${file:0:${#file}-5}")
makeComp_multiDif_0p("oct17Nuis/multDif/${dir}/${file}", "/data/t2k/dolan/software/nuisance/oct17/nuisance/data/T2K/CC0pi/multidif_2D_pcos.root", "/data/t2k/dolan/fitting/feb17_refit/summaryPlots/multDifCompTest/0p_${file}", "${file:0:${#file}-5}")
.q
EOF
root -b -l <<EOF
.L makeComp_multiDif.cc+
makeComp_allComb("/data/t2k/dolan/fitting/feb17_refit/summaryPlots/multDifCompTest/0p_${file}","/data/t2k/dolan/fitting/feb17_refit/summaryPlots/multDifCompTest/Np_${file}", "oct17Nuis/multDif/${dir}/${file}", "/home/dolan/cc0piPaper/pierreResult/relCovarWStat/out_xsec.root", "/data/t2k/dolan/fitting/feb17_refit/summaryPlots/multDifCompTest/comb_${file}", "${file:0:${#file}-5}")
.q
EOF
    echo "Finished with file: ${file}"
  done;
  cd $1
done;
cd $1

for file in $(ls *.pdf) ; do
  echo "Copying file: ${file}"
  mv ${file} /data/t2k/dolan/fitting/feb17_refit/summaryPlots/multDifCompTest/pdf/${file}
done;

for file in $(ls *.png) ; do
  echo "Copying file: ${file}"
  mv ${file} /data/t2k/dolan/fitting/feb17_refit/summaryPlots/multDifCompTest/png/${file}  
done;

cd /data/t2k/dolan/fitting/feb17_refit/summaryPlots/multDifCompTest
root -b -l <<EOF
.L multiComp_multiDif.cc+
multiComp_multiDif("nuwro11_LFGRPA_multiDif_out.root", "nuwro11_SF_multiDif_out.root", "NuWro11q LFG+RPA", "NuWro11q SF w/ 2p2h", "multiCompOut/NuWroSF_LFG_comp")
multiComp_multiDif("nuwro11_LFGRPA_multiDif_out.root", "nuwro11_RFGRPA_multiDif_out.root", "NuWro11q LFG+RPA", "NuWro11q RFG+RPA", "multiCompOut/NuWroLFG_RFG_comp")
multiComp_multiDif("nuwro11_SF_multiDif_out.root", "nuwro11_SF_no2p2h_multiDif_out.root", "NuWro11q SF w/2p2h", "NuWro11q SF w/o 2p2h", "multiCompOut/NuWroSF_2p2h_comp")
multiComp_multiDif("genier2124_default_multiDif_out.root", "genier2124_defaultpMEC_nuisv1r0p1_multidif_out.root", "GENIE 2.12.4 BRRFG w/o 2p2h", "GENIE 2.12.4 BRRFG w/ 2p2h", "multiCompOut/Genie_2p2h_comp")
multiComp_multiDif("neut_540_lyon_multidif_out.root", "genier2124_default_multiDif_out.root", "NEUT 5.4.0 LFG+RPA (Nieves)", "GENIE 2.12.4 BRRFG w/o 2p2h", "multiCompOut/Neut540_Genie_comp")
multiComp_multiDif("neut_540_lyon_multidif_out.root", "nuwro11_LFGRPA_multiDif_out.root", "NEUT 5.4.0 LFG+RPA (Nieves)", "NuWro11q LFG+RPA", "multiCompOut/Neut540_NuWroLFG_comp")
multiComp_multiDif("nuwro11_LFGRPA_multiDif_out.root", "genier2124_default_multiDif_out.root", "NuWro11q LFG+RPA", "GENIE 2.12.4 BRRFG w/o 2p2h", "multiCompOut/NuWroLFG_Genie_comp")
multiComp_multiDif("neut_540_lyon_multidif_out.root", "gibuu_211117_out.root", "NEUT 5.4.0 LFG_{N}+RPA  w/ 2p2h_{N}", "GiBUU 2016 w/ 2p2h", "multiCompOut/Neut540_GiBUU_comp")
//
multiComp_multiDif_4comp("nuwro11_SF_multiDif_out.root", "nuwro11_SF_no2p2h_multiDif_out.root", "neut_540_lyon_multidif_out.root", "nuwro11_LFGRPA_multiDif_out.root", "NuWro11q SF w/ 2p2h_{N}", "NuWro11q SF w/o 2p2h", "NEUT 5.4.0 LFG_{N}+RPA  w/ 2p2h_{N}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "multiCompOut/fourComp_genComp")
multiComp_multiDif_4comp("genier2124_defaultpMEC_nuisv1r0p1_multidif_out.root", "nuwro11_RFGRPA_multiDif_out.root", "neut5322_RFG_RPARW_out.root", "neut5322_RFG_RPARW_out.root", "GENIE 2.12.4 BRRFG w/ 2p2h_{E}", "NuWro11q RFG+RPA w/ 2p2h_{N}", "NEUT 5.3.2.2 RFG+RPA w/ 2p2h_{N}", "NEUT 5.3.2.2 RFG+RPA w/ 2p2h_{N}", "multiCompOut/fourComp_rfgComp", true)
multiComp_multiDif_4comp("neut5322_SF_MA1p0RW_multidif_out.root", "neut5322_SF_ExtraFSI_MA1p0RW_multidif_out.root", "neut5322_SF_NoFSI_MA1p0RW_multidif_out.root", "neut5322_SF_MA1p0RW_no2p2hRW_multidif_out.root", "NEUT 5.3.2.2 SF w/ 2p2h_{N}", "NEUT 5.3.2.2 SF 2 times FSI w/ 2p2h_{N}", "NEUT 5.3.2.2 SF no FSI w/ 2p2h_{N}", "NEUT 5.3.2.2 SF w/o 2p2h", "multiCompOut/fourComp_FSIComp")
multiComp_multiDif_4comp("neut_540_lyon_multidif_out.root", "gibuu_211117_out.root", "nuwro11_LFGRPA_multiDif_out.root", "nuwro11_LFGRPA_multiDif_out.root", "NEUT 5.4.0 LFG_{N}+RPA  w/ 2p2h_{N}", "GiBUU 2016 w/ 2p2h_{G}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "NuWro11q LFG+RPA w/ 2p2h_{N}", "multiCompOut/fourComp_lfgComp", true)
multiComp_multiDif_4comp("${nu}neut5322_SF_MA1p0RW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_no2p2hRW_multidif_out.root", "${nu}neut5322_SF_MA1p0RW_onlyRES_multidif_out.root","${nu}neut5322_SF_MA1p0RW_onlyRES_multidif_out.root", "NEUT 5.3.2.2 SF CCQE+2p2h_{N}+RES+Other", "NEUT 5.3.2.2 SF CCQE+RES", "NEUT 5.3.2.2 SF RES", "NEUT 5.3.2.2 SF RES", "multiCompOut/fourComp_resComp", true)
EOF

exit
