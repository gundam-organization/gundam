#!/bin/sh
cd /data/t2k/dolan/paperStuff/recoPlots/inputs
./makeFitterInputs_paper.sh /data/t2k/dolan/MECProcessing/CC0Pi/feb17HL2/job_NeutAiNoSystV2_out/allMerged.root NeutAirV2 0.2 $PWD &>NeutAirInputMaker.log
root <<EOF
.L treeConvert_paper.cc+
treeConvert_paper("/data/t2k/dolan/MECProcessing/CC0Pi/feb17HL2/p6K_rdp_6kto12k_FHC_out.root", "default", "default", "rdp_dpt_p6kFHC_feb17HL2.root", 1, 0, 0, 0, 0, "recDpT", "trueDpT", "trueDpT", "recDalphaT", "trueDalphaT", "trueDalphaT", true, true, true)
treeConvert_paper("/data/t2k/dolan/MECProcessing/CC0Pi/feb17HL2/p6K_rdp_6kto12k_FHC_out.root", "default", "default", "rdp_dphit_p6kFHC_feb17HL2.root", 1, 0, 0, 0, 0, "recDphiT", "trueDphiT", "trueDphiT", "recDalphaT", "trueDalphaT", "trueDalphaT", true, true, true)
treeConvert_paper("/data/t2k/dolan/MECProcessing/CC0Pi/feb17HL2/p6K_rdp_6kto12k_FHC_out.root", "default", "default", "rdp_dat_p6kFHC_feb17HL2.root", 1, 0, 0, 0, 0, "recDalphaT", "trueDalphaT", "trueDalphaT", "recDpT", "trueDpT", "trueDpT", true, true, true)
.q
EOF
sleep 10
cd ../
root <<EOF
.L plotReco_paper.cc+
plotReco("./inputs/NeutAirV2_dpt_allStats.root", "./inputs/rdp_dpt_p6kFHC_feb17HL2.root", "dptPlotRecoOut_fine.root", "dpt", 0.2013)
plotReco("./inputs/NeutAirV2_dphit_allStats.root", "./inputs/rdp_dphit_p6kFHC_feb17HL2.root", "dphitPlotRecoOut_fine.root", "dphit", 0.2013)
plotReco("./inputs/NeutAirV2_dat_allStats.root", "./inputs/rdp_dat_p6kFHC_feb17HL2.root", "datPlotRecoOut_fine.root", "dat", 0.2013)
.q
EOF