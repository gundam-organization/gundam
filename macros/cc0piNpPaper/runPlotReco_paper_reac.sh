#!/bin/sh
#First argument is a boolian specifying whether fine binning is to be used
#Second argument is a boolian specifying whether to show the postfit result (only for SB with coarse binning)
cd /data/t2k/dolan/paperStuff/recoPlots/reac
root <<EOF
.L plotReco_paper_reac.cc+
plotReco("../inputs/NeutAirV2_dpt_allStats.root", "../inputs/rdp_dpt_p6kFHC_feb17HL2.root", "dptPlotRecoOut_fine.root", "dpt", 0.2013, $1, true, $2)
plotReco("../inputs/NeutAirV2_dphit_allStats.root", "../inputs/rdp_dphit_p6kFHC_feb17HL2.root", "dphitPlotRecoOut_fine.root", "dphit", 0.2013, $1, true, $2)
plotReco("../inputs/NeutAirV2_dat_allStats.root", "../inputs/rdp_dat_p6kFHC_feb17HL2.root", "datPlotRecoOut_fine.root", "dat", 0.2013, $1, true, $2)
.q
EOF