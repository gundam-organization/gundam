#!/bin/sh
source /home/dolan/.bashrc
#runxstoolmar16v1r5
time
mt=$1
wd="."
mkdir splineLog
#runnd280
runxstoolmar17_v2r4
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NXSec_MaCCQE\"       ,1.00,0.14)"   > splineLog/NXSec_MaCCQE.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2014a_pF_C12\"   ,1.00,0.15)"   > splineLog/NIWG2014a_pF_C12.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWGMEC_Norm_C12\"   ,1.00,0.73)"   > splineLog/NIWGMEC_Norm_C12.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NXSec_CA5RES\"       ,1.00,0.12)"   > splineLog/NXSec_CA5RES.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NXSec_MaNFFRES\"     ,1.00,0.16)"   > splineLog/NXSec_MaNFFRES.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NXSec_BgSclRES\"     ,1.00,0.15)"   > splineLog/NXSec_BgSclRES.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2012a_ccnueE0\"  ,1.00,0.02)"   > splineLog/NIWG2012a_ccnueE0.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2012a_dismpishp\",1.00,0.40)"   > splineLog/NIWG2012a_dismpishp.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2012a_cccohE0\"  ,1.00,1.00)"   > splineLog/NIWG2012a_cccohE0.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2012a_nccohE0\"  ,1.00,0.30)"   > splineLog/NIWG2012a_nccohE0.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2012a_ncotherE0\",1.00,0.30)"   > splineLog/NIWG2012a_ncotherE0.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2014a_Eb_C12\"   ,1.00,0.36)"   > splineLog/NIWG2014a_Eb_C12.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2014a_Eb_Pb208\" ,1.00,0.2045)" > splineLog/NIWG2014a_Eb_Pb208.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2014a_pF_Pb208\" ,1.00,0.1000)" > splineLog/NIWG2014a_pF_Pb208.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2014a_Eb_Fe56\"  ,1.00,0.2727)" > splineLog/NIWG2014a_Eb_Fe56.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2014a_pF_Fe56\"  ,1.00,0.1000)" > splineLog/NIWG2014a_pF_Fe56.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2014a_Eb_O16\"   ,1.00,0.45)"   > splineLog/NIWG2014a_Eb_O16.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWG2014a_pF_O16\"   ,1.00,0.45)"   > splineLog/NIWG2014a_pF_O16.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NIWGMEC_Norm_O12\"   ,1.00,0.45)"   > splineLog/NIWGMEC_Norm_O12.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NCasc_FrInelHigh_pi\",1.00,0.33)"   > splineLog/NCasc_FrInelHigh_pi.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NCasc_FrInelLow_pi\" ,1.00,0.41)"   > splineLog/NCasc_FrInelLow_pi.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NCasc_FrPiProd_pi\"  ,1.00,0.50)"   > splineLog/NCasc_FrPiProd_pi.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NCasc_FrAbs_pi\"     ,1.00,0.41)"   > splineLog/NCasc_FrAbs_pi.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NCasc_FrCExLow_pi\"  ,1.00,0.57)"   > splineLog/NCasc_FrCExLow_pi.log
root -b -q -l "genResponse_eat.C(\"${mt}\",\"${wd}\",\"NCasc_FrCExHigh_pi\" ,1.00,0.28)"   > splineLog/NCasc_FrCExHigh_pi.log
time