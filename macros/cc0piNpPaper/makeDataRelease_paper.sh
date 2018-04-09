#!/bin/sh
for var in dptResults datResults dphitResults; do
  cd $var
root -b -l quickFtX_xsecOut.root <<EOF
TH1D* result = new TH1D();
TMatrixD* cormat = new TMatrixD()
TMatrixD* covmat = new TMatrixD()
result  = (TH1D*) _file0->Get("dif_xSecFit_allError");
cormat  = (TMatrixD*) _file0->Get("cormatrix");
covmat  = (TMatrixD*) _file0->Get("covarXsec");
TH2D* corhist = new TH2D(*cormat);
TH2D* covhist = new TH2D(*covmat);
covhist->Scale(1E76)
TFile* outFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/summaryPlots/dataRelease/${var}.root", "RECREATE");
result->Write("Result");
corhist->Write("Correlation_Matrix");
covhist->Write("Covariance_Matrix");
EOF
  cd ..
done;
for var in dptResults datResults dphitResults; do
  if [[ "${var}" == "dptResults" ]]
  then
    cd /data/t2k/dolan/fitting/feb17_refit/dptv5/rdp/noreg
  fi
  if [[ "${var}" == "datResults" ]]
  then
    cd /data/t2k/dolan/fitting/feb17_refit/datv5/rdp/noreg
  fi
  if [[ "${var}" == "dphitResults" ]]
  then
    cd /data/t2k/dolan/fitting/feb17_refit/dphitv5/rdp/noreg
  fi
root -b -l quickFtX_xsecOut.root <<EOF
TH1D* result = new TH1D();
TMatrixD* cormat = new TMatrixD()
TMatrixD* covmat = new TMatrixD()
result  = (TH1D*) _file0->Get("dif_xSecFit_allError");
cormat  = (TMatrixD*) _file0->Get("cormatrix");
covmat  = (TMatrixD*) _file0->Get("covarXsec");
TH2D* corhist = new TH2D(*cormat);
TH2D* covhist = new TH2D(*covmat);
covhist->Scale(1E76)
TFile* outFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/summaryPlots/dataRelease/${var}_noreg.root", "RECREATE");
result->Write("Result");
corhist->Write("Correlation_Matrix");
covhist->Write("Covariance_Matrix");
EOF
  cd -
done;