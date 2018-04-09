#!/bin/sh
time
var=$1
workdir=$2

varToFit=var

cd ${workdir};


echo "Starting plotResults.sh"
echo "Variable is"
echo ${var}

# sub dirs: 1reg 3_1reg 5_1reg 9_1reg noreg
subdirs="1reg 3_1reg 5_1reg 9_1reg noreg" 

# Compare results with GENIE and NEUT priors
for dir in ${subdirs}; do 
root -b -l rdp/${dir}/quickFtX_xsecOut.root rdp_geniePrior/${dir}/quickFtX_xsecOut.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* neutHist     =(TH1D*)     _file0->Get("dif_xSecFit_allError");
TMatrixD* neutMat  =(TMatrixD*) _file0->Get("covarXsec");
TH1D* genieHist    =(TH1D*)     _file1->Get("dif_xSecFit_allError");
TMatrixD* genieMat =(TMatrixD*) _file1->Get("covarXsec");
.L ~/calcChi2.cc
double neutchi2  = calcChi2(neutHist, genieHist, *neutMat);
double geniechi2 = calcChi2(neutHist, genieHist, *genieMat);
neutHist->SetLineColor(kRed);
neutHist->SetMarkerColor(kRed);
genieHist->SetLineColor(kBlue);
genieHist->SetMarkerColor(kBlue);
TLegend* leg = new TLegend(0.4,0.6,0.85,0.85);
if("${var}"=="dat") {leg = new TLegend(0.2,0.6,0.75,0.85);}
leg->AddEntry(neutHist, Form("NEUT inp., #chi^{2}=%.2f",neutchi2),"lep");
leg->AddEntry(genieHist, Form("Genie inp., #chi^{2}=%.2f",geniechi2),"lep");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
genieHist->Draw();
neutHist->Draw("same");
genieHist->Draw("same");
leg->Draw("same");
c1->SaveAs("priorComp_${dir}.png")
c1->SaveAs("priorComp_${dir}.root")
EOF
done;

# Compare results with GENIE and NEUT priors
for dir in ${subdirs}; do 
root -b -l rdp/${dir}/quickFtX_xsecOut.root rdp_geniePrior_noSB/${dir}/quickFtX_xsecOut.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* neutHist     =(TH1D*)     _file0->Get("dif_xSecFit_allError");
TMatrixD* neutMat  =(TMatrixD*) _file0->Get("covarXsec");
TH1D* genieHist    =(TH1D*)     _file1->Get("dif_xSecFit_allError");
TMatrixD* genieMat =(TMatrixD*) _file1->Get("covarXsec");
.L ~/calcChi2.cc
double neutchi2  = calcChi2(neutHist, genieHist, *neutMat);
double geniechi2 = calcChi2(neutHist, genieHist, *genieMat);
neutHist->SetLineColor(kRed);
neutHist->SetMarkerColor(kRed);
genieHist->SetLineColor(kBlue);
genieHist->SetMarkerColor(kBlue);
TLegend* leg = new TLegend(0.4,0.6,0.85,0.85);
if("${var}"=="dat") {leg = new TLegend(0.2,0.6,0.75,0.85);}
leg->AddEntry(neutHist, Form("NEUT inp., #chi^{2}=%.2f",neutchi2),"lep");
leg->AddEntry(genieHist, Form("Genie inp., #chi^{2}=%.2f",geniechi2),"lep");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
genieHist->Draw();
neutHist->Draw("same");
genieHist->Draw("same");
leg->Draw("same");
c1->SaveAs("priorComp_genieNoSB_${dir}.png")
c1->SaveAs("priorComp_genieNoSB_${dir}.root")
EOF
done

#L curve:

echo "Running L curve generation"
for dir in asimov genie neutNo2p2h nuWro neutFD neutBask rdp rdp_geniePrior; do 
  cd ${dir};
  mkdir lCurve;
  cd lCurve;
  echo "Building l curve for ${dir}"
  for subdir in $(ls ../); do 
    if [[ "${subdir}" != "noreg" ]]
    then
      cp ../${subdir}/quickFtX_fitOut.root ./${subdir}.root
    fi
  done;
  ln -s /data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/macros/buildLcurve.C .
root -b -l <<EOF
.L buildLcurve.C
buildLcurve();
.q
EOF
  cd ${workdir};
done;

# plot post and prefit params
echo "Plotting pre and post fit parameter comparison"
for dir in asimov genie neutNo2p2h nuWro neutFD neutBask rdp rdp_geniePrior; do 
  cd ${dir};
  #for subdir in $(ls); do 
  for subdir in ${subdirs}; do 
    cd ${subdir}
root -b -l quickFtX_propErrOut.root ${workdir}/../${var}PreFit_apr17.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* h1 = new TH1D();
c1 = (TCanvas*) _file1->Get("Canvas_1");
h1 = (TH1D*) _file0->Get("paramResultHisto_red");
c1->Draw()
h1->Draw("same");
c1->SaveAs("paramResults.png")
c1->SaveAs("paramResults.root")
EOF
    cd ../
  done;
  cd ${workdir}
done;

# plot postfit params cor mat
echo "Plotting post fit parameter cor matrix"
for dir in asimov genie neutNo2p2h nuWro neutFD neutBask rdp rdp_geniePrior; do 
  cd ${dir};
  #for subdir in $(ls); do 
  for subdir in ${subdirs}; do 
    cd ${subdir}
root -b -l quickFtX_fitOut.root <<EOF
.L /home/dolan/loadPalettes.C
SetUserPalette(2);
TCanvas* c1 = new TCanvas();
c1->cd();
TH2D* corMat = (TH2D*) _file0->Get("res_cor_matrix"); 
TH2D* corMat_red =  new TH2D("corMat_red", "corMat_red", 89, 0, 89, 89, 0, 89);
corMat->Draw("colz");
for(int i=1; i<90; i++){ 
  for(int j=1; j<90; j++){ 
    //cout << i << ", " << j << " " << corMat->GetBinContent(i,j) << endl;
    //corMat_red->SetBinContent(i, j, corMat->GetBinContent(i,j));
    if(i<83 && j<83) corMat_red->SetBinContent(i, j, TMatrixDBase->GetBinContent(i,j));
    if(i>=83 && j>=83) corMat_red->SetBinContent(i, j, TMatrixDBase->GetBinContent(i+3,j+3));
    if(i<83 && j>=83) corMat_red->SetBinContent(i, j, TMatrixDBase->GetBinContent(i,j+3));
    if(i>=83 && j<83) corMat_red->SetBinContent(i, j, TMatrixDBase->GetBinContent(i+3,j));

  }
}
corMat_red->Draw("colz");
corMat_red->GetZaxis()->SetRangeUser(-1,1);
corMat_red->Draw("colz");
c1->SaveAs("postFitParamCorMat.png");
c1->SaveAs("postFitParamCorMat.root");
EOF
    cd ../
  done;
  cd ${workdir}
done;

# plot xsec cor mat
echo "Plotting xsec cor matrix"
for dir in asimov genie neutNo2p2h nuWro neutFD neutBask rdp rdp_geniePrior; do 
  cd ${dir};
  #for subdir in $(ls); do 
  for subdir in ${subdirs}; do 
    cd ${subdir}
root -b -l quickFtX_xsecOut.root <<EOF
.L /home/dolan/loadPalettes.C
SetUserPalette(2)
TCanvas* c1 = new TCanvas()
c1->cd()
TH2D* corMat = (TH2D*) _file0->Get("cormatrix") 
corMat->Draw("colz")
TMatrixDBase->GetZaxis()->SetRangeUser(-1,1)
TMatrixDBase->Draw("colz")
c1->SaveAs("xsecCorMat.png")
c1->SaveAs("xsecCorMat.root")
EOF
    cd ../
  done;
  cd ${workdir}
done;

# plot extra syst cov mat
echo "Plotting extra syst cov matrix"
for dir in asimov genie neutNo2p2h nuWro neutFD neutBask rdp rdp_geniePrior; do 
  cd ${dir};
  #for subdir in $(ls); do 
  for subdir in ${subdirs}; do 
    cd ${subdir}
root -b -l quickFtX_xsecOut.root <<EOF
TCanvas* c1 = new TCanvas()
c1->cd()
TGaxis::SetMaxDigits(3);
TH2D* corMat = (TH2D*) _file0->Get("${var}CovarCombExtraSyst") 
corMat->Draw("colz")
c1->SaveAs("extraSystCorMat.png")
c1->SaveAs("extraSystCorMat.root")
EOF
    cd ../
  done;
  cd ${workdir}
done;

# plot post and prefit params
echo "Plotting pre and post fit bin comparison"
for dir in asimov genie neutNo2p2h nuWro neutFD neutBask rdp rdp_geniePrior; do 
  cd ${dir};
  #for subdir in $(ls); do 
  for subdir in ${subdirs}; do 
    cd ${subdir}
    cd plotReco
    echo "In dir: "
    pwd
root -b -l plotRecoOut_postfit.root plotRecoOut_prefit.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* h1 = new TH1D();
TH1D* prefith = new TH1D();
TH1D* data = new TH1D();
h1 =  (TH1D*) _file0->Get("allSelEvtsMC_globBins");
prefith =  (TH1D*) _file1->Get("allSelEvtsMC_globBins");
data =  (TH1D*) _file1->Get("allSelEvtsMC_globBins_Data");
postfitChi2Hist =  (TH1D*) _file0->Get("allSelEvtsMC_globBins_Chi2");
prefitChi2Hist  =  (TH1D*) _file1->Get("allSelEvtsMC_globBins_Chi2");
postfitChi2 = postfitChi2Hist->Integral();
prefitChi2  = prefitChi2Hist->Integral();
h1->SetMarkerStyle(0)
prefith->SetMarkerStyle(0)
h1->SetLineColor(kBlue)
prefith->SetLineColor(kRed)
leg = new TLegend(0.4,0.65,0.7,0.85);
leg->AddEntry(data,"Data","lep");
leg->AddEntry(prefith,Form("Pre-fit, #chi^{2}=%.2f",prefitChi2),"l");
leg->AddEntry(h1,Form("Post-fit, #chi^{2}=%.2f",postfitChi2),"l");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
c1 = (TCanvas*) _file1->Get("canv_globalBins");
c1->cd();
c1->Draw();
h1->Draw("samehist");
leg->Draw("same");
c1->SaveAs("postPreFitComp.png")
c1->SaveAs("postPreFitComp.root")
EOF
    cp postPreFitComp.png ${workdir}/${dir}/${subdir}/postPreFitComp.png
    cd ${workdir}/${dir}
  done;
  cd ${workdir}
done;

# plot dif reg strengths
echo "Plotting xsec for dif reg strengths"
for dir in neutFD genie neutNo2p2h nuWro neutBask rdp rdp_geniePrior; do 
  cd ${dir};
root -b -l noreg/quickFtX_xsecOut.root 1reg/quickFtX_xsecOut.root 3_1reg/quickFtX_xsecOut.root 5_1reg/quickFtX_xsecOut.root 7_1reg/quickFtX_xsecOut.root 9_1reg/quickFtX_xsecOut.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* fakeData = (TH1D*) _file0->Get("Fake Data") 
TH1D* inputMC = (TH1D*) _file0->Get("Monte Carlo") 
TH1D* xsecNoReg = (TH1D*) _file0->Get("dif_xSecFit_allError") 
TH1D* xsec1Reg = (TH1D*) _file1->Get("dif_xSecFit_allError") 
TH1D* xsec3Reg = (TH1D*) _file2->Get("dif_xSecFit_allError") 
TH1D* xsec5Reg = (TH1D*) _file3->Get("dif_xSecFit_allError") 
TH1D* xsec7Reg = (TH1D*) _file4->Get("dif_xSecFit_allError") 
TH1D* xsec9Reg = (TH1D*) _file5->Get("dif_xSecFit_allError") 
xsecNoReg->SetLineColor(kBlack);
xsec1Reg->SetLineColor(kRed-3);
xsec3Reg->SetLineColor(kViolet-4);
xsec5Reg->SetLineColor(kBlue-4);
xsec7Reg->SetLineColor(kGreen+2);
xsec9Reg->SetLineColor(kCyan+2);
xsec1Reg->SetMarkerColor(kRed-3);
xsec3Reg->SetMarkerColor(kViolet-4);
xsec5Reg->SetMarkerColor(kBlue-4);
xsec7Reg->SetMarkerColor(kGreen+2);
xsec9Reg->SetMarkerColor(kCyan+2);
fakeData->SetLineColor(kGreen+2);
inputMC->SetLineColor(kBlue);
fakeData->SetLineStyle(2);
inputMC->SetLineStyle(2);
fakeData->SetMarkerStyle(1);
inputMC->SetMarkerStyle(1);
xsecNoReg->SetMarkerStyle(28);
xsec1Reg->SetMarkerStyle(24);
xsec3Reg->SetMarkerStyle(25);
xsec5Reg->SetMarkerStyle(26);
xsec7Reg->SetMarkerStyle(27);
xsec9Reg->SetMarkerStyle(32);
leg = new TLegend(0.4,0.3,0.85,0.85);
if("${var}"=="dat") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->AddEntry(xsecNoReg,"Result p_{reg}=0","lep");
leg->AddEntry(xsec1Reg, "Result p_{reg}=1","lep");
leg->AddEntry(xsec3Reg, "Result p_{reg}=3","lep");
leg->AddEntry(xsec5Reg, "Result p_{reg}=5","lep");
leg->AddEntry(xsec7Reg, "Result p_{reg}=7","lep");
leg->AddEntry(xsec9Reg, "Result p_{reg}=9","lep");
if("${dir}"!="rdp") leg->AddEntry(fakeData, "Fake Data","lf");
leg->AddEntry(inputMC,  "Input Sim.","lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
inputMC->Draw();
if("${dir}"!="rdp") fakeData->Draw("same");
xsecNoReg->Draw("same");
xsec1Reg->Draw("same");
xsec3Reg->Draw("same");
xsec5Reg->Draw("same");
xsec7Reg->Draw("same");
xsec9Reg->Draw("same");
leg->Draw("same");
c1->SaveAs("xsecRegComp.png")
c1->SaveAs("xsecRegComp.root")
EOF
  cd ${workdir}
done;

echo "Plotting chi2 for dif reg strengths"
for dir in neutFD genie neutNo2p2h nuWro neutBask rdp rdp_geniePrior; do 
  cd ${dir};
root -b -l noreg/quickFtX_xsecOut.root 1reg/quickFtX_xsecOut.root 3_1reg/quickFtX_xsecOut.root 5_1reg/quickFtX_xsecOut.root 7_1reg/quickFtX_xsecOut.root 9_1reg/quickFtX_xsecOut.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* xsecNoReg =(TH1D*) _file0->Get("chi2Hist_realdata");
TH1D* xsec1Reg = (TH1D*) _file1->Get("chi2Hist_realdata");
TH1D* xsec3Reg = (TH1D*) _file2->Get("chi2Hist_realdata");
TH1D* xsec5Reg = (TH1D*) _file3->Get("chi2Hist_realdata");
TH1D* xsec7Reg = (TH1D*) _file4->Get("chi2Hist_realdata");
TH1D* xsec9Reg = (TH1D*) _file5->Get("chi2Hist_realdata");
if("${var}"=="dat" || "${var}"=="dphit"){
  xsecNoReg =(TH1D*) _file0->Get("chi2Hist_realdata_${var}");
  xsec1Reg = (TH1D*) _file1->Get("chi2Hist_realdata_${var}");
  xsec3Reg = (TH1D*) _file2->Get("chi2Hist_realdata_${var}");
  xsec5Reg = (TH1D*) _file3->Get("chi2Hist_realdata_${var}");
  xsec7Reg = (TH1D*) _file4->Get("chi2Hist_realdata_${var}");
  xsec9Reg = (TH1D*) _file5->Get("chi2Hist_realdata_${var}");
}
xsecNoReg->SetLineColor(kBlack);
xsec1Reg->SetLineColor(kRed-3);
xsec3Reg->SetLineColor(kViolet-4);
xsec5Reg->SetLineColor(kBlue-4);
xsec7Reg->SetLineColor(kGreen+2);
xsec9Reg->SetLineColor(kCyan+2);
TLegend* leg;
if("${dir}"=="neutFD" || "${dir}"=="genie" || "${dir}"=="rdp") {leg = new TLegend(0.2,0.6,0.45,0.85);} else leg = new TLegend(0.5,0.6,0.85,0.85);
leg->AddEntry(xsecNoReg,"p_{reg}=0","lf");
leg->AddEntry(xsec1Reg, "p_{reg}=1","lf");
leg->AddEntry(xsec3Reg, "p_{reg}=3","lf");
leg->AddEntry(xsec5Reg, "p_{reg}=5","lf");
leg->AddEntry(xsec7Reg, "p_{reg}=7","lf");
leg->AddEntry(xsec9Reg, "p_{reg}=9","lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
xsecNoReg->GetYaxis()->SetRangeUser(0,250);
if("${dir}"=="genie") xsecNoReg->GetYaxis()->SetRangeUser(0,500);
if("${dir}"=="neutFD") xsecNoReg->GetYaxis()->SetRangeUser(0,500);
xsecNoReg->Draw();
xsec1Reg->Draw("same");
xsec3Reg->Draw("same");
xsec5Reg->Draw("same");
xsec7Reg->Draw("same");
xsec9Reg->Draw("same");
leg->Draw("same");
c1->SaveAs("chi2RegComp.png")
c1->SaveAs("chi2RegComp.root")
EOF
  cd ${workdir}
done;

# plot SB comp
echo "Plotting xsec with and WO SB"
for dir in neutFD genie neutNo2p2h nuWro neutBask rdp rdp_geniePrior; do 
root -b -l ${dir}/3_1reg/quickFtX_xsecOut.root ${dir}_noSB/3_1reg/quickFtX_xsecOut.root ${dir}/5_1reg/quickFtX_xsecOut.root ${dir}_noSB/5_1reg/quickFtX_xsecOut.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* fakeData = (TH1D*) _file0->Get("Fake Data") 
TH1D* inputMC = (TH1D*) _file0->Get("Monte Carlo") 
TH1D* xsec3Reg = (TH1D*) _file0->Get("dif_xSecFit_allError") 
TH1D* xsec3Reg_noSB = (TH1D*) _file1->Get("dif_xSecFit_allError") 
TH1D* xsec5Reg = (TH1D*) _file2->Get("dif_xSecFit_allError") 
TH1D* xsec5Reg_noSB = (TH1D*) _file3->Get("dif_xSecFit_allError") 
xsec3Reg->SetLineColor(kRed-3);
xsec3Reg_noSB->SetLineColor(kRed-3);
xsec5Reg->SetLineColor(kViolet-4);
xsec5Reg_noSB->SetLineColor(kViolet-4);
xsec3Reg->SetMarkerColor(kRed-3);
xsec3Reg_noSB->SetMarkerColor(kRed-3);
xsec5Reg->SetMarkerColor(kViolet-4);
xsec5Reg_noSB->SetMarkerColor(kViolet-4);
fakeData->SetLineColor(kGreen+2);
inputMC->SetLineColor(kBlue);
fakeData->SetLineStyle(2);
inputMC->SetLineStyle(2);
xsec3Reg_noSB->SetLineStyle(3);
xsec5Reg_noSB->SetLineStyle(3);
fakeData->SetMarkerStyle(1);
inputMC->SetMarkerStyle(1);
xsec3Reg->SetMarkerStyle(28);
xsec3Reg_noSB->SetMarkerStyle(24);
xsec5Reg->SetMarkerStyle(25);
xsec5Reg_noSB->SetMarkerStyle(26);
leg = new TLegend(0.4,0.3,0.85,0.85);
if("${var}"=="dat") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->AddEntry(xsec3Reg,      "Result p_{reg}=3","lep");
leg->AddEntry(xsec3Reg_noSB, "Result p_{reg}=3, no sidebands","lep");
leg->AddEntry(xsec5Reg,      "Result p_{reg}=5","lep");
leg->AddEntry(xsec5Reg_noSB, "Result p_{reg}=5, no sidebands","lep");
if("${dir}"!="rdp") leg->AddEntry(fakeData, "Fake Data","lf");
leg->AddEntry(inputMC,  "Input Sim.","lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
inputMC->Draw();
if("${dir}"!="rdp") fakeData->Draw("same");
xsec3Reg->Draw("same");
xsec3Reg_noSB->Draw("same");
xsec5Reg->Draw("same");
xsec5Reg_noSB->Draw("same");
leg->Draw("same");
c1->SaveAs("${dir}/xsecSBComp.png")
c1->SaveAs("${dir}/xsecSBComp.root")
EOF
done;

echo "Plotting chi2 with and WO SB"
for dir in neutFD genie neutNo2p2h nuWro neutBask rdp rdp_geniePrior; do 
root -b -l ${dir}/3_1reg/quickFtX_xsecOut.root ${dir}_noSB/3_1reg/quickFtX_xsecOut.root ${dir}/5_1reg/quickFtX_xsecOut.root ${dir}_noSB/5_1reg/quickFtX_xsecOut.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* xsec3Reg      =(TH1D*) _file0->Get("chi2Hist_realdata");
TH1D* xsec3Reg_noSB = (TH1D*) _file1->Get("chi2Hist_realdata");
TH1D* xsec5Reg      = (TH1D*) _file2->Get("chi2Hist_realdata");
TH1D* xsec5Reg_noSB = (TH1D*) _file3->Get("chi2Hist_realdata");
if("${var}"=="dat" || "${var}"=="dphit"){
  xsec3Reg      = (TH1D*) _file0->Get("chi2Hist_realdata_${var}");
  xsec3Reg_noSB = (TH1D*) _file1->Get("chi2Hist_realdata_${var}");
  xsec5Reg      = (TH1D*) _file2->Get("chi2Hist_realdata_${var}");
  xsec5Reg_noSB = (TH1D*) _file3->Get("chi2Hist_realdata_${var}");
}
xsec3Reg->SetLineColor(kRed-3);
xsec3Reg_noSB->SetLineColor(kViolet-4);
xsec5Reg->SetLineColor(kGreen+2);
xsec5Reg_noSB->SetLineColor(kCyan+2);
xsec3Reg->SetMarkerStyle(1);
xsec3Reg_noSB->SetMarkerStyle(1);
xsec5Reg->SetMarkerStyle(1);
xsec5Reg_noSB->SetMarkerStyle(1);
TLegend* leg;
if("${dir}"=="neutFD" || "${dir}"=="genie" || "${dir}"=="rdp") {leg = new TLegend(0.2,0.6,0.45,0.85);} else leg = new TLegend(0.5,0.6,0.85,0.85);
leg->AddEntry(xsec3Reg,      "p_{reg}=3","lf");
leg->AddEntry(xsec3Reg_noSB, "p_{reg}=3, no SB","lf");
leg->AddEntry(xsec5Reg,      "p_{reg}=5","lf");
leg->AddEntry(xsec5Reg_noSB, "p_{reg}=5, no SB","lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
xsec3Reg->GetYaxis()->SetRangeUser(0,250);
if("${dir}"=="genie") xsec3Reg->GetYaxis()->SetRangeUser(0,500);
if("${dir}"=="neutFD") xsec3Reg->GetYaxis()->SetRangeUser(0,500);
xsec3Reg->Draw();
xsec3Reg_noSB->Draw("same");
xsec5Reg->Draw("same");
xsec5Reg_noSB->Draw("same");
leg->Draw("same");
c1->SaveAs("${dir}/chi2SBComp.png")
c1->SaveAs("${dir}/chi2SBComp.root")
EOF
done;

echo "Overlay chi2 for each FD study"
for dir in 3_1reg 5_1reg; do 
#for dir in neutFD genie neutNo2p2h nuWro neutBask; do 
root -b -l neutFD/${dir}/quickFtX_xsecOut.root genie/${dir}/quickFtX_xsecOut.root neutNo2p2h/${dir}/quickFtX_xsecOut.root nuWro/${dir}/quickFtX_xsecOut.root neutBask/${dir}/quickFtX_xsecOut.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* neutFD      =(TH1D*) _file0->Get("chi2Hist_realdata");
TH1D* genie = (TH1D*) _file1->Get("chi2Hist_realdata");
TH1D* neutNo2p2h      = (TH1D*) _file2->Get("chi2Hist_realdata");
TH1D* nuWro = (TH1D*) _file3->Get("chi2Hist_realdata");
TH1D* neutBask = (TH1D*) _file4->Get("chi2Hist_realdata");
if("${var}"=="dat" || "${var}"=="dphit"){
  neutFD      = (TH1D*) _file0->Get("chi2Hist_realdata_${var}");
  genie = (TH1D*) _file1->Get("chi2Hist_realdata_${var}");
  neutNo2p2h      = (TH1D*) _file2->Get("chi2Hist_realdata_${var}");
  nuWro = (TH1D*) _file3->Get("chi2Hist_realdata_${var}");
  neutBask = (TH1D*) _file4->Get("chi2Hist_realdata_${var}");
}
neutFD->SetLineColor(kBlack);
genie->SetLineColor(kRed-3);
neutNo2p2h->SetLineColor(kBlue-4);
nuWro->SetLineColor(kGreen+2);
neutBask->SetLineColor(kCyan+2);
genie->SetLineStyle(2);
neutNo2p2h->SetLineStyle(3);
nuWro->SetLineStyle(1);
neutBask->SetLineStyle(5); 
TLegend* leg = new TLegend(0.2,0.6,0.45,0.85);
leg->AddEntry(neutFD, "NEUT 6B stat fluct.","l");
leg->AddEntry(genie, "GENIE","l");
leg->AddEntry(neutBask, "NEUT 6D","l");
leg->AddEntry(neutNo2p2h, "NEUT 6D w/o 2p2h","l");
leg->AddEntry(nuWro, "NuWro","l");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
neutFD->GetYaxis()->SetRangeUser(0,500);
if("${var}"=="dat") neutFD->GetYaxis()->SetRangeUser(0,350);
neutFD->Draw();
genie->Draw("same");
neutNo2p2h->Draw("same");
nuWro->Draw("same");
neutBask->Draw("same");
leg->Draw("same");
c1->SaveAs("chi2FDComp_${dir}.png")
c1->SaveAs("chi2FDComp_${dir}.root")
EOF
done;

time
