#!/bin/sh
# Takes $PWD as an argument
#Do extended shape only generator comparrison
# SET CANVAS SIZE!
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l gibuu_150317_out.root neut_540_lyon.root nuwro11_LFGRPA.root nuwro11_SF.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* gibuu = new TH1D();
TH1D* neut_sf = new TH1D();
TH1D* nuwro = new TH1D();
TH1D* nuwrosf = new TH1D();
result =  (TH1D*) _file0->Get("Result_shapeOnly");
gibuu =  (TH1D*) _file0->Get("nuisMC_shapeOnly");
neut_sf =  (TH1D*) _file1->Get("nuisMC_shapeOnly");
nuwro =  (TH1D*) _file2->Get("nuisMC_shapeOnly");
nuwrosf =  (TH1D*) _file3->Get("nuisMC_shapeOnly");
_file0->cd(); double gibuuChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file1->cd(); double neut_sfChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file2->cd(); double nuwroChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file3->cd(); double nuwrosfChi2 = chi2Hist_shapeOnly->GetBinContent(5);
if("${var}"=="dptOut") gibuu->SetYTitle("#frac{1}{#sigma} #frac{d#sigma}{d#deltap_{T}}");
if("${var}"=="datOut") gibuu->SetYTitle("#frac{1}{#sigma} #frac{d#sigma}{d#delta#alpha_{T}}");
if("${var}"=="dphitOut") gibuu->SetYTitle("#frac{1}{#sigma} #frac{d#sigma}{d#delta#phi_{T}}");
if("${var}"=="dptOut") gibuu->SetXTitle("#deltap_{T} (GeV)");
if("${var}"=="datOut") gibuu->SetXTitle("#delta#alpha_{T} (radians)");
if("${var}"=="dphitOut") gibuu->SetXTitle("#delta#phi_{T} (radians)");
gibuu->GetYaxis()->SetTitleSize(0.060);
gibuu->GetYaxis()->SetTitleOffset(1.15);
result->SetMarkerSize(1.0);
nuwrosf->SetLineColor(kAzure+8);
gibuu->SetLineColor(kAzure-2);
neut_sf->SetLineColor(kGray+2);
nuwro->SetLineColor(kRed+3);
nuwro->SetLineStyle(4);
nuwrosf->SetLineStyle(1);
gibuu->SetLineStyle(2);
neut_sf->SetLineStyle(1);
nuwrosf->SetLineWidth(4);
gibuu->SetLineWidth(3);
nuwro->SetLineWidth(2);
neut_sf->SetLineWidth(2);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->AddEntry(result,"T2K Fit to Data","ep");
leg->AddEntry(nuwrosf, Form("NuWro 11q SF+2p2h_{N}, #chi^{2}=%.1f",nuwrosfChi2), "lf");
leg->AddEntry(gibuu, Form("GiBUU 2016 (incl. 2p2h_{G}), #chi^{2}=%.1f",gibuuChi2), "lf");
leg->AddEntry(neut_sf, Form("NEUT 5.4.0 LFG_{N}+2p2h_{N}, #chi^{2}=%.1f",neut_sfChi2), "lf");
leg->AddEntry(nuwro, Form("NuWro 11q LFG+RPA+2p2h_{N}, #chi^{2}=%.1f",nuwroChi2), "lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
if("${var}"=="datOut") gibuu->GetYaxis()->SetRangeUser(0.0,0.29999);
gibuu->Draw();
nuwrosf->Draw("][same");
gibuu->Draw("][same");
neut_sf->Draw("][samehist");
nuwro->Draw("][same");
result->Draw("][same");
leg->Draw("][same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut_paper/${var}_GenCompExShapeOnly.png");
c1->SaveAs("${1}/plotsOut_paper/${var}_GenCompExShapeOnly.pdf");
c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_GenCompExShapeOnly.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") gibuu->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") gibuu->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") neut_sf->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") neut_sf->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") nuwro->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") nuwro->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") nuwrosf->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") nuwrosf->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(0.002,0.5);
gibuu->Draw();
nuwrosf->Draw("][same");
gibuu->Draw("][same");
neut_sf->Draw("][samehist");
nuwro->Draw("][same");
result->Draw("][same");
//leg->Draw("][same");
gPad->SetLogy();
c2->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_GenCompExShapeOnly.png");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_GenCompExShapeOnly.pdf");
//***********************************
// Add inlay to plots
//***********************************
c1->cd();
leg->Delete();
if("${var}"=="dptOut") gibuu->GetYaxis()->SetRangeUser(0.0,0.3999);
if("${var}"=="dphitOut") gibuu->GetYaxis()->SetRangeUser(0.0,0.34999);
inlayLeg = new TLegend(0.32,0.5,0.84,0.85);
inlayLeg->AddEntry(result,"T2K Fit to Data","ep");
inlayLeg->AddEntry(nuwrosf, Form("NuWro 11q SF+2p2h_{N}, #chi^{2}=%.1f",nuwrosfChi2), "lf");
inlayLeg->AddEntry(gibuu, Form("GiBUU 2016 (incl. 2p2h_{G}), #chi^{2}=%.1f",gibuuChi2), "lf");
inlayLeg->AddEntry(neut_sf, Form("NEUT 540 LFG+2p2h_{N}, #chi^{2}=%.1f",neut_sfChi2), "lf");
inlayLeg->AddEntry(nuwro, Form("NuWro 11q LFG+RPA+2p2h_{N}, #chi^{2}=%.1f",nuwroChi2), "lf");
inlayLeg->SetFillColor(kWhite);
inlayLeg->SetFillStyle(0);
if("${var}"=="dptOut" || "${var}"=="dphitOut"){
  TPad* inlay = new TPad("inlay","inlay",0.28,0.28,0.92,0.93);
  inlay->cd();
  TH1D* gibuu_clone = new TH1D(*gibuu);
  gibuu_clone->GetYaxis()->SetRangeUser(0.003,2.0);
  if("${var}"=="dphitOut") gibuu_clone->GetYaxis()->SetRangeUser(0.003,1.5);
  gibuu_clone->GetYaxis()->SetTitle("");
  gibuu_clone->GetXaxis()->SetTitle("");
  gibuu_clone->Draw();
  TH1D* result_clone = new TH1D(*result);
  result_clone->SetMarkerSize(0.5);
  nuwrosf->Draw("][same");
  gibuu_clone->Draw("][same");
  neut_sf->Draw("][samehist");
  nuwro->Draw("][same");
  result_clone->Draw("][same");
  inlayLeg->Draw("][same");
  gPad->SetLogy();
  inlay->SetFillStyle(0);
  inlay->Update();
  c1->cd();
  inlay->Draw();
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_GenCompExShapeOnly.png");
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_GenCompExShapeOnly.pdf");
  c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_GenCompExShapeOnly_inlay.root");
}
EOF
cd $1
done;
#Do extended shape only generator comparrison - normalisation
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l gibuu_150317_out.root  neut_540_lyon.root nuwro11_LFGRPA.root nuwro11_SF.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* gibuu = new TH1D();
TH1D* neut_sf = new TH1D();
TH1D* nuwro = new TH1D();
TH1D* nuwrosf = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
gibuu =  (TH1D*) _file0->Get("nuisMC");
neut_sf =  (TH1D*) _file1->Get("nuisMC");
nuwro =  (TH1D*) _file2->Get("nuisMC");
nuwrosf =  (TH1D*) _file3->Get("nuisMC");
_file0->cd(); double gibuuChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double neut_sfChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double nuwroChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double nuwrosfChi2 = chi2Hist->GetBinContent(5);
gibuu->SetLineColor(kRed-3);
neut_sf->SetLineColor(kViolet-4);
nuwro->SetLineColor(kGreen-6);
nuwrosf->SetLineColor(kGreen+2);
gibuu->SetLineStyle(1);
neut_sf->SetLineStyle(1);
neut_sf->SetLineWidth(3);
nuwro->SetLineStyle(1);
nuwrosf->SetLineStyle(2);
nuwrosf->SetLineWidth(3);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->AddEntry(result,"T2K Fit to Data","lep");
leg->AddEntry(neut_sf, Form("NEUT 540 LFG+2p2h_{N}, #chi^{2}=%.1f",neut_sfChi2), "lf");
leg->AddEntry(nuwro, Form("NuWro 11 LFG+RPA, #chi^{2}=%.1f",nuwroChi2), "lf");
leg->AddEntry(nuwrosf, Form("NuWro 11 SF, #chi^{2}=%.1f",nuwrosfChi2), "lf");
leg->AddEntry(gibuu, Form("GiBUU 2016, #chi^{2}=%.1f",gibuuChi2), "lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
gibuu->Draw();
nuwro->Draw("][same");
neut_sf->Draw("][samehist");
nuwrosf->Draw("][same");
result->Draw("][same");
leg->Draw("][same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut_paper/${var}_GenCompEx.png");
c1->SaveAs("${1}/plotsOut_paper/${var}_GenCompEx.pdf");
c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_GenCompEx.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") gibuu->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") gibuu->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") neut_sf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") neut_sf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nuwro->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") nuwro->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nuwrosf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") nuwrosf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") genie->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") genie->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
gibuu->Draw();
neut_sf->Draw("][same");
nuwro->Draw("][same");
nuwrosf->Draw("][samehist");
result->Draw("][same");
//leg->Draw("][same");
gPad->SetLogy();
c2->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_GenCompEx.png");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_GenCompEx.pdf");
//***********************************
// Add inlay to plots
//***********************************
c1->cd();
leg->Delete();
if("${var}"=="dptOut") gibuu->GetYaxis()->SetRangeUser(0.0,10E-39);
if("${var}"=="dphitOut") gibuu->GetYaxis()->SetRangeUser(0.0,5E-39);
inlayLeg = new TLegend(0.32,0.5,0.84,0.85);
inlayLeg->AddEntry(result,"T2K Fit to Data","lep");
inlayLeg->AddEntry(neut_sf, Form("NEUT 540 LFG+2p2h_{N}, #chi^{2}=%.1f",neut_sfChi2), "l");
inlayLeg->AddEntry(nuwro, Form("NuWro 11 LFG+RPA+2p2h_{N}, #chi^{2}=%.1f",nuwroChi2), "l");
inlayLeg->AddEntry(nuwrosf, Form("NuWro 11 SF+2p2h_{N}, #chi^{2}=%.1f",nuwrosfChi2), "l");
inlayLeg->AddEntry(gibuu, Form("GiBUU 2016 (incl. 2p2h_{G}), #chi^{2}=%.1f",gibuuChi2), "l");
inlayLeg->SetFillColor(kWhite);
inlayLeg->SetFillStyle(0);
if("${var}"=="dptOut" || "${var}"=="dphitOut"){
  TPad* inlay = new TPad("inlay","inlay",0.28,0.28,0.92,0.93);
  inlay->cd();
  TH1D* gibuu_clone = new TH1D(*gibuu);
  gibuu_clone->GetYaxis()->SetRangeUser(1E-40,30E-39);
  if("${var}"=="dphitOut") gibuu_clone->GetYaxis()->SetRangeUser(4E-41,8E-39);
  gibuu_clone->GetYaxis()->SetTitle("");
  gibuu_clone->GetXaxis()->SetTitle("");
  gibuu_clone->Draw();
  TH1D* result_clone = new TH1D(*result);
  result_clone->SetMarkerSize(0.5);
  nuwro->Draw("][same");
  neut_sf->Draw("][samehist");
  nuwrosf->Draw("][same");
  result_clone->Draw("][same");
  inlayLeg->Draw("][same");
  gPad->SetLogy();
  inlay->SetFillStyle(0);
  inlay->Update();
  c1->cd();
  inlay->Draw();
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_GenCompEx.png");
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_GenCompEx.pdf");
  c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_GenCompEx_inlay.root");
}
EOF
cd $1
done;
#Do FSI comparrison for SF (xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_SF_NoFSI_MA1p0RW_out.root neut5322_SF_MA1p0RW_out.root neut5322_SF_ExtraFSI_MA1p0RW_out.root neut5322_SF_MA1p0RW_no2p2hRW_out.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* nofsi = new TH1D();
TH1D* nomfsi = new TH1D();
TH1D* twofsi = new TH1D();
TH1D* no2p2h = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
nofsi =  (TH1D*) _file0->Get("nuisMC");
nomfsi =  (TH1D*) _file1->Get("nuisMC");
twofsi =  (TH1D*) _file2->Get("nuisMC");
no2p2h =  (TH1D*) _file3->Get("nuisMC");
_file0->cd(); double nofsiChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double nomfsiChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double twofsiChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double no2p2hChi2 = chi2Hist->GetBinContent(5);
if("${var}"=="dptOut") nofsi->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (cm^{2} Nucleon^{-1} GeV^{-1})");
if("${var}"=="datOut") nofsi->SetYTitle("#frac{d#sigma}{d#delta#alpha_{T}} (cm^{2} Nucleon^{-1} radian^{-1})");
if("${var}"=="dphitOut") nofsi->SetYTitle("#frac{d#sigma}{d#delta#phi_{T}} (cm^{2} Nucleon^{-1} radian^{-1})");
if("${var}"=="dptOut") nofsi->SetXTitle("#deltap_{T} (GeV)");
if("${var}"=="datOut") nofsi->SetXTitle("#delta#alpha_{T} (radians)");
if("${var}"=="dphitOut") nofsi->SetXTitle("#delta#phi_{T} (radians)");
nofsi->GetYaxis()->SetTitleSize(0.060);
nofsi->GetYaxis()->SetTitleOffset(1.15);
result->SetMarkerSize(1.0);
nomfsi->SetLineColor(kAzure+8);
no2p2h->SetLineColor(kAzure-2);
twofsi->SetLineColor(kGray+2);
nofsi->SetLineColor(kGray+3);
nomfsi->SetLineStyle(1);
twofsi->SetLineStyle(1);
no2p2h->SetLineStyle(2);
nofsi->SetLineStyle(3);
nomfsi->SetLineWidth(4);
twofsi->SetLineWidth(3);
no2p2h->SetLineWidth(2);
nofsi->SetLineWidth(2);
leg = new TLegend(0.35,0.3,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
//leg->SetHeader("NEUT, CH, Benhar SF, BBBA05");
leg->SetHeader("NEUT 5.3.2.2 SF");
leg->AddEntry(result,"T2K Fit to Data","ep");
leg->AddEntry(nomfsi,Form("w/ 2p2h_{N}, #chi^{2}=%.1f",nomfsiChi2),"lf");
leg->AddEntry(no2p2h,Form("w/o 2p2h, #chi^{2}=%.1f",no2p2hChi2),"lf");
leg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"lf");
leg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
nofsi->Draw();
nomfsi->Draw("][same");
no2p2h->Draw("][same");
twofsi->Draw("][same");
nofsi->Draw("][same");
result->Draw("][same")
leg->Draw("][same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut_paper/${var}_SF_FSIComp.png");
c1->SaveAs("${1}/plotsOut_paper/${var}_SF_FSIComp.pdf");
c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_SF_FSIComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nomfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nomfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") twofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") twofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") no2p2h->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") no2p2h->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
nofsi->Draw();
nomfsi->Draw("][same");
no2p2h->Draw("][same");
twofsi->Draw("][same");
nofsi->Draw("][same");
result->Draw("][same")
//leg->Draw("][same");
gPad->SetLogy();
c2->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_SF_FSIComp.png");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_SF_FSIComp.pdf");
//***********************************
// Add inlay to plots
//***********************************
c1->cd();
leg->Delete();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.0,10E-39);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(0.0,5E-39);
inlayLeg = new TLegend(0.32,0.5,0.84,0.85);
//inlayLeg->SetHeader("NEUT, CH, Benhar SF, BBBA05");
inlayLeg->SetHeader("NEUT 5.3.2.2 SF");
inlayLeg->AddEntry(result,"T2K Fit to Data","ep");
inlayLeg->AddEntry(nomfsi,Form("w/ 2p2h_{N}, #chi^{2}=%.1f",nomfsiChi2),"lf");
inlayLeg->AddEntry(no2p2h,Form("w/o 2p2h, #chi^{2}=%.1f",no2p2hChi2),"lf");
inlayLeg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"lf");
inlayLeg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"lf");
inlayLeg->SetFillColor(kWhite);
inlayLeg->SetFillStyle(0);
if("${var}"=="dptOut" || "${var}"=="dphitOut"){
  TPad* inlay = new TPad("inlay","inlay",0.28,0.28,0.92,0.93);
  inlay->cd();
  TH1D* nofsi_clone = new TH1D(*nofsi);
  nofsi_clone->GetYaxis()->SetRangeUser(4E-41,90E-39);
  if("${var}"=="dphitOut") nofsi_clone->GetYaxis()->SetRangeUser(4E-41,8E-39);
  nofsi_clone->GetYaxis()->SetTitle("");
  nofsi_clone->GetXaxis()->SetTitle("");
  TH1D* result_clone = new TH1D(*result);
  result_clone->SetMarkerSize(0.5);
  nofsi_clone->Draw();
  nomfsi->Draw("][same");
  no2p2h->Draw("][same");
  twofsi->Draw("][same");
  nofsi->Draw("][same");
  result_clone->Draw("][same");
  inlayLeg->Draw("][same");
  gPad->SetLogy();
  inlay->SetFillStyle(0);
  inlay->Update();
  c1->cd();
  inlay->Draw();
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_SF_FSIComp.png");
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_SF_FSIComp.pdf");
  c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_SF_FSIComp_inlay.root");
}
EOF
cd $1
done;
#Do FSI comparrison for LFG (xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut_540_default_noFSI.card.root neut_540_lyon.root neut_540_default_extraFSI.card.root neut_540_default_no2p2h.card.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* nofsi = new TH1D();
TH1D* nomfsi = new TH1D();
TH1D* twofsi = new TH1D();
TH1D* no2p2h = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
nofsi =  (TH1D*) _file0->Get("nuisMC");
nomfsi =  (TH1D*) _file1->Get("nuisMC");
twofsi =  (TH1D*) _file2->Get("nuisMC");
no2p2h =  (TH1D*) _file3->Get("nuisMC");
_file0->cd(); double nofsiChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double nomfsiChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double twofsiChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double no2p2hChi2 = chi2Hist->GetBinContent(5);
nofsi->SetLineColor(kBlue-4);
nomfsi->SetLineColor(kBlack);
twofsi->SetLineColor(kViolet-4);
no2p2h->SetLineColor(kRed-3);
nofsi->SetLineStyle(2);
nomfsi->SetLineStyle(1);
twofsi->SetLineStyle(1);
no2p2h->SetLineStyle(2);
leg = new TLegend(0.35,0.3,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
//leg->SetHeader("NEUT, CH, Benhar SF, BBBA05");
leg->SetHeader("NEUT 540 LFG+2p2h_{N}");
leg->AddEntry(result,"T2K Fit to Data","lep");
leg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"lf");
leg->AddEntry(nomfsi,Form("Nominal FSI, #chi^{2}=%.1f",nomfsiChi2),"lf");
leg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"lf");
leg->AddEntry(no2p2h,Form("Nom. FSI w/o 2p2h, #chi^{2}=%.1f",no2p2hChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
nofsi->Draw("HIST");
nofsi->Draw("][sameHIST");
nomfsi->Draw("][sameHIST");
twofsi->Draw("][sameHIST");
no2p2h->Draw("][sameHIST");
result->Draw("][same");
leg->Draw("][same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut_paper/${var}_LFG_FSIComp.png");
c1->SaveAs("${1}/plotsOut_paper/${var}_LFG_FSIComp.pdf");
c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_SF_FSIComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nomfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nomfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") twofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") twofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") no2p2h->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") no2p2h->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
nofsi->Draw("HIST");
nomfsi->Draw("][sameHIST");
twofsi->Draw("][sameHIST");
no2p2h->Draw("][sameHIST");
result->Draw("][same");
//leg->Draw("][same");
gPad->SetLogy();
c2->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_LFG_FSIComp.png");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_LFG_FSIComp.pdf");
//***********************************
// Add inlay to plots
//***********************************
c1->cd();
leg->Delete();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.0,10E-39);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(0.0,5E-39);
inlayLeg = new TLegend(0.32,0.5,0.84,0.85);
//inlayLeg->SetHeader("NEUT, CH, Benhar SF, BBBA05");
inlayLeg->SetHeader("NEUT 540 LFG+2p2h_{N}");
inlayLeg->AddEntry(result,"T2K Fit to Data","lep");
inlayLeg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"l");
inlayLeg->AddEntry(nomfsi,Form("Nominal FSI, #chi^{2}=%.1f",nomfsiChi2),"l");
inlayLeg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"l");
inlayLeg->AddEntry(no2p2h,Form("Nom. FSI w/o 2p2h, #chi^{2}=%.1f",no2p2hChi2),"l");
inlayLeg->SetFillColor(kWhite);
inlayLeg->SetFillStyle(0);
if("${var}"=="dptOut" || "${var}"=="dphitOut"){
  TPad* inlay = new TPad("inlay","inlay",0.28,0.28,0.92,0.93);
  inlay->cd();
  TH1D* nofsi_clone = new TH1D(*nofsi);
  nofsi_clone->GetYaxis()->SetRangeUser(4E-41,90E-39);
  if("${var}"=="dphitOut") nofsi_clone->GetYaxis()->SetRangeUser(4E-41,8E-39);
  nofsi_clone->GetYaxis()->SetTitle("");
  nofsi_clone->GetXaxis()->SetTitle("");
  TH1D* result_clone = new TH1D(*result);
  result_clone->SetMarkerSize(0.5);
  nofsi_clone->Draw("HIST");
  nofsi_clone->Draw("][sameHIST");
  nomfsi->Draw("][sameHIST");
  twofsi->Draw("][sameHIST");
  no2p2h->Draw("][sameHIST");
  result_clone->Draw("][same");
  inlayLeg->Draw("][same");
  gPad->SetLogy();
  inlay->SetFillStyle(0);
  inlay->Update();
  c1->cd();
  inlay->Draw();
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_LFG_FSIComp.png");
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_LFG_FSIComp.pdf");
  c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}LFGF_FSIComp_inlay.root");
}
EOF
cd $1
done;
#Do NuWro model comp (full xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l nuwro11_RFGRPA.root nuwro11_LFGRPA.root nuwro11_SF.root nuwro11_SF_no2p2h.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
TH1D* rfgrpa = new TH1D();
TH1D* lfg = new TH1D();
TH1D* sf = new TH1D();
TH1D* sfno2p2h = new TH1D();
rfgrpa =  (TH1D*) _file0->Get("nuisMC");
lfg =  (TH1D*) _file1->Get("nuisMC");
sf =  (TH1D*) _file2->Get("nuisMC");
sfno2p2h =  (TH1D*) _file3->Get("nuisMC");
_file0->cd(); double rfgrpaChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double lfgChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double sfChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double sfno2p2hChi2 = chi2Hist->GetBinContent(5);
if("${var}"=="dptOut") rfgrpa->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (cm^{2} Nucleon^{-1} GeV^{-1})");
if("${var}"=="datOut") rfgrpa->SetYTitle("#frac{d#sigma}{d#delta#alpha_{T}} (cm^{2} Nucleon^{-1} radian^{-1})");
if("${var}"=="dphitOut") rfgrpa->SetYTitle("#frac{d#sigma}{d#delta#phi_{T}} (cm^{2} Nucleon^{-1} radian^{-1})");
if("${var}"=="dptOut") rfgrpa->SetXTitle("#deltap_{T} (GeV)");
if("${var}"=="datOut") rfgrpa->SetXTitle("#delta#alpha_{T} (radians)");
if("${var}"=="dphitOut") rfgrpa->SetXTitle("#delta#phi_{T} (radians)");
rfgrpa->GetYaxis()->SetTitleSize(0.060);
rfgrpa->GetYaxis()->SetTitleOffset(1.15);
result->SetMarkerSize(1.0);
sf->SetLineColor(kAzure+8);
sfno2p2h->SetLineColor(kAzure-2);
rfgrpa->SetLineColor(kRed-4);
lfg->SetLineColor(kRed+3);
sf->SetLineStyle(1);
sfno2p2h->SetLineStyle(2);
rfgrpa->SetLineStyle(1);
lfg->SetLineStyle(3);
sf->SetLineWidth(4);
sfno2p2h->SetLineWidth(3);
rfgrpa->SetLineWidth(2);
lfg->SetLineWidth(2);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") leg = new TLegend(0.2,0.6,0.55,0.85);
//leg->SetHeader("NuWro11, CH");
leg->SetHeader("NuWro 11q");
leg->AddEntry(result,"T2K Fit to Data","ep");
leg->AddEntry(sf,Form("SF w/2p2h_{N}, #chi^{2}=%.1f",sfChi2),"lf");
leg->AddEntry(sfno2p2h,Form("SF w/o 2p2h, #chi^{2}=%.1f",sfno2p2hChi2),"lf");
leg->AddEntry(rfgrpa,Form("RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",rfgrpaChi2),"lf");
leg->AddEntry(lfg,Form("LFG+RPA+2p2h_{N}, #chi^{2}=%.1f",lfgChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
rfgrpa->Draw();
sf->Draw("][same");
sfno2p2h->Draw("][same");
rfgrpa->Draw("][same");
lfg->Draw("][same");
result->Draw("][same");
leg->Draw("][same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut_paper/${var}_NuWroModelComp.png");
c1->SaveAs("${1}/plotsOut_paper/${var}_NuWroModelComp.pdf");
c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_NuWroModelComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") lfg->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") lfg->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sfno2p2h->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sfno2p2h->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
rfgrpa->Draw();
sf->Draw("][same");
sfno2p2h->Draw("][same");
rfgrpa->Draw("][same");
lfg->Draw("][same");
result->Draw("][same");
//leg->Draw("][same");
gPad->SetLogy();
c2->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_NuWroModelComp.png");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_NuWroModelComp.pdf");
//***********************************
// Add inlay to plots
//***********************************
c1->cd();
leg->Delete();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.0,10E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(0.0,5E-39);
inlayLeg = new TLegend(0.32,0.5,0.84,0.85);
//inlayLeg->SetHeader("NuWro11, CH");
inlayLeg->SetHeader("NuWro 11q");
inlayLeg->AddEntry(result,"T2K Fit to Data","ep");
inlayLeg->AddEntry(sf,Form("SF w/2p2h_{N}, #chi^{2}=%.1f",sfChi2),"l");
inlayLeg->AddEntry(sfno2p2h,Form("SF w/o 2p2h, #chi^{2}=%.1f",sfno2p2hChi2),"l");
inlayLeg->AddEntry(rfgrpa,Form("RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",rfgrpaChi2),"l");
inlayLeg->AddEntry(lfg,Form("LFG+RPA+2p2h_{N}, #chi^{2}=%.1f",lfgChi2),"l");
inlayLeg->SetFillColor(kWhite);
inlayLeg->SetFillStyle(0);
if("${var}"=="dptOut" || "${var}"=="dphitOut"){
  TPad* inlay = new TPad("inlay","inlay",0.28,0.28,0.92,0.93);
  inlay->cd();
  TH1D* rfgrpa_clone = new TH1D(*rfgrpa);
  rfgrpa_clone->GetYaxis()->SetRangeUser(0.5E-40,60E-39);
  if("${var}"=="dphitOut") rfgrpa_clone->GetYaxis()->SetRangeUser(2E-41,10E-39);
  rfgrpa_clone->GetYaxis()->SetTitle("");
  rfgrpa_clone->GetXaxis()->SetTitle("");
  TH1D* result_clone = new TH1D(*result);
  result_clone->SetMarkerSize(0.5);
  rfgrpa_clone->Draw();
  sf->Draw("][same");
  sfno2p2h->Draw("][same");
  rfgrpa->Draw("][same");
  lfg->Draw("][same");
  result->Draw("][same");
  inlayLeg->Draw("][same");
  gPad->SetLogy();
  inlay->SetFillStyle(0);
  inlay->Update();
  c1->cd();
  inlay->Draw();
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_NuWroModelComp.png");
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_NuWroModelComp.pdf");
  c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_NuWroModelComp_inlay.root");
}
EOF
cd $1
done;
#Do NEUT model comp (full xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_RFG_RPARW_out.root neut_540_lyon.root neut5322_SF_MA1p0RW_out.root neut5322_SF_MA1p0RW_no2p2hRW_out.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
TH1D* rfgrpa = new TH1D();
TH1D* lfg = new TH1D();
TH1D* sf = new TH1D();
TH1D* sfno2p2h = new TH1D();
rfgrpa =  (TH1D*) _file0->Get("nuisMC");
lfg =  (TH1D*) _file1->Get("nuisMC");
sf =  (TH1D*) _file2->Get("nuisMC");
sfno2p2h =  (TH1D*) _file3->Get("nuisMC");
_file0->cd(); double rfgrpaChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double lfgChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double sfChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double sfno2p2hChi2 = chi2Hist->GetBinContent(5);
rfgrpa->SetLineColor(kRed-3);
lfg->SetLineColor(kBlue-4);
sf->SetLineColor(kGreen-6);
sfno2p2h->SetLineColor(kGreen-6);
sfno2p2h->SetLineStyle(2);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") leg = new TLegend(0.2,0.6,0.55,0.85);
leg->SetHeader("NEUT 5322/540");
leg->AddEntry(result,"T2K Fit to Data","lep");
leg->AddEntry(rfgrpa,Form("RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",rfgrpaChi2),"lf");
leg->AddEntry(lfg,Form("LFG+RPA+2p2h_{N}, #chi^{2}=%.1f",lfgChi2),"lf");
leg->AddEntry(sf,Form("SF+2p2h_{N}, #chi^{2}=%.1f",sfChi2),"lf");
leg->AddEntry(sfno2p2h,Form("SF, #chi^{2}=%.1f",sfno2p2hChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
rfgrpa->Draw();
lfg->Draw("][same");
sf->Draw("][same");
sfno2p2h->Draw("][same");
result->Draw("][same");
leg->Draw("][same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut_paper/${var}_NeutModelComp.png");
c1->SaveAs("${1}/plotsOut_paper/${var}_NeutModelComp.pdf");
c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_NeutModelComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") lfg->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") lfg->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sfno2p2h->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sfno2p2h->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
rfgrpa->Draw();
lfg->Draw("][same");
sf->Draw("][same");
sfno2p2h->Draw("][same");
result->Draw("][same");
//leg->Draw("][same");
gPad->SetLogy();
c2->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_NeutModelComp.png");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_NeutModelComp.pdf");
//***********************************
// Add inlay to plots
//***********************************
c1->cd();
leg->Delete();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.0,10E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(0.0,5E-39);
inlayLeg = new TLegend(0.32,0.5,0.84,0.85);
#inlayLeg->SetHeader("NEUT 5322/540");
inlayLeg->AddEntry(result,"T2K Fit to Data","lep");
inlayLeg->AddEntry(rfgrpa,Form("RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",rfgrpaChi2),"l");
inlayLeg->AddEntry(lfg,Form("LFG+RPA+2p2h_{N}, #chi^{2}=%.1f",lfgChi2),"l");
inlayLeg->AddEntry(sf,Form("SF+2p2h_{N}, #chi^{2}=%.1f",sfChi2),"l");
inlayLeg->AddEntry(sfno2p2h,Form("SF, #chi^{2}=%.1f",sfno2p2hChi2),"l");
inlayLeg->SetFillColor(kWhite);
inlayLeg->SetFillStyle(0);
if("${var}"=="dptOut" || "${var}"=="dphitOut"){
  TPad* inlay = new TPad("inlay","inlay",0.28,0.28,0.92,0.93);
  inlay->cd();
  TH1D* rfgrpa_clone = new TH1D(*rfgrpa);
  rfgrpa_clone->GetYaxis()->SetRangeUser(0.5E-40,60E-39);
  if("${var}"=="dphitOut") rfgrpa_clone->GetYaxis()->SetRangeUser(2E-41,10E-39);
  rfgrpa_clone->GetYaxis()->SetTitle("");
  rfgrpa_clone->GetXaxis()->SetTitle("");
  TH1D* result_clone = new TH1D(*result);
  result_clone->SetMarkerSize(0.5);
  rfgrpa_clone->Draw();
  lfg->Draw("][same");
  sf->Draw("][same");
  sfno2p2h->Draw("][same");
  result->Draw("][same");
  inlayLeg->Draw("][same");
  gPad->SetLogy();
  inlay->SetFillStyle(0);
  inlay->Update();
  c1->cd();
  inlay->Draw();
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_NeutModelComp.png");
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_NeutModelComp.pdf");
  c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_NeutModelComp_inlay.root");
}
EOF
cd $1
done;
#Do MA comparrison for SF (xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_6DCloneMA1p0RW_out.root neut5322_6DCloneMA1p1RW_out.root neut5322_6DCloneMA1p2RW_out.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* ma1p0 = new TH1D();
TH1D* ma1p1 = new TH1D();
TH1D* ma1p2 = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
ma1p0 =  (TH1D*) _file0->Get("nuisMC");
ma1p1 =  (TH1D*) _file1->Get("nuisMC");
ma1p2 =  (TH1D*) _file2->Get("nuisMC");
_file0->cd(); double ma1p0Chi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double ma1p1Chi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double ma1p2Chi2 = chi2Hist->GetBinContent(5);
ma1p0->SetLineColor(kBlue-4);
ma1p1->SetLineColor(kBlack);
ma1p2->SetLineColor(kViolet-4);
ma1p0->SetLineStyle(2);
ma1p1->SetLineStyle(1);
ma1p2->SetLineStyle(1);
leg = new TLegend(0.35,0.3,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
//leg->SetHeader("NEUT, CH, Benhar SF, BBBA05");
leg->SetHeader("NEUT 5322 SF+2p2h_{N}");
leg->AddEntry(result,"T2K Fit to Data","lep");
leg->AddEntry(ma1p0,Form("M_{Q}^{QE}=1.0, #chi^{2}=%.1f",ma1p0Chi2),"lf");
leg->AddEntry(ma1p1,Form("M_{Q}^{QE}=1.1, #chi^{2}=%.1f",ma1p1Chi2),"lf");
leg->AddEntry(ma1p2,Form("M_{Q}^{QE}=1.2, #chi^{2}=%.1f",ma1p2Chi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
ma1p0->Draw();
ma1p1->Draw("][same");
ma1p2->Draw("][same");
result->Draw("][same")
leg->Draw("][same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut_paper/${var}_SF_MAComp.png");
c1->SaveAs("${1}/plotsOut_paper/${var}_SF_MAComp.pdf");
c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_SF_MAComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") ma1p0->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") ma1p0->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") ma1p1->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") ma1p1->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") ma1p2->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") ma1p2->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
ma1p0->Draw();
ma1p1->Draw("][same");
ma1p2->Draw("][same");
result->Draw("][same")
//leg->Draw("][same");
gPad->SetLogy();
c2->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_SF_MAComp.png");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_SF_MAComp.pdf");
EOF
cd $1
done;
#Do RPA model comp (full xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l nuwro11_RFG.root nuwro11_LFG.root nuwro11_RFGRPA.root nuwro11_LFGRPA.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
TH1D* rfgrpa = new TH1D();
TH1D* lfg = new TH1D();
TH1D* sf = new TH1D();
TH1D* sfno2p2h = new TH1D();
rfgrpa =  (TH1D*) _file0->Get("nuisMC");
lfg =  (TH1D*) _file1->Get("nuisMC");
sf =  (TH1D*) _file2->Get("nuisMC");
sfno2p2h =  (TH1D*) _file3->Get("nuisMC");
_file0->cd(); double rfgrpaChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double lfgChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double sfChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double sfno2p2hChi2 = chi2Hist->GetBinContent(5);
if("${var}"=="dptOut") rfgrpa->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (cm^{2} Nucleon^{-1} GeV^{-1})");
if("${var}"=="datOut") rfgrpa->SetYTitle("#frac{d#sigma}{d#delta#alpha_{T}} (cm^{2} Nucleon^{-1} radian^{-1})");
if("${var}"=="dphitOut") rfgrpa->SetYTitle("#frac{d#sigma}{d#delta#phi_{T}} (cm^{2} Nucleon^{-1} radian^{-1})");
if("${var}"=="dptOut") rfgrpa->SetXTitle("#deltap_{T} (GeV)");
if("${var}"=="datOut") rfgrpa->SetXTitle("#delta#alpha_{T} (radians)");
if("${var}"=="dphitOut") rfgrpa->SetXTitle("#delta#phi_{T} (radians)");
rfgrpa->GetYaxis()->SetTitleSize(0.060);
rfgrpa->GetYaxis()->SetTitleOffset(1.15);
result->SetMarkerSize(1.0);
rfgrpa->SetLineColor(kAzure+8);
lfg->SetLineColor(kAzure-2);
sf->SetLineColor(kRed-4);
sfno2p2h->SetLineColor(kRed+3);
lfg->SetLineStyle(2);
sfno2p2h->SetLineStyle(3);
rfgrpa->SetLineWidth(4);
lfg->SetLineWidth(3);
sf->SetLineWidth(2);
sfno2p2h->SetLineWidth(2);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") leg = new TLegend(0.2,0.6,0.55,0.85);
//leg->SetHeader("NuWro11, CH");
leg->SetHeader("NuWro 11q");
leg->AddEntry(result,"T2K Fit to Data","ep");
leg->AddEntry(rfgrpa,Form("RFG+2p2h_{N}, #chi^{2}=%.1f",rfgrpaChi2),"lf");
leg->AddEntry(lfg,Form("LFG+2p2h_{N}, #chi^{2}=%.1f",lfgChi2),"lf");
leg->AddEntry(sf,Form("RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",sfChi2),"lf");
leg->AddEntry(sfno2p2h,Form("LFG+RPA+2p2h_{N}, #chi^{2}=%.1f",sfno2p2hChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
rfgrpa->Draw();
lfg->Draw("][same");
sf->Draw("][same");
sfno2p2h->Draw("][same");
result->Draw("][same");
leg->Draw("][same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut_paper/${var}_NuWroRPAModelComp.png");
c1->SaveAs("${1}/plotsOut_paper/${var}_NuWroRPAModelComp.pdf");
c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_NuWroRPAModelComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") lfg->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") lfg->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sfno2p2h->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sfno2p2h->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
rfgrpa->Draw();
lfg->Draw("][same");
sf->Draw("][same");
sfno2p2h->Draw("][same");
result->Draw("][same");
//leg->Draw("][same");
gPad->SetLogy();
c2->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_NuWroRPAModelComp.png");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_NuWroRPAModelComp.pdf");
//***********************************
// Add inlay to plots
//***********************************
c1->cd();
leg->Delete();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.0,10E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(0.0,5E-39);
inlayLeg = new TLegend(0.32,0.5,0.84,0.85);
//inlayLeg->SetHeader("NuWro 11q, CH");
inlayLeg->SetHeader("NuWro 11q");
inlayLeg->AddEntry(result,"T2K Fit to Data","ep");
inlayLeg->AddEntry(rfgrpa,Form("RFG+2p2h_{N}, #chi^{2}=%.1f",rfgrpaChi2),"lf");
inlayLeg->AddEntry(lfg,Form("LFG+2p2h_{N}, #chi^{2}=%.1f",lfgChi2),"lf");
inlayLeg->AddEntry(sf,Form("RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",sfChi2),"lf");
inlayLeg->AddEntry(sfno2p2h,Form("LFG+RPA+2p2h_{N}, #chi^{2}=%.1f",sfno2p2hChi2),"lf");
inlayLeg->SetFillColor(kWhite);
inlayLeg->SetFillStyle(0);
if("${var}"=="dptOut" || "${var}"=="dphitOut"){
  TPad* inlay = new TPad("inlay","inlay",0.28,0.28,0.92,0.93);
  inlay->cd();
  TH1D* rfgrpa_clone = new TH1D(*rfgrpa);
  rfgrpa_clone->GetYaxis()->SetRangeUser(0.5E-40,60E-39);
  if("${var}"=="dphitOut") rfgrpa_clone->GetYaxis()->SetRangeUser(2E-41,10E-39);
  rfgrpa_clone->GetYaxis()->SetTitle("");
  rfgrpa_clone->GetXaxis()->SetTitle("");
  TH1D* result_clone = new TH1D(*result);
  result_clone->SetMarkerSize(0.5);
  rfgrpa_clone->Draw();
  lfg->Draw("][same");
  sf->Draw("][same");
  sfno2p2h->Draw("][same");
  result->Draw("][same");
  inlayLeg->Draw("][same");
  gPad->SetLogy();
  inlay->SetFillStyle(0);
  inlay->Update();
  c1->cd();
  inlay->Draw();
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_NuWroRPAModelComp.png");
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_NuWroRPAModelComp.pdf");
  c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_NuWroRPAModelComp_inlay.root");
}
EOF
cd $1
done;

#Do RPA model comp between NEUT and NuWro (full xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l nuwro11_RFG.root neut5322_RFG_noRPA_out.root nuwro11_RFGRPA.root neut5322_RFG_RPARW_out.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
TH1D* rfgrpa = new TH1D();
TH1D* lfg = new TH1D();
TH1D* sf = new TH1D();
TH1D* sfno2p2h = new TH1D();
rfgrpa =  (TH1D*) _file0->Get("nuisMC");
lfg =  (TH1D*) _file1->Get("nuisMC");
sf =  (TH1D*) _file2->Get("nuisMC");
sfno2p2h =  (TH1D*) _file3->Get("nuisMC");
_file0->cd(); double rfgrpaChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double lfgChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double sfChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double sfno2p2hChi2 = chi2Hist->GetBinContent(5);
rfgrpa->SetLineColor(kRed-3);
lfg->SetLineColor(kBlue-4);
sf->SetLineColor(kRed-3);
sfno2p2h->SetLineColor(kBlue-4);
sf->SetLineStyle(2);
sfno2p2h->SetLineStyle(2);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") leg = new TLegend(0.2,0.6,0.55,0.85);
//leg->SetHeader("NuWro11, CH");
leg->AddEntry(result,"T2K Fit to Data","lep");
leg->AddEntry(rfgrpa,Form("NuWro RFG+2p2h_{N}, #chi^{2}=%.1f",rfgrpaChi2),"lf");
leg->AddEntry(lfg,Form("NEUT RFG+2p2h_{N}, #chi^{2}=%.1f",lfgChi2),"lf");
leg->AddEntry(sf,Form("NuWro RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",sfChi2),"lf");
leg->AddEntry(sfno2p2h,Form("NEUT RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",sfno2p2hChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
rfgrpa->Draw();
lfg->Draw("][same");
sf->Draw("][same");
sfno2p2h->Draw("][same");
result->Draw("][same");
leg->Draw("][same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut_paper/${var}_NeutNuWroRPAModelComp.png");
c1->SaveAs("${1}/plotsOut_paper/${var}_NeutNuWroRPAModelComp.pdf");
c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_NeutNuWroRPAModelComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") lfg->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") lfg->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sfno2p2h->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sfno2p2h->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
rfgrpa->Draw();
lfg->Draw("][same");
sf->Draw("][same");
sfno2p2h->Draw("][same");
result->Draw("][same");
//leg->Draw("][same");
gPad->SetLogy();
c2->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_NeutNuWroRFGRPAComp.png");
c2->SaveAs("${1}/plotsOut_paper/log/${var}_NeutNuWroRFGRPAComp.pdf");
//***********************************
// Add inlay to plots
//***********************************
c1->cd();
leg->Delete();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.0,10E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(0.0,5E-39);
inlayLeg = new TLegend(0.32,0.5,0.84,0.85);
//inlayLeg->SetHeader("NuWro11, CH");
inlayLeg->AddEntry(result,"T2K Fit to Data","lep");
inlayLeg->AddEntry(rfgrpa,Form("NuWro RFG+2p2h_{N}, #chi^{2}=%.1f",rfgrpaChi2),"lf");
inlayLeg->AddEntry(lfg,Form("NEUT RFG+2p2h_{N}, #chi^{2}=%.1f",lfgChi2),"lf");
inlayLeg->AddEntry(sf,Form("NuWro RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",sfChi2),"lf");
inlayLeg->AddEntry(sfno2p2h,Form("NEUT RFG+RPA+2p2h_{N}, #chi^{2}=%.1f",sfno2p2hChi2),"lf");
inlayLeg->SetFillColor(kWhite);
inlayLeg->SetFillStyle(0);
if("${var}"=="dptOut" || "${var}"=="dphitOut"){
  TPad* inlay = new TPad("inlay","inlay",0.28,0.28,0.92,0.93);
  inlay->cd();
  TH1D* rfgrpa_clone = new TH1D(*rfgrpa);
  rfgrpa_clone->GetYaxis()->SetRangeUser(0.5E-40,60E-39);
  if("${var}"=="dphitOut") rfgrpa_clone->GetYaxis()->SetRangeUser(2E-41,10E-39);
  rfgrpa_clone->GetYaxis()->SetTitle("");
  rfgrpa_clone->GetXaxis()->SetTitle("");
  TH1D* result_clone = new TH1D(*result);
  result_clone->SetMarkerSize(0.5);
  rfgrpa_clone->Draw();
  lfg->Draw("][same");
  sf->Draw("][same");
  sfno2p2h->Draw("][same");
  result->Draw("][same");
  inlayLeg->Draw("][same");
  gPad->SetLogy();
  inlay->SetFillStyle(0);
  inlay->Update();
  c1->cd();
  inlay->Draw();
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_NeutNuWroRFGRPAComp.png");
  c1->SaveAs("${1}/plotsOut_paper/inlay/${var}_NeutNuWroRFGRPAComp.pdf");
  c1->SaveAs("${1}/plotsOut_paper/rootfiles/${var}_NeutNuWroRFGRPAComp_inlay.root");
}
EOF
cd $1
done;