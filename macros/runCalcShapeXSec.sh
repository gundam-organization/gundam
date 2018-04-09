#!/bin/sh
for dir in $(ls nuisance); do
  echo "Using files in dir: ${dir}"
  cd nuisance/${dir};
  for file in $(ls *.root) ; do
    echo "Using file: ${file}"
    cd $1
root -b -l <<EOF
.L calcShapeXSecWithErrors.cc+
calcShapeXsecWithErrors("./dptResults/quickFtX_xsecOut.root", "./dptOut_half2p2h/${file}", 50000, "dpt", "nuisance/${dir}/${file}", false, true)
calcShapeXsecWithErrors("./datResults/quickFtX_xsecOut.root", "./datOut_half2p2h/${file}", 50000, "dat", "nuisance/${dir}/${file}", false, true)
calcShapeXsecWithErrors("./dphitResults/quickFtX_xsecOut.root", "./dphitOut_half2p2h/${file}", 50000, "dphit", "nuisance/${dir}/${file}", false, true)
.q
EOF
    echo "Finished with file: ${file}"
  done;
done;
cd $1
for dir in $(ls nuisance); do
  echo "Using files in dir: ${dir}"
  cd nuisance/${dir};
  for file in $(ls *.root) ; do
    echo "Using file: ${file}"
    cd $1
root -b -l <<EOF
.L calcShapeXSecWithErrors.cc+
calcShapeXsecWithErrors("./dptResults/quickFtX_xsecOut.root", "./dptOut/${file}", 50000, "dpt", "nuisance/${dir}/${file}")
calcShapeXsecWithErrors("./datResults/quickFtX_xsecOut.root", "./datOut/${file}", 50000, "dat", "nuisance/${dir}/${file}")
calcShapeXsecWithErrors("./dphitResults/quickFtX_xsecOut.root", "./dphitOut/${file}", 50000, "dphit", "nuisance/${dir}/${file}")
.q
EOF
    echo "Finished with file: ${file}"
  done;
done;
cd $1
for dir in $(ls oct17Nuis); do
  echo "Using files in dir: ${dir}"
  cd oct17Nuis/${dir};
  for file in $(ls *540*.root) ; do
    echo "Using file: ${file}"
    cd $1
root -b -l <<EOF
.L calcShapeXSecWithErrors.cc+
calcShapeXsecWithErrors("./dptResults/quickFtX_xsecOut.root", "./dptOut/${file}", 50000, "dpt", "oct17Nuis/${dir}/${file}", true)
calcShapeXsecWithErrors("./datResults/quickFtX_xsecOut.root", "./datOut/${file}", 50000, "dat", "oct17Nuis/${dir}/${file}", true)
calcShapeXsecWithErrors("./dphitResults/quickFtX_xsecOut.root", "./dphitOut/${file}", 50000, "dphit", "oct17Nuis/${dir}/${file}", true)
.q
EOF
    echo "Finished with file: ${file}"
  done;
  cd $1
done;
exit;
#Do generator comparrison (xsec) (Gibuu is: GiBUU 2016 (ANL Pi-Prod, Oset Delta in-medium width broadening)):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l gibuu_150317_out.root  neut5322_SF_out.root neut5322_RFG_RPARW_out.root nuwro11_LFG.root genier2124_default.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* gibuu = new TH1D();
TH1D* neut_sf = new TH1D();
TH1D* neut_rfgrpa = new TH1D();
TH1D* nuwro = new TH1D();
TH1D* genie = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
gibuu =  (TH1D*) _file0->Get("nuisMC");
neut_sf =  (TH1D*) _file1->Get("nuisMC");
neut_rfgrpa =  (TH1D*) _file2->Get("nuisMC");
nuwro =  (TH1D*) _file3->Get("nuisMC");
genie =  (TH1D*) _file4->Get("nuisMC");
_file0->cd(); double gibuuChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double neut_sfChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double neut_rfgChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double nuwroChi2 = chi2Hist->GetBinContent(5);
_file4->cd(); double genieChi2 = chi2Hist->GetBinContent(5);
gibuu->SetLineColor(kRed-3);
neut_sf->SetLineColor(kViolet-4);
neut_rfgrpa->SetLineColor(kBlue-4);
nuwro->SetLineColor(kGreen-6);
genie->SetLineColor(kCyan+2);
gibuu->SetLineStyle(1);
neut_sf->SetLineStyle(1);
neut_rfgrpa->SetLineStyle(2);
nuwro->SetLineStyle(1);
genie->SetLineStyle(1);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(neut_sf, Form("NEUT 5322 SF, #chi^{2}=%.1f",neut_sfChi2), "lf");
leg->AddEntry(neut_rfgrpa, Form("NEUT 5322 RFG+RPA, #chi^{2}=%.1f",neut_rfgChi2), "lf");
leg->AddEntry(nuwro, Form("NuWro 11 LFG, #chi^{2}=%.1f",nuwroChi2), "lf");
leg->AddEntry(genie, Form("GENIE 2.12.4 RFG, #chi^{2}=%.1f",genieChi2), "lf");
leg->AddEntry(gibuu, Form("GiBUU 2016, #chi^{2}=%.1f",gibuuChi2), "lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
gibuu->Draw();
neut_sf->Draw("same");
neut_rfgrpa->Draw("same");
nuwro->Draw("same");
genie->Draw("same");
result->Draw("same");
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_GenComp.png");
c1->SaveAs("${1}/plotsOut/${var}_GenComp.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_GenComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") gibuu->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") gibuu->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") neut_sf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") neut_sf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") neut_rfgrpa->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") neut_rfgrpa->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nuwro->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") nuwro->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") genie->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") genie->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
gibuu->Draw();
neut_sf->Draw("same");
neut_rfgrpa->Draw("same");
nuwro->Draw("same");
genie->Draw("same");
result->Draw("same");
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_GenComp.png");
c2->SaveAs("${1}/plotsOut/log/${var}_GenComp.pdf");
EOF
cd $1
done;
#Do shape only generator comparrison
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l gibuu_150317_out.root  neut5322_SF_out.root neut5322_RFG_RPARW_out.root nuwro11_LFG.root genier2124_default.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* gibuu = new TH1D();
TH1D* neut_sf = new TH1D();
TH1D* neut_rfgrpa = new TH1D();
TH1D* nuwro = new TH1D();
TH1D* genie = new TH1D();
result =  (TH1D*) _file0->Get("Result_shapeOnly");
gibuu =  (TH1D*) _file0->Get("nuisMC_shapeOnly");
neut_sf =  (TH1D*) _file1->Get("nuisMC_shapeOnly");
neut_rfgrpa =  (TH1D*) _file2->Get("nuisMC_shapeOnly");
nuwro =  (TH1D*) _file3->Get("nuisMC_shapeOnly");
genie =  (TH1D*) _file4->Get("nuisMC_shapeOnly");
_file0->cd(); double gibuuChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file1->cd(); double neut_sfChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file2->cd(); double neut_rfgChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file3->cd(); double nuwroChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file4->cd(); double genieChi2 = chi2Hist_shapeOnly->GetBinContent(5);
gibuu->SetLineColor(kRed-3);
neut_sf->SetLineColor(kViolet-4);
neut_rfgrpa->SetLineColor(kBlue-4);
nuwro->SetLineColor(kGreen-6);
genie->SetLineColor(kCyan+2);
gibuu->SetLineStyle(1);
neut_sf->SetLineStyle(1);
neut_rfgrpa->SetLineStyle(2);
nuwro->SetLineStyle(1);
genie->SetLineStyle(1);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(neut_sf, Form("NEUT 5322 SF, #chi^{2}=%.1f",neut_sfChi2), "lf");
leg->AddEntry(neut_rfgrpa, Form("NEUT 5322 RFG+RPA, #chi^{2}=%.1f",neut_rfgChi2), "lf");
leg->AddEntry(nuwro, Form("NuWro 11 LFG, #chi^{2}=%.1f",nuwroChi2), "lf");
leg->AddEntry(genie, Form("GENIE 2.12.4 RFG, #chi^{2}=%.1f",genieChi2), "lf");
leg->AddEntry(gibuu, Form("GiBUU 2016, #chi^{2}=%.1f",gibuuChi2), "lf")
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
gibuu->Draw();
neut_sf->Draw("same");
neut_rfgrpa->Draw("same");
nuwro->Draw("same");
genie->Draw("same");
result->Draw("same");
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_GenCompShapeOnly.png");
c1->SaveAs("${1}/plotsOut/${var}_GenCompShapeOnly.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_GenCompShapeOnly.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") gibuu->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") gibuu->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") neut_sf->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") neut_sf->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") neut_rfgrpa->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") neut_rfgrpa->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") nuwro->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") nuwro->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") genie->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") genie->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(0.002,0.5);
gibuu->Draw();
neut_sf->Draw("same");
neut_rfgrpa->Draw("same");
nuwro->Draw("same");
genie->Draw("same");
result->Draw("same");
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_GenCompShapeOnly.png");
c2->SaveAs("${1}/plotsOut/log/${var}_GenCompShapeOnly.pdf");
EOF
cd $1
done;
#Do extended generator comparrison (xsec) (Gibuu is: GiBUU 2016 (ANL Pi-Prod, Oset Delta in-medium width broadening)):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l gibuu_150317_out.root  neut5322_SF_out.root neut5322_RFG_RPARW_out.root nuwro11_LFG.root genier2124_default.root nuwro11_SF.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* gibuu = new TH1D();
TH1D* neut_sf = new TH1D();
TH1D* neut_rfgrpa = new TH1D();
TH1D* nuwro = new TH1D();
TH1D* genie = new TH1D();
TH1D* nuwrosf = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
gibuu =  (TH1D*) _file0->Get("nuisMC");
neut_sf =  (TH1D*) _file1->Get("nuisMC");
neut_rfgrpa =  (TH1D*) _file2->Get("nuisMC");
nuwro =  (TH1D*) _file3->Get("nuisMC");
genie =  (TH1D*) _file4->Get("nuisMC");
nuwrosf =  (TH1D*) _file5->Get("nuisMC");
_file0->cd(); double gibuuChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double neut_sfChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double neut_rfgChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double nuwroChi2 = chi2Hist->GetBinContent(5);
_file4->cd(); double genieChi2 = chi2Hist->GetBinContent(5);
_file5->cd(); double nuwrosfChi2 = chi2Hist->GetBinContent(5);
gibuu->SetLineColor(kRed-3);
neut_sf->SetLineColor(kViolet-4);
neut_rfgrpa->SetLineColor(kBlue-4);
nuwro->SetLineColor(kGreen-6);
nuwrosf->SetLineColor(kGreen-6);
genie->SetLineColor(kOrange-3);
gibuu->SetLineStyle(1);
genie->SetLineWidth(3);
neut_sf->SetLineStyle(2);
neut_sf->SetLineWidth(3);
neut_rfgrpa->SetLineStyle(1);
nuwro->SetLineStyle(1);
nuwrosf->SetLineStyle(2);
nuwrosf->SetLineWidth(3);
genie->SetLineStyle(1);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(neut_sf, Form("NEUT 5322 SF, #chi^{2}=%.1f",neut_sfChi2), "lf");
leg->AddEntry(neut_rfgrpa, Form("NEUT 5322 RFG+RPA, #chi^{2}=%.1f",neut_rfgChi2), "lf");
leg->AddEntry(nuwro, Form("NuWro 11 LFG, #chi^{2}=%.1f",nuwroChi2), "lf");
leg->AddEntry(nuwrosf, Form("NuWro 11 SF, #chi^{2}=%.1f",nuwrosfChi2), "lf");
leg->AddEntry(genie, Form("GENIE 2.12.4 RFG, #chi^{2}=%.1f",genieChi2), "lf");
leg->AddEntry(gibuu, Form("GiBUU 2016, #chi^{2}=%.1f",gibuuChi2), "lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
gibuu->Draw();
neut_sf->Draw("same");
neut_rfgrpa->Draw("same");
nuwro->Draw("same");
nuwrosf->Draw("same");
genie->Draw("same");
result->Draw("same");
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_GenCompExt.png");
c1->SaveAs("${1}/plotsOut/${var}_GenCompExt.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_GenCompExt.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") gibuu->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") gibuu->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") neut_sf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") neut_sf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") neut_rfgrpa->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") neut_rfgrpa->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nuwro->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") nuwro->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nuwrosf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") nuwrosf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") genie->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") genie->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
gibuu->Draw();
neut_sf->Draw("same");
neut_rfgrpa->Draw("same");
nuwro->Draw("same");
nuwrosf->Draw("same");
genie->Draw("same");
result->Draw("same");
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_GenCompExt.png");
c2->SaveAs("${1}/plotsOut/log/${var}_GenCompExt.pdf");
EOF
cd $1
done;
#Do extended shape only generator comparrison
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l gibuu_150317_out.root  neut5322_SF_out.root neut5322_RFG_RPARW_out.root nuwro11_LFG.root genier2124_default.root nuwro11_SF.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* gibuu = new TH1D();
TH1D* neut_sf = new TH1D();
TH1D* neut_rfgrpa = new TH1D();
TH1D* nuwro = new TH1D();
TH1D* genie = new TH1D();
TH1D* nuwrosf = new TH1D();
result =  (TH1D*) _file0->Get("Result_shapeOnly");
gibuu =  (TH1D*) _file0->Get("nuisMC_shapeOnly");
neut_sf =  (TH1D*) _file1->Get("nuisMC_shapeOnly");
neut_rfgrpa =  (TH1D*) _file2->Get("nuisMC_shapeOnly");
nuwro =  (TH1D*) _file3->Get("nuisMC_shapeOnly");
genie =  (TH1D*) _file4->Get("nuisMC_shapeOnly");
nuwrosf =  (TH1D*) _file5->Get("nuisMC_shapeOnly");
_file0->cd(); double gibuuChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file1->cd(); double neut_sfChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file2->cd(); double neut_rfgChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file3->cd(); double nuwroChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file4->cd(); double genieChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file5->cd(); double nuwrosfChi2 = chi2Hist_shapeOnly->GetBinContent(5);
gibuu->SetLineColor(kRed-3);
neut_sf->SetLineColor(kViolet-4);
neut_rfgrpa->SetLineColor(kBlue-4);
nuwro->SetLineColor(kGreen-6);
nuwrosf->SetLineColor(kGreen-6);
genie->SetLineColor(kOrange-3);
gibuu->SetLineStyle(1);
genie->SetLineWidth(3);
neut_sf->SetLineStyle(2);
neut_sf->SetLineWidth(3);
neut_rfgrpa->SetLineStyle(1);
nuwro->SetLineStyle(1);
nuwrosf->SetLineStyle(2);
nuwrosf->SetLineWidth(3);
genie->SetLineStyle(1);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(neut_sf, Form("NEUT 5322 SF, #chi^{2}=%.1f",neut_sfChi2), "lf");
leg->AddEntry(neut_rfgrpa, Form("NEUT 5322 RFG+RPA, #chi^{2}=%.1f",neut_rfgChi2), "lf");
leg->AddEntry(nuwro, Form("NuWro 11 LFG, #chi^{2}=%.1f",nuwroChi2), "lf");
leg->AddEntry(nuwrosf, Form("NuWro 11 SF, #chi^{2}=%.1f",nuwrosfChi2), "lf");
leg->AddEntry(genie, Form("GENIE 2.12.4 RFG, #chi^{2}=%.1f",genieChi2), "lf");
leg->AddEntry(gibuu, Form("GiBUU 2016, #chi^{2}=%.1f",gibuuChi2), "lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
gibuu->Draw();
neut_sf->Draw("same");
neut_rfgrpa->Draw("same");
nuwro->Draw("same");
nuwrosf->Draw("same");
genie->Draw("same");
result->Draw("same");
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_GenCompExShapeOnly.png");
c1->SaveAs("${1}/plotsOut/${var}_GenCompExShapeOnly.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_GenCompExShapeOnly.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") gibuu->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") gibuu->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") neut_sf->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") neut_sf->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") neut_rfgrpa->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") neut_rfgrpa->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") nuwro->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") nuwro->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") nuwrosf->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") nuwrosf->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") genie->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") genie->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(0.002,0.5);
gibuu->Draw();
neut_sf->Draw("same");
neut_rfgrpa->Draw("same");
nuwro->Draw("same");
nuwrosf->Draw("same");
genie->Draw("same");
result->Draw("same");
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_GenCompExShapeOnly.png");
c2->SaveAs("${1}/plotsOut/log/${var}_GenCompExShapeOnly.pdf");
EOF
cd $1
done;
#Do FSI comparrison for RFG+nonRelRPA (xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_RFGRPA_NoFSI_out.root neut5322_RFGRPA_out.root neut5322_RFGRPA_ExtraFSI_out.root neut5322_RFGRPA_4ExtraFSI_out.root neut5322_RFGRPA_6ExtraFSI_out.root neut5322_RFGRPA_8ExtraFSI_out.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* nofsi = new TH1D();
TH1D* nomfsi = new TH1D();
TH1D* twofsi = new TH1D();
TH1D* fourfsi = new TH1D();
TH1D* sixfsi = new TH1D();
TH1D* eightfsi = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
nofsi =  (TH1D*) _file0->Get("nuisMC");
nomfsi =  (TH1D*) _file1->Get("nuisMC");
twofsi =  (TH1D*) _file2->Get("nuisMC");
fourfsi =  (TH1D*) _file3->Get("nuisMC");
sixfsi =  (TH1D*) _file4->Get("nuisMC");
eightfsi =  (TH1D*) _file5->Get("nuisMC");
_file0->cd(); double nofsiChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double nomfsiChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double twofsiChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double fourfsiChi2 = chi2Hist->GetBinContent(5);
_file4->cd(); double sixfsiChi2 = chi2Hist->GetBinContent(5);
_file5->cd(); double eightfsiChi2 = chi2Hist->GetBinContent(5);
nofsi->SetLineColor(kBlue-4);
nomfsi->SetLineColor(kBlack);
twofsi->SetLineColor(kViolet-4);
fourfsi->SetLineColor(kGreen-6);
sixfsi->SetLineColor(kRed-3);
eightfsi->SetLineColor(kCyan+2);
nofsi->SetLineStyle(2);
nomfsi->SetLineStyle(1);
twofsi->SetLineStyle(1);
fourfsi->SetLineStyle(1);
sixfsi->SetLineStyle(1);
eightfsi->SetLineStyle(1);
leg = new TLegend(0.35,0.3,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->SetHeader("NEUT, CH, RFG+RPA(non rel.), BBBA05");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"lf");
leg->AddEntry(nomfsi,Form("1 #times FSI, #chi^{2}=%.1f",nomfsiChi2),"lf");
leg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"lf");
leg->AddEntry(fourfsi,Form("4 #times FSI, #chi^{2}=%.1f",fourfsiChi2),"lf");
leg->AddEntry(sixfsi,Form("6 #times FSI, #chi^{2}=%.1f",sixfsiChi2),"lf");
leg->AddEntry(eightfsi,Form("8 #times FSI, #chi^{2}=%.1f",eightfsiChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_RFGnonRelRPA_FSIComp.png");
c1->SaveAs("${1}/plotsOut/${var}_RFGnonRelRPA_FSIComp.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_RFGnonRelRPA_FSIComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nomfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nomfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") twofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") twofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") fourfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") fourfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sixfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") sixfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") eightfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") eightfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_RFGnonRelRPA_FSIComp.png");
c2->SaveAs("${1}/plotsOut/log/${var}_RFGnonRelRPA_FSIComp.pdf");
EOF
cd $1
done;
#Do FSI comparrison for RFG+nonRelRPA (shapeOnly):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_RFGRPA_NoFSI_out.root neut5322_RFGRPA_out.root neut5322_RFGRPA_ExtraFSI_out.root neut5322_RFGRPA_4ExtraFSI_out.root neut5322_RFGRPA_6ExtraFSI_out.root neut5322_RFGRPA_8ExtraFSI_out.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* nofsi = new TH1D();
TH1D* nomfsi = new TH1D();
TH1D* twofsi = new TH1D();
TH1D* fourfsi = new TH1D();
TH1D* sixfsi = new TH1D();
TH1D* eightfsi = new TH1D();
result =  (TH1D*) _file0->Get("Result_shapeOnly");
nofsi =  (TH1D*) _file0->Get("nuisMC_shapeOnly");
nomfsi =  (TH1D*) _file1->Get("nuisMC_shapeOnly");
twofsi =  (TH1D*) _file2->Get("nuisMC_shapeOnly");
fourfsi =  (TH1D*) _file3->Get("nuisMC_shapeOnly");
sixfsi =  (TH1D*) _file4->Get("nuisMC_shapeOnly");
eightfsi =  (TH1D*) _file5->Get("nuisMC_shapeOnly");
_file0->cd(); double nofsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file1->cd(); double nomfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file2->cd(); double twofsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file3->cd(); double fourfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file4->cd(); double sixfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file5->cd(); double eightfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
nofsi->SetLineColor(kBlue-4);
nomfsi->SetLineColor(kBlack);
twofsi->SetLineColor(kViolet-4);
fourfsi->SetLineColor(kGreen-6);
sixfsi->SetLineColor(kRed-3);
eightfsi->SetLineColor(kCyan+2);
nofsi->SetLineStyle(2);
nomfsi->SetLineStyle(1);
twofsi->SetLineStyle(1);
fourfsi->SetLineStyle(1);
sixfsi->SetLineStyle(1);
eightfsi->SetLineStyle(1);
leg = new TLegend(0.35,0.3,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->SetHeader("NEUT, CH, RFG+RPA(non rel.), BBBA05");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"lf");
leg->AddEntry(nomfsi,Form("1 #times FSI, #chi^{2}=%.1f",nomfsiChi2),"lf");
leg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"lf");
leg->AddEntry(fourfsi,Form("4 #times FSI, #chi^{2}=%.1f",fourfsiChi2),"lf");
leg->AddEntry(sixfsi,Form("6 #times FSI, #chi^{2}=%.1f",sixfsiChi2),"lf");
leg->AddEntry(eightfsi,Form("8 #times FSI, #chi^{2}=%.1f",eightfsiChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_RFGnonRelRPA_FSICompShapeOnly.png");
c1->SaveAs("${1}/plotsOut/${var}_RFGnonRelRPA_FSICompShapeOnly.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_RFGnonRelRPA_FSICompShapeOnly.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") nomfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") nomfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") twofsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") twofsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") fourfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") fourfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") sixfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") sixfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") eightfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") eightfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(0.001,0.5);
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_RFGnonRelRPA_FSICompShapeOnly.png");
c2->SaveAs("${1}/plotsOut/log/${var}_RFGnonRelRPA_FSICompShapeOnly.pdf");
EOF
cd $1
done;
#Do FSI comparrison for RFG+RPA (xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_RFGRPARW_NoFSI.card.root neut5322_RFG_RPARW_out.root neut5322_RFGRPARW_ExtraFSI.card.root neut5322_RFGRPARW_4ExtraFSI.card.root neut5322_RFGRPARW_6ExtraFSI.card.root neut5322_RFGRPARW_8ExtraFSI.card.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* nofsi = new TH1D();
TH1D* nomfsi = new TH1D();
TH1D* twofsi = new TH1D();
TH1D* fourfsi = new TH1D();
TH1D* sixfsi = new TH1D();
TH1D* eightfsi = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
nofsi =  (TH1D*) _file0->Get("nuisMC");
nomfsi =  (TH1D*) _file1->Get("nuisMC");
twofsi =  (TH1D*) _file2->Get("nuisMC");
fourfsi =  (TH1D*) _file3->Get("nuisMC");
sixfsi =  (TH1D*) _file4->Get("nuisMC");
eightfsi =  (TH1D*) _file5->Get("nuisMC");
_file0->cd(); double nofsiChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double nomfsiChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double twofsiChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double fourfsiChi2 = chi2Hist->GetBinContent(5);
_file4->cd(); double sixfsiChi2 = chi2Hist->GetBinContent(5);
_file5->cd(); double eightfsiChi2 = chi2Hist->GetBinContent(5);
nofsi->SetLineColor(kBlue-4);
nomfsi->SetLineColor(kBlack);
twofsi->SetLineColor(kViolet-4);
fourfsi->SetLineColor(kGreen-6);
sixfsi->SetLineColor(kRed-3);
eightfsi->SetLineColor(kCyan+2);
nofsi->SetLineStyle(2);
nomfsi->SetLineStyle(1);
twofsi->SetLineStyle(1);
fourfsi->SetLineStyle(1);
sixfsi->SetLineStyle(1);
eightfsi->SetLineStyle(1);
leg = new TLegend(0.35,0.3,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->SetHeader("NEUT, CH, RFG+RPA, BBBA05");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"lf");
leg->AddEntry(nomfsi,Form("1 #times FSI, #chi^{2}=%.1f",nomfsiChi2),"lf");
leg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"lf");
leg->AddEntry(fourfsi,Form("4 #times FSI, #chi^{2}=%.1f",fourfsiChi2),"lf");
leg->AddEntry(sixfsi,Form("6 #times FSI, #chi^{2}=%.1f",sixfsiChi2),"lf");
leg->AddEntry(eightfsi,Form("8 #times FSI, #chi^{2}=%.1f",eightfsiChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_RFGRPA_FSIComp.png");
c1->SaveAs("${1}/plotsOut/${var}_RFGRPA_FSIComp.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_RFGRPA_FSIComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nomfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nomfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") twofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") twofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") fourfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") fourfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sixfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") sixfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") eightfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") eightfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_RFGRPA_FSIComp.png");
c2->SaveAs("${1}/plotsOut/log/${var}_RFGRPA_FSIComp.pdf");
EOF
cd $1
done;
#Do FSI comparrison for RFG+RPA (shapeOnly):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_RFGRPARW_NoFSI.card.root neut5322_RFG_RPARW_out.root neut5322_RFGRPARW_ExtraFSI.card.root neut5322_RFGRPARW_4ExtraFSI.card.root neut5322_RFGRPARW_6ExtraFSI.card.root neut5322_RFGRPARW_8ExtraFSI.card.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* nofsi = new TH1D();
TH1D* nomfsi = new TH1D();
TH1D* twofsi = new TH1D();
TH1D* fourfsi = new TH1D();
TH1D* sixfsi = new TH1D();
TH1D* eightfsi = new TH1D();
result =  (TH1D*) _file0->Get("Result_shapeOnly");
nofsi =  (TH1D*) _file0->Get("nuisMC_shapeOnly");
nomfsi =  (TH1D*) _file1->Get("nuisMC_shapeOnly");
twofsi =  (TH1D*) _file2->Get("nuisMC_shapeOnly");
fourfsi =  (TH1D*) _file3->Get("nuisMC_shapeOnly");
sixfsi =  (TH1D*) _file4->Get("nuisMC_shapeOnly");
eightfsi =  (TH1D*) _file5->Get("nuisMC_shapeOnly");
_file0->cd(); double nofsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file1->cd(); double nomfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file2->cd(); double twofsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file3->cd(); double fourfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file4->cd(); double sixfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file5->cd(); double eightfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
nofsi->SetLineColor(kBlue-4);
nomfsi->SetLineColor(kBlack);
twofsi->SetLineColor(kViolet-4);
fourfsi->SetLineColor(kGreen-6);
sixfsi->SetLineColor(kRed-3);
eightfsi->SetLineColor(kCyan+2);
nofsi->SetLineStyle(2);
nomfsi->SetLineStyle(1);
twofsi->SetLineStyle(1);
fourfsi->SetLineStyle(1);
sixfsi->SetLineStyle(1);
eightfsi->SetLineStyle(1);
leg = new TLegend(0.35,0.3,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->SetHeader("NEUT, CH, RFG+RPA, BBBA05");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"lf");
leg->AddEntry(nomfsi,Form("1 #times FSI, #chi^{2}=%.1f",nomfsiChi2),"lf");
leg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"lf");
leg->AddEntry(fourfsi,Form("4 #times FSI, #chi^{2}=%.1f",fourfsiChi2),"lf");
leg->AddEntry(sixfsi,Form("6 #times FSI, #chi^{2}=%.1f",sixfsiChi2),"lf");
leg->AddEntry(eightfsi,Form("8 #times FSI, #chi^{2}=%.1f",eightfsiChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_RFGRPA_FSICompShapeOnly.png");
c1->SaveAs("${1}/plotsOut/${var}_RFGRPA_FSICompShapeOnly.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_RFGRPA_FSICompShapeOnly.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") nomfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") nomfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") twofsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") twofsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") fourfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") fourfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") sixfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") sixfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") eightfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") eightfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(0.001,0.5);
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_RFGRPA_FSICompShapeOnly.png");
c2->SaveAs("${1}/plotsOut/log/${var}_RFGRPA_FSICompShapeOnly.pdf");
EOF
cd $1
done;
#Do FSI comparrison (xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_SF_NoFSI_out.root neut5322_SF_out.root neut5322_SF_ExtraFSI_out.root neut5322_SF_4ExtraFSI_out.root neut5322_SF_6ExtraFSI_out.root neut5322_SF_8ExtraFSI_out.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* nofsi = new TH1D();
TH1D* nomfsi = new TH1D();
TH1D* twofsi = new TH1D();
TH1D* fourfsi = new TH1D();
TH1D* sixfsi = new TH1D();
TH1D* eightfsi = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
nofsi =  (TH1D*) _file0->Get("nuisMC");
nomfsi =  (TH1D*) _file1->Get("nuisMC");
twofsi =  (TH1D*) _file2->Get("nuisMC");
fourfsi =  (TH1D*) _file3->Get("nuisMC");
sixfsi =  (TH1D*) _file4->Get("nuisMC");
eightfsi =  (TH1D*) _file5->Get("nuisMC");
_file0->cd(); double nofsiChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double nomfsiChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double twofsiChi2 = chi2Hist->GetBinContent(5);
_file3->cd(); double fourfsiChi2 = chi2Hist->GetBinContent(5);
_file4->cd(); double sixfsiChi2 = chi2Hist->GetBinContent(5);
_file5->cd(); double eightfsiChi2 = chi2Hist->GetBinContent(5);
nofsi->SetLineColor(kBlue-4);
nomfsi->SetLineColor(kBlack);
twofsi->SetLineColor(kViolet-4);
fourfsi->SetLineColor(kGreen-6);
sixfsi->SetLineColor(kRed-3);
eightfsi->SetLineColor(kCyan+2);
nofsi->SetLineStyle(2);
nomfsi->SetLineStyle(1);
twofsi->SetLineStyle(1);
fourfsi->SetLineStyle(1);
sixfsi->SetLineStyle(1);
eightfsi->SetLineStyle(1);
leg = new TLegend(0.35,0.3,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->SetHeader("NEUT, CH, SF, BBBA05");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"lf");
leg->AddEntry(nomfsi,Form("1 #times FSI, #chi^{2}=%.1f",nomfsiChi2),"lf");
leg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"lf");
leg->AddEntry(fourfsi,Form("4 #times FSI, #chi^{2}=%.1f",fourfsiChi2),"lf");
leg->AddEntry(sixfsi,Form("6 #times FSI, #chi^{2}=%.1f",sixfsiChi2),"lf");
leg->AddEntry(eightfsi,Form("8 #times FSI, #chi^{2}=%.1f",eightfsiChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_FSIComp.png");
c1->SaveAs("${1}/plotsOut/${var}_FSIComp.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_FSIComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") nomfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") nomfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") twofsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") twofsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") fourfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") fourfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sixfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") sixfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") eightfsi->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") eightfsi->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,15E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_FSIComp.png");
c2->SaveAs("${1}/plotsOut/log/${var}_FSIComp.pdf");
EOF
cd $1
done;
#Do FSI comparrison (shapeOnly):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_SF_NoFSI_out.root neut5322_SF_out.root neut5322_SF_ExtraFSI_out.root neut5322_SF_4ExtraFSI_out.root neut5322_SF_6ExtraFSI_out.root neut5322_SF_8ExtraFSI_out.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
TH1D* nofsi = new TH1D();
TH1D* nomfsi = new TH1D();
TH1D* twofsi = new TH1D();
TH1D* fourfsi = new TH1D();
TH1D* sixfsi = new TH1D();
TH1D* eightfsi = new TH1D();
result =  (TH1D*) _file0->Get("Result_shapeOnly");
nofsi =  (TH1D*) _file0->Get("nuisMC_shapeOnly");
nomfsi =  (TH1D*) _file1->Get("nuisMC_shapeOnly");
twofsi =  (TH1D*) _file2->Get("nuisMC_shapeOnly");
fourfsi =  (TH1D*) _file3->Get("nuisMC_shapeOnly");
sixfsi =  (TH1D*) _file4->Get("nuisMC_shapeOnly");
eightfsi =  (TH1D*) _file5->Get("nuisMC_shapeOnly");
_file0->cd(); double nofsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file1->cd(); double nomfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file2->cd(); double twofsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file3->cd(); double fourfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file4->cd(); double sixfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file5->cd(); double eightfsiChi2 = chi2Hist_shapeOnly->GetBinContent(5);
nofsi->SetLineColor(kBlue-4);
nomfsi->SetLineColor(kBlack);
twofsi->SetLineColor(kViolet-4);
fourfsi->SetLineColor(kGreen-6);
sixfsi->SetLineColor(kRed-3);
eightfsi->SetLineColor(kCyan+2);
nofsi->SetLineStyle(2);
nomfsi->SetLineStyle(1);
twofsi->SetLineStyle(1);
fourfsi->SetLineStyle(1);
sixfsi->SetLineStyle(1);
eightfsi->SetLineStyle(1);
leg = new TLegend(0.35,0.3,0.85,0.85);
if("${var}"=="datOut") {leg = new TLegend(0.2,0.6,0.85,0.85); leg->SetNColumns(2);}
leg->SetHeader("NEUT, CH, SF, BBBA05");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(nofsi,Form("No FSI, #chi^{2}=%.1f",nofsiChi2),"lf");
leg->AddEntry(nomfsi,Form("1 #times FSI, #chi^{2}=%.1f",nomfsiChi2),"lf");
leg->AddEntry(twofsi,Form("2 #times FSI, #chi^{2}=%.1f",twofsiChi2),"lf");
leg->AddEntry(fourfsi,Form("4 #times FSI, #chi^{2}=%.1f",fourfsiChi2),"lf");
leg->AddEntry(sixfsi,Form("6 #times FSI, #chi^{2}=%.1f",sixfsiChi2),"lf");
leg->AddEntry(eightfsi,Form("8 #times FSI, #chi^{2}=%.1f",eightfsiChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_FSICompShapeOnly.png");
c1->SaveAs("${1}/plotsOut/${var}_FSICompShapeOnly.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_FSICompShapeOnly.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") nofsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") nofsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") nomfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") nomfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") twofsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") twofsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") fourfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") fourfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") sixfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") sixfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") eightfsi->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") eightfsi->GetYaxis()->SetRangeUser(0.001,0.5);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.001,0.4);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(0.001,0.5);
nofsi->Draw();
nomfsi->Draw("same");
twofsi->Draw("same");
fourfsi->Draw("same");
sixfsi->Draw("same");
eightfsi->Draw("same");
result->Draw("same")
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_FSICompShapeOnly.png");
c2->SaveAs("${1}/plotsOut/log/${var}_FSICompShapeOnly.pdf");
EOF
cd $1
done;
#Do NuWro model comp (full xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l nuwro11_RFGRPA.root nuwro11_LFG.root nuwro11_SF.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
TH1D* rfgrpa = new TH1D();
TH1D* lfg = new TH1D();
TH1D* sf = new TH1D();
rfgrpa =  (TH1D*) _file0->Get("nuisMC");
lfg =  (TH1D*) _file1->Get("nuisMC");
sf =  (TH1D*) _file2->Get("nuisMC");
_file0->cd(); double rfgrpaChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double lfgChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double sfChi2 = chi2Hist->GetBinContent(5);
rfgrpa->SetLineColor(kRed-3);
lfg->SetLineColor(kBlue-4);
sf->SetLineColor(kGreen-6);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") leg = new TLegend(0.2,0.6,0.55,0.85);
leg->SetHeader("NuWro11, CH");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(rfgrpa,Form("RFG+RPA(non rel.), #chi^{2}=%.1f",rfgrpaChi2),"lf");
leg->AddEntry(lfg,Form("LFG, #chi^{2}=%.1f",lfgChi2),"lf");
leg->AddEntry(sf,Form("SF, #chi^{2}=%.1f",sfChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
rfgrpa->Draw();
lfg->Draw("same");
sf->Draw("same");
result->Draw("same");
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_NuWroModelComp.png");
c1->SaveAs("${1}/plotsOut/${var}_NuWroModelComp.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_NuWroModelComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") lfg->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") lfg->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
rfgrpa->Draw();
lfg->Draw("same");
sf->Draw("same");
result->Draw("same");
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_NuWroModelComp.png");
c2->SaveAs("${1}/plotsOut/log/${var}_NuWroModelComp.pdf");
EOF
cd $1
done;
#Do NuWro model comp (shapeOnly):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l nuwro11_RFGRPA.root nuwro11_LFG.root nuwro11_SF.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
result =  (TH1D*) _file0->Get("Result_shapeOnly");
TH1D* rfgrpa = new TH1D();
TH1D* lfg = new TH1D();
TH1D* sf = new TH1D();
rfgrpa =  (TH1D*) _file0->Get("nuisMC_shapeOnly");
lfg =  (TH1D*) _file1->Get("nuisMC_shapeOnly");
sf =  (TH1D*) _file2->Get("nuisMC_shapeOnly");;
_file0->cd(); double rfgrpaChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file1->cd(); double lfgChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file2->cd(); double sfChi2 = chi2Hist_shapeOnly->GetBinContent(5);
rfgrpa->SetLineColor(kRed-3);
lfg->SetLineColor(kBlue-4);
sf->SetLineColor(kGreen-6);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") leg = new TLegend(0.2,0.6,0.55,0.85);
leg->SetHeader("NuWro11, CH");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(rfgrpa,Form("RFG+RPA(non rel.), #chi^{2}=%.1f",rfgrpaChi2),"lf");
leg->AddEntry(lfg,Form("LFG, #chi^{2}=%.1f",lfgChi2),"lf");
leg->AddEntry(sf,Form("SF, #chi^{2}=%.1f",sfChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
rfgrpa->Draw();
lfg->Draw("same");
sf->Draw("same");
result->Draw("same");
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_NuWroModelCompShapeOnly.png");
c1->SaveAs("${1}/plotsOut/${var}_NuWroModelCompShapeOnly.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_NuWroModelCompShapeOnly.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") lfg->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") lfg->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") sf->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") sf->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(0.002,0.5);
rfgrpa->Draw();
lfg->Draw("same");
sf->Draw("same");
result->Draw("same");
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_NuWroModelCompShapeOnly.png");
c2->SaveAs("${1}/plotsOut/log/${var}_NuWroModelCompShapeOnly.pdf");
EOF
cd $1
done;

#Do NEUT model comp (full xsec):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_RFG_RPARW_out.root neut5322_RFG_noRPA_out.root neut5322_SF_out.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
result =  (TH1D*) _file0->Get("Result_Xsec");
TH1D* rfgrpa = new TH1D();
TH1D* rfg = new TH1D();
TH1D* sf = new TH1D();
rfgrpa =  (TH1D*) _file0->Get("nuisMC");
rfg =  (TH1D*) _file1->Get("nuisMC");
sf =  (TH1D*) _file2->Get("nuisMC");
_file0->cd(); double rfgrpaChi2 = chi2Hist->GetBinContent(5);
_file1->cd(); double rfgChi2 = chi2Hist->GetBinContent(5);
_file2->cd(); double sfChi2 = chi2Hist->GetBinContent(5);
rfgrpa->SetLineColor(kRed-3);
rfg->SetLineColor(kBlue-4);
sf->SetLineColor(kGreen-6);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") leg = new TLegend(0.2,0.6,0.55,0.85);
leg->SetHeader("NEUT 5322, CH");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(rfgrpa,Form("RFG+RPA, #chi^{2}=%.1f",rfgrpaChi2),"lf");
leg->AddEntry(rfg,Form("RFG, #chi^{2}=%.1f",rfgChi2),"lf");
leg->AddEntry(sf,Form("SF, #chi^{2}=%.1f",sfChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
rfgrpa->Draw();
rfg->Draw("same");
sf->Draw("same");
result->Draw("same");
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_NeutModelComp.png");
c1->SaveAs("${1}/plotsOut/${var}_NeutModelComp.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_NeutModelComp.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") rfg->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") rfg->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") sf->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") sf->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.05E-39,12E-39);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(30.0E-42,8E-39);
rfgrpa->Draw();
rfg->Draw("same");
sf->Draw("same");
result->Draw("same");
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_NeutModelComp.png");
c2->SaveAs("${1}/plotsOut/log/${var}_NeutModelComp.pdf");
EOF
cd $1
done;
#Do NEUT model comp (shapeOnly):
for var in dptOut datOut dphitOut; do
  cd $var
root -b -l neut5322_RFG_RPARW_out.root neut5322_RFG_noRPA_out.root neut5322_SF_out.root<<EOF
TCanvas* c1 = new TCanvas();
TH1D* result = new TH1D();
result =  (TH1D*) _file0->Get("Result_shapeOnly");
TH1D* rfgrpa = new TH1D();
TH1D* rfg = new TH1D();
TH1D* sf = new TH1D();
rfgrpa =  (TH1D*) _file0->Get("nuisMC_shapeOnly");
rfg =  (TH1D*) _file1->Get("nuisMC_shapeOnly");
sf =  (TH1D*) _file2->Get("nuisMC_shapeOnly");;
_file0->cd(); double rfgrpaChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file1->cd(); double rfgChi2 = chi2Hist_shapeOnly->GetBinContent(5);
_file2->cd(); double sfChi2 = chi2Hist_shapeOnly->GetBinContent(5);
rfgrpa->SetLineColor(kRed-3);
rfg->SetLineColor(kBlue-4);
sf->SetLineColor(kGreen-6);
leg = new TLegend(0.3,0.5,0.85,0.85);
if("${var}"=="datOut") leg = new TLegend(0.2,0.6,0.55,0.85);
leg->SetHeader("NEUT 5322, CH");
leg->AddEntry(result,"T2K","lep");
leg->AddEntry(rfgrpa,Form("RFG+RPA, #chi^{2}=%.1f",rfgrpaChi2),"lf");
leg->AddEntry(rfg,Form("RFG, #chi^{2}=%.1f",rfgChi2),"lf");
leg->AddEntry(sf,Form("SF, #chi^{2}=%.1f",sfChi2),"lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
TLatex latex;
latex.SetTextSize(0.05);
//result->Draw()
rfgrpa->Draw();
rfg->Draw("same");
sf->Draw("same");
result->Draw("same");
leg->Draw("same");
c1->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${1}/plotsOut/${var}_NeutModelCompShapeOnly.png");
c1->SaveAs("${1}/plotsOut/${var}_NeutModelCompShapeOnly.pdf");
c1->SaveAs("${1}/plotsOut/rootfiles/${var}_NeutModelCompShapeOnly.root");
TCanvas* c2 = new TCanvas();
if("${var}"=="dptOut") rfgrpa->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") rfgrpa->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") rfg->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") rfg->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") sf->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") sf->GetYaxis()->SetRangeUser(0.002,0.5);
if("${var}"=="dptOut") result->GetYaxis()->SetRangeUser(0.003,0.4);
if("${var}"=="dphitOut") result->GetYaxis()->SetRangeUser(0.002,0.5);
rfgrpa->Draw();
rfg->Draw("same");
sf->Draw("same");
result->Draw("same");
//leg->Draw("same");
gPad->SetLogy();
c2->Update();
latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c2->SaveAs("${1}/plotsOut/log/${var}_NeutModelCompShapeOnly.png");
c2->SaveAs("${1}/plotsOut/log/${var}_NeutModelCompShapeOnly.pdf");
EOF
cd $1
done;
