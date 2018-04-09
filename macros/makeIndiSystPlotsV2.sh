#!/bin/sh
var=$1
root -b -l ${var}_systmode0_fitOut.root ${var}_systmode1_fitOut.root ${var}_systmode2_fitOut.root ${var}_systmode3_fitOut.root ${var}_systmode4_fitOut.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* noSyst = new TH1D();
TH1D* allSyst = new TH1D();
TH1D* fluxSyst = new TH1D();
TH1D* detSyst = new TH1D();
TH1D* modelSyst = new TH1D();
noSyst =  (TH1D*) _file0->Get("paramerrhist_parpar_ccqe_result");
allSyst =  (TH1D*) _file1->Get("paramerrhist_parpar_ccqe_result");
fluxSyst =  (TH1D*) _file2->Get("paramerrhist_parpar_ccqe_result");
detSyst =  (TH1D*) _file3->Get("paramerrhist_parpar_ccqe_result");
modelSyst =  (TH1D*) _file4->Get("paramerrhist_parpar_ccqe_result");
for(int i=1; i<noSyst->GetNbinsX(); i++){
  noSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
  allSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
  fluxSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
  detSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
  modelSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
}
noSyst->UseCurrentStyle();
allSyst->UseCurrentStyle();
fluxSyst->UseCurrentStyle();
detSyst->UseCurrentStyle();
modelSyst->UseCurrentStyle();
noSyst->SetLineColor(kCyan+2);
allSyst->SetLineColor(kBlack);
fluxSyst->SetLineColor(kRed-3);
detSyst->SetLineColor(kViolet-4);
modelSyst->SetLineColor(kGreen-6);
noSyst->SetLineStyle(2);
allSyst->SetLineStyle(2);
fluxSyst->SetLineStyle(1);
detSyst->SetLineStyle(1);
modelSyst->SetLineStyle(1);
noSyst->SetLineWidth(4);
allSyst->SetLineWidth(4);
fluxSyst->SetLineWidth(2);
detSyst->SetLineWidth(2);
modelSyst->SetLineWidth(2);
noSyst->SetMarkerStyle(1);
allSyst->SetMarkerStyle(1);
fluxSyst->SetMarkerStyle(1);
detSyst->SetMarkerStyle(1);
modelSyst->SetMarkerStyle(1);
leg = new TLegend(0.25,0.18,0.75,0.43);
leg->AddEntry(noSyst,"c_{i} Only","lf");
leg->AddEntry(fluxSyst,"c_{i} + Flux","lf");
leg->AddEntry(detSyst,"c_{i} + Detector","lf");
leg->AddEntry(modelSyst,"c_{i} + Model","lf");
leg->AddEntry(allSyst,"All Parmeters","lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
//TLatex latex;
//latex.SetTextSize(0.05);
allSyst->GetXaxis()->SetRangeUser(0,8);
allSyst->GetYaxis()->SetRangeUser(0,0.25);
allSyst->GetYaxis()->SetTitle("Relative Error");
if("${var}"=="dpt") allSyst->GetXaxis()->SetTitle("#deltap_{T} Analysis Bin");
if("${var}"=="dphit") allSyst->GetXaxis()->SetTitle("#delta#phi_{T} Analysis Bin");
if("${var}"=="dat") allSyst->GetXaxis()->SetTitle("#delta#alpha_{T} Analysis Bin");
allSyst->Draw()
noSyst->Draw("same");
fluxSyst->Draw("same");
detSyst->Draw("same");
modelSyst->Draw("same");
leg->Draw("same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${var}_inidiSystPlot.png");
EOF

root -b -l ${var}_systmode0_fitOut.root ${var}_systmode2_fitOut.root ${var}_systmode3_fitOut.root ${var}_systmode4_fitOut.root ${var}_systmode1_fitOut.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* noSyst = new TH1D();
TH1D* fluxSyst = new TH1D();
TH1D* detSyst = new TH1D();
TH1D* modelSyst = new TH1D();
TH1D* allSyst = new TH1D();
noSyst =  (TH1D*) _file0->Get("paramerrhist_parpar_ccqe_result");
fluxSyst =  (TH1D*) _file1->Get("paramerrhist_parpar_ccqe_result");
detSyst =  (TH1D*) _file2->Get("paramerrhist_parpar_ccqe_result");
modelSyst =  (TH1D*) _file3->Get("paramerrhist_parpar_ccqe_result");
allSyst =  (TH1D*) _file4->Get("paramerrhist_parpar_ccqe_result");
TH1D* allSyst_Only = new TH1D("allSyst_Only", "allSyst_Only", noSyst->GetNbinsX(), 0, noSyst->GetNbinsX());
//
fluxSyst->UseCurrentStyle();
detSyst->UseCurrentStyle();
modelSyst->UseCurrentStyle();
fluxSyst->SetLineColor(kRed-3);
detSyst->SetLineColor(kViolet-4);
modelSyst->SetLineColor(kGreen-6);
fluxSyst->SetLineStyle(2);
detSyst->SetLineStyle(2);
modelSyst->SetLineStyle(2);
fluxSyst->SetLineWidth(2);
detSyst->SetLineWidth(2);
modelSyst->SetLineWidth(2);
fluxSyst->SetMarkerStyle(1);
detSyst->SetMarkerStyle(1);
modelSyst->SetMarkerStyle(1);
//
allSyst_Only->UseCurrentStyle();
allSyst_Only->SetLineColor(kCyan+2);
allSyst_Only->SetLineStyle(1);
allSyst_Only->SetLineWidth(3);
allSyst_Only->SetMarkerStyle(1);
allSyst->UseCurrentStyle();
allSyst->SetLineColor(kBlack);
allSyst->SetLineStyle(1);
allSyst->SetLineWidth(3);
allSyst->SetMarkerStyle(1);
noSyst->UseCurrentStyle();
noSyst->SetLineColor(kBlue);
noSyst->SetLineStyle(1);
noSyst->SetLineWidth(3);
noSyst->SetMarkerStyle(1);
leg = new TLegend(0.2,0.57,0.85,0.85);
leg->AddEntry(allSyst,"Total Uncertanty","lf");
leg->AddEntry(noSyst,"c_{i} Only","lf");
leg->AddEntry(allSyst_Only,"Sys. Only","lf");
leg->AddEntry(fluxSyst,"  Flux","lf");
leg->AddEntry(detSyst,"  Detector","lf");
leg->AddEntry(modelSyst,"  Model","lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
//TLatex latex;
//latex.SetTextSize(0.05);
allSyst_Only->GetXaxis()->SetRangeUser(0,8);
allSyst_Only->GetYaxis()->SetRangeUser(0,0.40);
allSyst_Only->GetYaxis()->SetTitle("Extrapolated Error Contrib.");
if("${var}"=="dpt") allSyst_Only->GetXaxis()->SetTitle("#deltap_{T} Analysis Bin");
if("${var}"=="dphit") allSyst_Only->GetXaxis()->SetTitle("#delta#phi_{T} Analysis Bin");
if("${var}"=="dat") allSyst_Only->GetXaxis()->SetTitle("#delta#alpha_{T} Analysis Bin");
//
#include <cmath>
for(int i=1; i<10; i++){
  double baseEr = noSyst->GetBinContent(i);
  double fluxEr = fluxSyst->GetBinContent(i);
  double detEr = detSyst->GetBinContent(i);
  double modelEr = modelSyst->GetBinContent(i);
  double allEr = allSyst->GetBinContent(i);
  //cout << "Bin: " << i-1 << endl;
  //cout << "baseEr: " << baseEr << endl;
  //cout << "fluxEr: " << fluxEr << endl;
  //cout << "detEr: " << detEr << endl;
  //cout << "modelEr: " << modelEr << endl;
  fluxSyst->SetBinContent(i, sqrt(abs(fluxEr**2 - baseEr**2)));
  detSyst->SetBinContent(i, sqrt(abs(detEr**2 - baseEr**2)));
  modelSyst->SetBinContent(i, sqrt(abs(modelEr**2 - baseEr**2)));
  allSyst_Only->SetBinContent(i, sqrt(abs(allEr**2 - baseEr**2)));
  //cout << "fluxEr Contrib: " << sqrt(abs(fluxEr**2 - baseEr**2)) << endl;
  //cout << "detEr Contrib: " << sqrt(abs(detEr**2 - baseEr**2)) << endl;
  //cout << "modelEr Contrib: " << sqrt(abs(modelEr**2 - baseEr**2)) << endl;
}
//
allSyst_Only->Draw();
allSyst->Draw("same");
noSyst->Draw("same");
fluxSyst->Draw("same");
detSyst->Draw("same");
modelSyst->Draw("same");
leg->Draw("same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${var}_inidiSystContribPlot.png");
EOF

#!/bin/sh
var=$1
root -b -l ${var}_systmode0_fitOut_allStats.root ${var}_systmode1_fitOut_allStats.root ${var}_systmode2_fitOut_allStats.root ${var}_systmode3_fitOut_allStats.root ${var}_systmode4_fitOut_allStats.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* noSyst = new TH1D();
TH1D* allSyst = new TH1D();
TH1D* fluxSyst = new TH1D();
TH1D* detSyst = new TH1D();
TH1D* modelSyst = new TH1D();
noSyst =  (TH1D*) _file0->Get("paramerrhist_parpar_ccqe_result");
allSyst =  (TH1D*) _file1->Get("paramerrhist_parpar_ccqe_result");
fluxSyst =  (TH1D*) _file2->Get("paramerrhist_parpar_ccqe_result");
detSyst =  (TH1D*) _file3->Get("paramerrhist_parpar_ccqe_result");
modelSyst =  (TH1D*) _file4->Get("paramerrhist_parpar_ccqe_result");
for(int i=1; i<noSyst->GetNbinsX(); i++){
  noSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
  allSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
  fluxSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
  detSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
  modelSyst->GetXaxis()->SetBinLabel(i, Form("%d",i));
}
noSyst->UseCurrentStyle();
allSyst->UseCurrentStyle();
fluxSyst->UseCurrentStyle();
detSyst->UseCurrentStyle();
modelSyst->UseCurrentStyle();
noSyst->SetLineColor(kCyan+2);
allSyst->SetLineColor(kBlack);
fluxSyst->SetLineColor(kRed-3);
detSyst->SetLineColor(kViolet-4);
modelSyst->SetLineColor(kGreen-6);
noSyst->SetLineStyle(2);
allSyst->SetLineStyle(2);
fluxSyst->SetLineStyle(1);
detSyst->SetLineStyle(1);
modelSyst->SetLineStyle(1);
noSyst->SetLineWidth(4);
allSyst->SetLineWidth(4);
fluxSyst->SetLineWidth(2);
detSyst->SetLineWidth(2);
modelSyst->SetLineWidth(2);
noSyst->SetMarkerStyle(1);
allSyst->SetMarkerStyle(1);
fluxSyst->SetMarkerStyle(1);
detSyst->SetMarkerStyle(1);
modelSyst->SetMarkerStyle(1);
leg = new TLegend(0.25,0.18,0.75,0.43);
leg->AddEntry(noSyst,"c_{i} Only","lf");
leg->AddEntry(fluxSyst,"c_{i} + Flux","lf");
leg->AddEntry(detSyst,"c_{i} + Detector","lf");
leg->AddEntry(modelSyst,"c_{i} + Model","lf");
leg->AddEntry(allSyst,"All Parmeters","lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
//TLatex latex;
//latex.SetTextSize(0.05);
allSyst->GetXaxis()->SetRangeUser(0,8);
allSyst->GetYaxis()->SetRangeUser(0,0.25);
allSyst->GetYaxis()->SetTitle("Relative Error");
if("${var}"=="dpt") allSyst->GetXaxis()->SetTitle("#deltap_{T} Analysis Bin");
if("${var}"=="dphit") allSyst->GetXaxis()->SetTitle("#delta#phi_{T} Analysis Bin");
if("${var}"=="dat") allSyst->GetXaxis()->SetTitle("#delta#alpha_{T} Analysis Bin");
allSyst->Draw()
noSyst->Draw("same");
fluxSyst->Draw("same");
detSyst->Draw("same");
modelSyst->Draw("same");
leg->Draw("same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${var}_inidiSystPlot_allStats.png");
EOF

root -b -l ${var}_systmode0_fitOut_allStats.root ${var}_systmode2_fitOut_allStats.root ${var}_systmode3_fitOut_allStats.root ${var}_systmode4_fitOut_allStats.root ${var}_systmode1_fitOut_allStats.root <<EOF
TCanvas* c1 = new TCanvas();
TH1D* noSyst = new TH1D();
TH1D* fluxSyst = new TH1D();
TH1D* detSyst = new TH1D();
TH1D* modelSyst = new TH1D();
TH1D* allSyst = new TH1D();
noSyst =  (TH1D*) _file0->Get("paramerrhist_parpar_ccqe_result");
fluxSyst =  (TH1D*) _file1->Get("paramerrhist_parpar_ccqe_result");
detSyst =  (TH1D*) _file2->Get("paramerrhist_parpar_ccqe_result");
modelSyst =  (TH1D*) _file3->Get("paramerrhist_parpar_ccqe_result");
allSyst =  (TH1D*) _file4->Get("paramerrhist_parpar_ccqe_result");
TH1D* allSyst_Only = new TH1D("allSyst_Only", "allSyst_Only", noSyst->GetNbinsX(), 0, noSyst->GetNbinsX());
//
fluxSyst->UseCurrentStyle();
detSyst->UseCurrentStyle();
modelSyst->UseCurrentStyle();
fluxSyst->SetLineColor(kRed-3);
detSyst->SetLineColor(kViolet-4);
modelSyst->SetLineColor(kGreen-6);
fluxSyst->SetLineStyle(2);
detSyst->SetLineStyle(2);
modelSyst->SetLineStyle(2);
fluxSyst->SetLineWidth(2);
detSyst->SetLineWidth(2);
modelSyst->SetLineWidth(2);
fluxSyst->SetMarkerStyle(1);
detSyst->SetMarkerStyle(1);
modelSyst->SetMarkerStyle(1);
//
allSyst_Only->UseCurrentStyle();
allSyst_Only->SetLineColor(kCyan+2);
allSyst_Only->SetLineStyle(1);
allSyst_Only->SetLineWidth(3);
allSyst_Only->SetMarkerStyle(1);
allSyst->UseCurrentStyle();
allSyst->SetLineColor(kBlack);
allSyst->SetLineStyle(1);
allSyst->SetLineWidth(3);
allSyst->SetMarkerStyle(1);
noSyst->UseCurrentStyle();
noSyst->SetLineColor(kBlue);
noSyst->SetLineStyle(1);
noSyst->SetLineWidth(3);
noSyst->SetMarkerStyle(1);
leg = new TLegend(0.2,0.57,0.85,0.85);
leg->AddEntry(allSyst,"Total Uncertanty","lf");
leg->AddEntry(noSyst,"c_{i} Only","lf");
leg->AddEntry(allSyst_Only,"Sys. Only","lf");
leg->AddEntry(fluxSyst,"  Flux","lf");
leg->AddEntry(detSyst,"  Detector","lf");
leg->AddEntry(modelSyst,"  Model","lf");
leg->SetFillColor(kWhite);
leg->SetFillStyle(0);
//TLatex latex;
//latex.SetTextSize(0.05);
allSyst_Only->GetXaxis()->SetRangeUser(0,8);
allSyst_Only->GetYaxis()->SetRangeUser(0,0.40);
allSyst_Only->GetYaxis()->SetTitle("Extrapolated Error Contrib.");
if("${var}"=="dpt") allSyst_Only->GetXaxis()->SetTitle("#deltap_{T} Analysis Bin");
if("${var}"=="dphit") allSyst_Only->GetXaxis()->SetTitle("#delta#phi_{T} Analysis Bin");
if("${var}"=="dat") allSyst_Only->GetXaxis()->SetTitle("#delta#alpha_{T} Analysis Bin");
//
#include <cmath>
for(int i=1; i<10; i++){
  double baseEr = noSyst->GetBinContent(i);
  double fluxEr = fluxSyst->GetBinContent(i);
  double detEr = detSyst->GetBinContent(i);
  double modelEr = modelSyst->GetBinContent(i);
  double allEr = allSyst->GetBinContent(i);
  //cout << "Bin: " << i-1 << endl;
  //cout << "baseEr: " << baseEr << endl;
  //cout << "fluxEr: " << fluxEr << endl;
  //cout << "detEr: " << detEr << endl;
  //cout << "modelEr: " << modelEr << endl;
  fluxSyst->SetBinContent(i, sqrt(abs(fluxEr**2 - baseEr**2)));
  detSyst->SetBinContent(i, sqrt(abs(detEr**2 - baseEr**2)));
  modelSyst->SetBinContent(i, sqrt(abs(modelEr**2 - baseEr**2)));
  allSyst_Only->SetBinContent(i, sqrt(abs(allEr**2 - baseEr**2)));
  //cout << "fluxEr Contrib: " << sqrt(abs(fluxEr**2 - baseEr**2)) << endl;
  //cout << "detEr Contrib: " << sqrt(abs(detEr**2 - baseEr**2)) << endl;
  //cout << "modelEr Contrib: " << sqrt(abs(modelEr**2 - baseEr**2)) << endl;
}
//
allSyst_Only->Draw();
allSyst->Draw("same");
noSyst->Draw("same");
fluxSyst->Draw("same");
detSyst->Draw("same");
modelSyst->Draw("same");
leg->Draw("same");
c1->Update();
//latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
c1->SaveAs("${var}_inidiSystContribPlot_allStats.png");
EOF