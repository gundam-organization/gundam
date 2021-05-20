/******************************************************

Code to convert Pierre's CC0pi+1p bins into the input
format required by NUISNACE

Author: Stephen Dolan
Date Created: October 2017

******************************************************/

#include <iostream> 
#include <iomanip>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <assert.h>

#include <TCanvas.h>
#include <TH1F.h>
#include <TH2D.h>
#include <TH2Poly.h>
#include <TVectorD.h>
#include <TMatrixD.h>
#include <TGraph.h>
#include <TTree.h>
#include <TString.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TMath.h>
#include <TStyle.h>
#include <TPaveText.h>
#include <TLegend.h>
#include <TGaxis.h>

#include <math.h>
#include <TRandom3.h>
#include <TMatrixT.h>
#include <TMatrixTSym.h>
#include <TVectorT.h>
#include <TDecompChol.h>
#include <TDecompBK.h>
#include <TDecompSVD.h>
#include <TLatex.h>
#include <vector>


void multiComp_infKine_4comp(TString inFileName1, TString inFileName2, TString inFileName3, TString inFileName4, char* genName1, char* genName2, char* genName3, char* genName4, TString outFileTag,  TString varName, bool ignoreFour=false){

  TFile *inFile1 = new TFile(inFileName1.Data());
  TFile *inFile2 = new TFile(inFileName2.Data());
  TFile *inFile3 = new TFile(inFileName3.Data());
  TFile *inFile4 = new TFile(inFileName4.Data());


  //******** Collect inputs ******

  TH1D* data_LinResult    = new TH1D();
  TH1D* result1_LinResult = new TH1D();
  TH1D* result2_LinResult = new TH1D();
  TH1D* result3_LinResult = new TH1D();
  TH1D* result4_LinResult = new TH1D();

  cout << "Collecting histos with name: " << Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_MC",varName.Data()) << endl;

  data_LinResult    = (TH1D*) inFile1->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_data",varName.Data()));
  result1_LinResult = (TH1D*) inFile1->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_MC",varName.Data()));
  result2_LinResult = (TH1D*) inFile2->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_MC",varName.Data()));
  result3_LinResult = (TH1D*) inFile3->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_MC",varName.Data()));
  result4_LinResult = (TH1D*) inFile4->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_MC",varName.Data()));

  std::vector<TH1D*> data_slice;
  std::vector<TH1D*> result1_slice;
  std::vector<TH1D*> result2_slice;
  std::vector<TH1D*> result3_slice;
  std::vector<TH1D*> result4_slice;

  int nBins = 7;

  for (int i = 0; i < nBins; i++){ 
    data_slice.push_back(new TH1D);
    result1_slice.push_back(new TH1D);
    result2_slice.push_back(new TH1D);
    result3_slice.push_back(new TH1D);
    result4_slice.push_back(new TH1D);

    data_slice[i]    = (TH1D*) inFile1->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_data_Slice%d",varName.Data(),i));
    result1_slice[i] = (TH1D*) inFile1->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_MC_Slice%d",varName.Data(),i));
    result2_slice[i] = (TH1D*) inFile2->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_MC_Slice%d",varName.Data(),i));
    result3_slice[i] = (TH1D*) inFile3->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_MC_Slice%d",varName.Data(),i));
    result4_slice[i] = (TH1D*) inFile4->Get(Form("T2K_CC0pinp_ifk_XSec_3Dinf%s_nu_MC_Slice%d",varName.Data(),i));
  }

  const int nbins_p  = 7;
  const int nbins_a  = 5;
  const int nbins_tp = 7;

  int nBinsInfKine = 0;

  if(strstr(varName.Data(),"p"))  nBinsInfKine = nbins_p;
  if(strstr(varName.Data(),"ip")) nBinsInfKine = nbins_tp;
  if(strstr(varName.Data(),"a"))  nBinsInfKine = nbins_a;

  //Modified bin sizes for pretty plotting
  double bins_p[nbins_p+1]   = {-0.6,-0.3,0,0.1,0.2,0.3,0.5,1.5};
  double bins_a[nbins_a+1]   = {-30,-5. , 5., 10., 20., 80};
  double bins_tp[nbins_tp+1] = {0, 0.3, 0.4, 0.5, 0.6, 0.7,0.9, 1.5};

  TH1D* reBinHist;
  if(strstr(varName.Data(),"p"))  reBinHist = new TH1D("reBinHist","reBinHist",nbins_p,bins_p);
  if(strstr(varName.Data(),"ip")) reBinHist = new TH1D("reBinHist","reBinHist",nbins_tp,bins_tp);
  if(strstr(varName.Data(),"a"))  reBinHist = new TH1D("reBinHist","reBinHist",nbins_a,bins_a);

  for (int i = 0; i < nBins; i++){ 
    for (int j = 0; j < nBinsInfKine; j++){
      reBinHist->SetBinContent(j+1, data_slice[i]->GetBinContent(j+1));
      reBinHist->SetBinError(j+1, data_slice[i]->GetBinError(j+1));
    }
    data_slice[i] = (TH1D*) reBinHist->Clone();

    for (int j = 0; j < nBinsInfKine; j++){
      reBinHist->SetBinContent(j+1, result1_slice[i]->GetBinContent(j+1));
      reBinHist->SetBinError(j+1, result1_slice[i]->GetBinError(j+1));
    }
    result1_slice[i] = (TH1D*) reBinHist->Clone();

    for (int j = 0; j < nBinsInfKine; j++){
      reBinHist->SetBinContent(j+1, result2_slice[i]->GetBinContent(j+1));
      reBinHist->SetBinError(j+1, result2_slice[i]->GetBinError(j+1));
    }
    result2_slice[i] = (TH1D*) reBinHist->Clone();

    for (int j = 0; j < nBinsInfKine; j++){
      reBinHist->SetBinContent(j+1, result3_slice[i]->GetBinContent(j+1));
      reBinHist->SetBinError(j+1, result3_slice[i]->GetBinError(j+1));
    }
    result3_slice[i] = (TH1D*) reBinHist->Clone();

    for (int j = 0; j < nBinsInfKine; j++){
      reBinHist->SetBinContent(j+1, result4_slice[i]->GetBinContent(j+1));
      reBinHist->SetBinError(j+1, result4_slice[i]->GetBinError(j+1));
    }
    result4_slice[i] = (TH1D*) reBinHist->Clone();
  }


  double result1_chi2 = 0.00;
  double result2_chi2 = 0.00;
  double result3_chi2 = 0.00;
  double result4_chi2 = 0.00;

  if(strstr(varName.Data(),"p") && !(strstr(varName.Data(),"i"))) { // Adjustment to DOF to exclude bins that are always 0
    cout << " Working with delta p " << endl;
    result1_chi2 = ((TH1D*)(inFile1->Get("likelihood_hist")))->GetBinContent(1);
    result2_chi2 = ((TH1D*)(inFile2->Get("likelihood_hist")))->GetBinContent(1);
    result3_chi2 = ((TH1D*)(inFile3->Get("likelihood_hist")))->GetBinContent(1);
    result4_chi2 = ((TH1D*)(inFile4->Get("likelihood_hist")))->GetBinContent(1);
  }
  else if(strstr(varName.Data(),"i")) {
    cout << " Working with |delta p| (modulus) " << endl;
    result1_chi2 = ((TH1D*)(inFile1->Get("likelihood_hist")))->GetBinContent(2);
    result2_chi2 = ((TH1D*)(inFile2->Get("likelihood_hist")))->GetBinContent(2);
    result3_chi2 = ((TH1D*)(inFile3->Get("likelihood_hist")))->GetBinContent(2);
    result4_chi2 = ((TH1D*)(inFile4->Get("likelihood_hist")))->GetBinContent(2);
  }
  else if(strstr(varName.Data(),"a")) { // Adjustment to DOF to exclude bins that are always 0
        cout << " Working with delta theta" << endl; 
    result1_chi2 = ((TH1D*)(inFile1->Get("likelihood_hist")))->GetBinContent(3);
    result2_chi2 = ((TH1D*)(inFile2->Get("likelihood_hist")))->GetBinContent(3);
    result3_chi2 = ((TH1D*)(inFile3->Get("likelihood_hist")))->GetBinContent(3);
    result4_chi2 = ((TH1D*)(inFile4->Get("likelihood_hist")))->GetBinContent(3);
  }
  else{
    cout << " Warning: variable " << varName.Data() << " not recognised ... " << endl;
  }

  cout << "Goodness of fit: " << endl;
  cout << genName1 << " " << result1_chi2 << endl;
  cout << genName2 << " " << result2_chi2 << endl;
  cout << genName3 << " " << result3_chi2 << endl;
  cout << genName4 << " " << result4_chi2 << endl << endl;

  //******** Make output plots ******


  cout << "Writting output ... " << endl;

  TFile *outfile = new TFile(Form("%s.root",outFileTag.Data()),"recreate");
  outfile->cd();

  //result1_LinResult->SetLineColor(kViolet-4);
  //result2_LinResult->SetLineColor(kRed-3);
  //result3_LinResult->SetLineColor(kBlack);
  //result3_LinResult->SetLineStyle(2);
  //result4_LinResult->SetLineColor(kGreen-6);

  //result1_LinResult->SetMarkerStyle(1);

  //result1_LinResult->SetLineWidth(3);
  //result2_LinResult->SetLineWidth(3);
  //result3_LinResult->SetLineWidth(3);
  //result3_LinResult->SetLineWidth(3);
  //result4_LinResult->SetLineWidth(3);

  //for (int i = 0; i < nBins; i++){ result1_slice[i]->SetLineColor(kViolet-4); }
  //for (int i = 0; i < nBins; i++){ result2_slice[i]->SetLineColor(kRed-3); }
  //for (int i = 0; i < nBins; i++){ result3_slice[i]->SetLineColor(kBlack); }
  //for (int i = 0; i < nBins; i++){ result3_slice[i]->SetLineStyle(2); }    
  //for (int i = 0; i < nBins; i++){ result4_slice[i]->SetLineColor(kGreen-6); }

  //for (int i = 0; i < nBins; i++){ result1_slice[i]->SetLineWidth(3); }
  //for (int i = 0; i < nBins; i++){ result2_slice[i]->SetLineWidth(3); }
  //for (int i = 0; i < nBins; i++){ result3_slice[i]->SetLineWidth(3); }
  //for (int i = 0; i < nBins; i++){ result3_slice[i]->SetLineWidth(3); }    
  //for (int i = 0; i < nBins; i++){ result4_slice[i]->SetLineWidth(3); }


  result1_LinResult->SetLineColor(kAzure+8);
  result2_LinResult->SetLineColor(kAzure-2);
  result3_LinResult->SetLineColor(kRed-4);
  result4_LinResult->SetLineColor(kRed+3);

  result1_LinResult->SetMarkerStyle(1);

  result2_LinResult->SetLineStyle(2);
  result4_LinResult->SetLineStyle(3);

  result1_LinResult->SetLineWidth(4);
  result2_LinResult->SetLineWidth(3);
  result3_LinResult->SetLineWidth(2);
  result4_LinResult->SetLineWidth(1);

  for (int i = 0; i < nBins; i++){ result1_slice[i]->SetLineColor(kAzure+8); }
  for (int i = 0; i < nBins; i++){ result2_slice[i]->SetLineColor(kAzure-2); }
  for (int i = 0; i < nBins; i++){ result3_slice[i]->SetLineColor(kRed-4); }
  for (int i = 0; i < nBins; i++){ result4_slice[i]->SetLineColor(kRed+3); }

  for (int i = 0; i < nBins; i++){ result2_slice[i]->SetLineStyle(2); }
  for (int i = 0; i < nBins; i++){ result4_slice[i]->SetLineStyle(3); }

  for (int i = 0; i < nBins; i++){ result1_slice[i]->SetLineWidth(4); }
  for (int i = 0; i < nBins; i++){ result2_slice[i]->SetLineWidth(3); }
  for (int i = 0; i < nBins; i++){ result3_slice[i]->SetLineWidth(2); }
  for (int i = 0; i < nBins; i++){ result4_slice[i]->SetLineWidth(2); }

  for (int i = 0; i < nBins; i++){ data_slice[i]->SetMarkerSize(1.5); }

    

  if(strstr(outFileTag.Data(),"FSI")){
    result3_LinResult->SetLineColor(kGray+2);
    result4_LinResult->SetLineColor(kGray+3);
    for (int i = 0; i < nBins; i++){ result3_slice[i]->SetLineColor(kGray+2); }
    for (int i = 0; i < nBins; i++){ result4_slice[i]->SetLineColor(kGray+3); }
  }

  //with chi2

  TLegend* leg = new TLegend(0.05, 0.05, 0.95, 0.95);
  leg->AddEntry(data_slice[0],"T2K Unfolded","ep");
  leg->AddEntry(result1_LinResult, Form("%s, #chi^{2}=%.1f",genName1,result1_chi2), "lf");
  leg->AddEntry(result2_LinResult, Form("%s, #chi^{2}=%.1f",genName2,result2_chi2), "lf");
  if(!ignoreFour) leg->AddEntry(result3_LinResult, Form("%s, #chi^{2}=%.1f",genName3,result3_chi2), "lf");
  leg->AddEntry(result4_LinResult, Form("%s, #chi^{2}=%.1f",genName4,result4_chi2), "lf");
  leg->SetFillColor(kWhite);
  leg->SetFillStyle(0);


  //no chi2
/*
  TLegend* leg = new TLegend(0.05, 0.05, 0.95, 0.95);
  leg->AddEntry(data_slice[0],"T2K","ep");
  leg->AddEntry(result1_LinResult, Form("%s",genName1), "lf");
  leg->AddEntry(result2_LinResult, Form("%s",genName2), "lf");
  if(!ignoreFour) leg->AddEntry(result3_LinResult, Form("%s",genName3), "lf");
  leg->AddEntry(result4_LinResult, Form("%s",genName4), "lf");
  leg->SetFillColor(kWhite);
  leg->SetFillStyle(0);
*/


  cout << "Making pretty figures ... " << endl;

  gStyle->SetOptTitle(1);
  TGaxis::SetMaxDigits(3);


  // main canvas ******************************

    TCanvas* compCanv_0p = new TCanvas("compCanv", "compCanv", 1620, 1920);
    compCanv_0p->cd();

    std::vector<TPad*> pad;

    for (int i = 0; i < nBins+1; i++){ 
      switch(i){
        case 0 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.75,0.50,1.00));
                    if(strstr(varName.Data(),"p")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.04);
                    if(strstr(varName.Data(),"ip")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.07);
                    if(strstr(varName.Data(),"a")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.0025);
                    result1_slice[i]->SetNameTitle("-1.0 < cos(#theta_{#mu}) < -0.6","-1.0 < cos(#theta_{#mu}) < -0.6");
                    break;
                 }
        case 1 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.50,0.50,0.75));
                    if(strstr(varName.Data(),"p")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.05);
                    if(strstr(varName.Data(),"ip")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.04);
                    if(strstr(varName.Data(),"a")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.0008);
                    result1_slice[i]->SetNameTitle("-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} < 250 MeV","-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} < 250 MeV");
                    break;
                 }
        case 2 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.50,0.50,1.00,0.75));
                    if(strstr(varName.Data(),"p")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.1);
                    if(strstr(varName.Data(),"ip")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.08);
                    if(strstr(varName.Data(),"a")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.002);
                    result1_slice[i]->SetNameTitle("-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} > 250 MeV","-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} > 250 MeV");
                    break;
                 }
        case 3 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.25,0.50,0.50));
                    if(strstr(varName.Data(),"p")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.06);
                    if(strstr(varName.Data(),"ip")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.05);
                    if(strstr(varName.Data(),"a")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.0008);
                    result1_slice[i]->SetNameTitle("0.0 < cos(#theta_{#mu}) < 1.0, p_{#mu} < 250 MeV","0.0 < cos(#theta_{#mu}) < 1.0, p_{#mu} < 250 MeV");
                    break;
                 }
        case 4 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.50,0.25,1.00,0.50));
                    if(strstr(varName.Data(),"p")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.5);
                    if(strstr(varName.Data(),"ip")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.5);
                    if(strstr(varName.Data(),"a")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.005);
                    result1_slice[i]->SetNameTitle("0.0 < cos(#theta_{#mu}) < 0.8, p_{#mu} > 250 MeV","0.0 < cos(#theta_{#mu}) < 0.8, p_{#mu} > 250 MeV");
                    break;
                 }
        case 5 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.00,0.50,0.25));
                    if(strstr(varName.Data(),"p")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.08);
                    if(strstr(varName.Data(),"ip")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.04);
                    if(strstr(varName.Data(),"a")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.0005);
                    result1_slice[i]->SetNameTitle("0.8 < cos(#theta_{#mu}) < 1.0, 250 MeV < p_{#mu} < 750 MeV","0.8 < cos(#theta_{#mu}) < 1.0, 250 MeV < p_{#mu} < 750 MeV");
                    break;
                 }
        case 6 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.50,0.00,1.00,0.25)); 
                    if(strstr(varName.Data(),"p")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.1);
                    if(strstr(varName.Data(),"ip")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.08);
                    if(strstr(varName.Data(),"a")) result1_slice[i]->GetYaxis()->SetRangeUser(0, 0.0015);
                    result1_slice[i]->SetNameTitle("0.8 < cos(#theta_{#mu}) < 1.0, 750 MeV < p_{#mu}","0.8 < cos(#theta_{#mu}) < 1.0, p_{#mu} > 750 MeV");
                    break;
                 }
        case 7 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.50,0.75,1.00,1.00));
                    pad[i]->cd();
                    leg->Draw();
                    break; 
                 }
      }
      if(i<nBins){
        pad[i]->cd();
        if(strstr(varName.Data(),"p"))  result1_slice[i]->SetXTitle("#Delta p (GeV)");
        if(strstr(varName.Data(),"ip")) result1_slice[i]->SetXTitle("|#Delta #bf{p}| (GeV)");
        if(strstr(varName.Data(),"a"))  result1_slice[i]->SetXTitle("#Delta #theta (degrees)");
        if(strstr(varName.Data(),"p"))  result1_slice[i]->SetYTitle("#frac{d#sigma}{d #Delta p} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");
        if(strstr(varName.Data(),"ip")) result1_slice[i]->SetYTitle("#frac{d#sigma}{d |#Delta #bf{p}|} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");
        if(strstr(varName.Data(),"a"))  result1_slice[i]->SetYTitle("#frac{d#sigma}{d #Delta #theta} (10^{-39} cm^{2} Nucleon^{-1} degree^{-1})");
        result1_slice[i]->GetYaxis()->SetTitleOffset(1.35);
        result1_slice[i]->GetYaxis()->SetTitleSize(0.055);
        result1_slice[i]->Draw("HIST");
        result2_slice[i]->Draw("sameHIST");
        if(!ignoreFour) result3_slice[i]->Draw("sameHIST");
        result4_slice[i]->Draw("sameHIST");
        data_slice[i]->Draw("same");
        // Add inlay
        TPad* inlay = new TPad("inlay","inlay",0.52,0.28,0.89,0.93);
        inlay->cd();
        TH1D* result1_slice_clone = new TH1D(*(result1_slice[i]));
        result1_slice_clone->GetYaxis()->SetRangeUser(1E-5,1.0);
        if(strstr(varName.Data(),"a")) result1_slice_clone->GetYaxis()->SetRangeUser(1E-8,1E-2);
        result1_slice_clone->SetNameTitle("","");
        result1_slice_clone->GetYaxis()->SetTitle("");
        result1_slice_clone->GetXaxis()->SetTitle("");
        TH1D* data_slice_clone = new TH1D(*(data_slice[i]));
        data_slice_clone->SetMarkerSize(1.0);
        result1_slice_clone->Draw("HIST");
        result2_slice[i]->Draw("][sameHIST");
        if(!ignoreFour) result3_slice[i]->Draw("][sameHIST");
        result4_slice[i]->Draw("][sameHIST");
        data_slice_clone->Draw("][same");
        gPad->SetLogy();
        inlay->SetFillStyle(0);
        inlay->Update();
        pad[i]->cd();
        inlay->Draw();
      }
    }

    compCanv_0p->cd();
    for (int i = 0; i < nBins+1; i++){ 
      //cout << i << endl;
      pad[i]->Draw("same");
    }
    compCanv_0p->Write();
    compCanv_0p->SaveAs(Form("%s.png",outFileTag.Data()));
    compCanv_0p->SaveAs(Form("%s.pdf",outFileTag.Data()));

  cout << "Finished canvas ... " << endl;


  //char* saveName = Form("%s_multDifComp_0p.pdf",genName);
  //char* saveNamepng = Form("%s_multDifComp_0p.png",genName);
  //compCanv_0p->SaveAs(saveName);
  //compCanv_0p->SaveAs(saveNamepng);

  cout << "Finished :-D" << endl;

  return;

}
