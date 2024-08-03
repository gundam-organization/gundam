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


void multiComp_multiDif(TString inFileName1, TString inFileName2, char* genName1, char* genName2, TString outFileTag){

  TFile *inFile1_0p = new TFile(Form("0p_%s",inFileName1.Data()));
  TFile *inFile1_1p = new TFile(Form("Np_%s",inFileName1.Data()));
  TFile *inFile1_comb = new TFile(Form("comb_%s",inFileName1.Data()));

  TFile *inFile2_0p = new TFile(Form("0p_%s",inFileName2.Data()));
  TFile *inFile2_1p = new TFile(Form("Np_%s",inFileName2.Data()));
  TFile *inFile2_comb = new TFile(Form("comb_%s",inFileName2.Data()));

  //******** Collect inputs ******

  TH1D* data_cthmu_0p    = new TH1D();
  TH1D* result1_cthmu_0p = new TH1D();
  TH1D* result2_cthmu_0p = new TH1D();
  TH1D* data_cthmu_1p    = new TH1D();
  TH1D* result1_cthmu_1p = new TH1D();
  TH1D* result2_cthmu_1p = new TH1D();

  data_cthmu_0p    = (TH1D*)  inFile1_0p->Get("result_cthmu");
  result1_cthmu_0p = (TH1D*)  inFile1_0p->Get("MC_cthmu");
  result2_cthmu_0p = (TH1D*)  inFile2_0p->Get("MC_cthmu");
  data_cthmu_1p    = (TH1D*)  inFile1_1p->Get("result_cthmu");
  result1_cthmu_1p = (TH1D*)  inFile1_1p->Get("MC_cthmu");
  result2_cthmu_1p = (TH1D*)  inFile2_1p->Get("MC_cthmu");

  TH1D* data_NpHist    = new TH1D();
  TH1D* result1_NpHist = new TH1D();
  TH1D* result2_NpHist = new TH1D();
  TH1D* data_LinResult    = new TH1D();
  TH1D* result1_LinResult = new TH1D();
  TH1D* result2_LinResult = new TH1D();

  data_NpHist    = (TH1D*) inFile1_comb->Get("nprotonsResult");
  result1_NpHist = (TH1D*) inFile1_comb->Get("nprotonsResult_MC");
  result2_NpHist = (TH1D*) inFile2_comb->Get("nprotonsResult_MC");
  data_LinResult    = (TH1D*) inFile1_comb->Get("LinResult");
  result1_LinResult = (TH1D*) inFile1_comb->Get("LinResult_MC");
  result2_LinResult = (TH1D*) inFile2_comb->Get("LinResult_MC");

  std::vector<TH1D*> data_slice_0p;
  std::vector<TH1D*> data_slice_1p;
  std::vector<TH1D*> result1_slice_0p;
  std::vector<TH1D*> result1_slice_1p;
  std::vector<TH1D*> result2_slice_0p;
  std::vector<TH1D*> result2_slice_1p;

  for (int i = 0; i < 10; i++){ 
    data_slice_0p.push_back(new TH1D);
    result1_slice_0p.push_back(new TH1D);
    result2_slice_0p.push_back(new TH1D);

    data_slice_0p[i]    = (TH1D*) inFile1_0p->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_data_Slice%d",i));
    result1_slice_0p[i] = (TH1D*) inFile1_0p->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_MC_Slice%d",i));
    result2_slice_0p[i] = (TH1D*) inFile2_0p->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_MC_Slice%d",i));
  }


  for (int i = 0; i < 8; i++){ 
    data_slice_1p.push_back(new TH1D);
    result1_slice_1p.push_back(new TH1D);
    result2_slice_1p.push_back(new TH1D);
  }

  data_slice_1p[0]    = (TH1D*) inFile1_1p->Get("result_cthmu1");
  result1_slice_1p[0] = (TH1D*) inFile1_1p->Get("MC_cthmu1");
  result2_slice_1p[0] = (TH1D*) inFile2_1p->Get("MC_cthmu1");
  data_slice_1p[1]    = (TH1D*) inFile1_1p->Get("result_cthmu2");
  result1_slice_1p[1] = (TH1D*) inFile1_1p->Get("MC_cthmu2");
  result2_slice_1p[1] = (TH1D*) inFile2_1p->Get("MC_cthmu2");
  data_slice_1p[2]    = (TH1D*) inFile1_1p->Get("result_cthmu2_pmom");
  result1_slice_1p[2] = (TH1D*) inFile1_1p->Get("MC_cthmu2_pmom");
  result2_slice_1p[2] = (TH1D*) inFile2_1p->Get("MC_cthmu2_pmom");
  data_slice_1p[3]    = (TH1D*) inFile1_1p->Get("result_cthmu3");
  result1_slice_1p[3] = (TH1D*) inFile1_1p->Get("MC_cthmu3");
  result2_slice_1p[3] = (TH1D*) inFile2_1p->Get("MC_cthmu3");
  data_slice_1p[4]    = (TH1D*) inFile1_1p->Get("result_cthmu3_pmom1");
  result1_slice_1p[4] = (TH1D*) inFile1_1p->Get("MC_cthmu3_pmom1");
  result2_slice_1p[4] = (TH1D*) inFile2_1p->Get("MC_cthmu3_pmom1");
  data_slice_1p[5]    = (TH1D*) inFile1_1p->Get("result_cthmu3_pmom2");
  result1_slice_1p[5] = (TH1D*) inFile1_1p->Get("MC_cthmu3_pmom2");
  result2_slice_1p[5] = (TH1D*) inFile2_1p->Get("MC_cthmu3_pmom2");
  data_slice_1p[6]    = (TH1D*) inFile1_1p->Get("result_cthmu4");
  result1_slice_1p[6] = (TH1D*) inFile1_1p->Get("MC_cthmu4");
  result2_slice_1p[6] = (TH1D*) inFile2_1p->Get("MC_cthmu4");
  data_slice_1p[7]    = (TH1D*) inFile1_1p->Get("result_cthmu4_pmom");
  result1_slice_1p[7] = (TH1D*) inFile1_1p->Get("MC_cthmu4_pmom");
  result2_slice_1p[7] = (TH1D*) inFile2_1p->Get("MC_cthmu4_pmom");

  TH1D* result1_chi2Hist = new TH1D();
  TH1D* result2_chi2Hist = new TH1D();
  result1_chi2Hist = (TH1D*)  inFile1_comb->Get("chi2hist");
  result2_chi2Hist = (TH1D*)  inFile2_comb->Get("chi2hist");

  double result1_chi2 = result1_chi2Hist->GetBinContent(1);
  double result2_chi2 = result2_chi2Hist->GetBinContent(1);


  //******** Make output plots ******


  cout << "Writting output ... " << endl;

  TFile *outfile = new TFile(Form("%s.root",outFileTag.Data()),"recreate");
  outfile->cd();

  result1_cthmu_0p->SetLineColor(kBlue);
  result1_cthmu_1p->SetLineColor(kBlue);
  result1_NpHist->SetLineColor(kBlue);
  result1_LinResult->SetLineColor(kBlue);

  result2_cthmu_0p->SetLineColor(kRed);
  result2_cthmu_1p->SetLineColor(kRed);
  result2_NpHist->SetLineColor(kRed);
  result2_LinResult->SetLineColor(kRed);

  for (int i = 0; i < 10; i++){ result1_slice_0p[i]->SetLineColor(kBlue); }
  for (int i = 0; i < 10; i++){ result2_slice_0p[i]->SetLineColor(kRed); }

  for (int i = 0; i < 8; i++){ result1_slice_1p[i]->SetLineColor(kBlue); }
  for (int i = 0; i < 8; i++){ result2_slice_1p[i]->SetLineColor(kRed); }


  TLegend* leg = new TLegend(0.2, 0.2, .8, .8);
  leg->AddEntry(data_LinResult,"T2K Fit to Data","lep");
  leg->AddEntry(result1_LinResult, Form("%s, #chi^{2}=%.1f",genName1,result1_chi2), "lf");
  leg->AddEntry(result2_LinResult, Form("%s, #chi^{2}=%.1f",genName2,result2_chi2), "lf");
  leg->SetFillColor(kWhite);
  leg->SetFillStyle(0);


  cout << "Making pretty figures ... " << endl;

  gStyle->SetOptTitle(1);

  TCanvas* compCanv_comb = new TCanvas("compCanv_comb", "compCanv_comb", 1920, 360);
  compCanv_comb->cd();

  TPad* nprotonsPad = new TPad("nprotonsPad","nprotonsPad",0.0,0.00,0.33,1.00);
  nprotonsPad->cd();
  data_NpHist->Draw();
  result1_NpHist->Draw("sameHIST");
  result2_NpHist->Draw("sameHIST");
  data_NpHist->Draw("same");

  TPad* lineResultPad = new TPad("lineResultPad","lineResultPad",0.33,0.00,0.66,1.00);
  lineResultPad->cd();
  data_LinResult->GetYaxis()->SetRangeUser(0,0.3);
  data_LinResult->Draw();
  result1_LinResult->Draw("sameHIST");
  result2_LinResult->Draw("sameHIST");
  data_LinResult->Draw("same");

  TPad* legPadComb = new TPad("legPadComb","legPadComb",0.66,0.00,0.99,1.00);
  legPadComb->cd();
  leg->Draw();
  //result1_LinResult->Draw("histSAME")
  //result2_LinResult->Draw("histSAME")
  //data_LinResult->Draw("same");

  compCanv_comb->cd();

  nprotonsPad->Draw();
  lineResultPad->Draw("same");
  legPadComb->Draw("same");
  compCanv_comb->Write();
  compCanv_comb->SaveAs(Form("%s_comb.png",outFileTag.Data()));
  compCanv_comb->SaveAs(Form("%s_comb.pdf",outFileTag.Data()));


  // 0p canvas ******************************

  TCanvas* compCanv_0p = new TCanvas("compCanv_0p", "compCanv_0p", 1920, 1080);
  compCanv_0p->cd();

  TPad* cthmuPad0p = new TPad("cthmuPad0p","cthmuPad0p",0.0,0.75,0.33,1.00);
  cthmuPad0p->cd();
  data_cthmu_0p->GetYaxis()->SetRangeUser(0,10);
  data_cthmu_0p->SetXTitle("cos(#theta_{#mu})");
  data_cthmu_0p->SetYTitle("#frac{d#sigma}{d cos(#theta_{#mu})} (10^{-39} cm^{2} Nucleon^{-1})");
  data_cthmu_0p->SetNameTitle("result_cthmu","Integrated over all p_{#mu}");
  data_cthmu_0p->Draw();
  result1_cthmu_0p->Draw("sameHIST");
  result2_cthmu_0p->Draw("sameHIST");
  data_cthmu_0p->Draw("same");

  std::vector<TPad*> pad;

  for (int i = 0; i < 10; i++){ 
    switch(i){
      case 0 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.33,0.75,0.66,1.00));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 1.6);
                  result1_slice_0p[i+1]->SetNameTitle("-0.3 < cos(#theta_{#mu}) < 0.3", "-0.3 < cos(#theta_{#mu}) < 0.3");
                  break;
               }
      case 1 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.66,0.75,0.99,1.00));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 1.6);
                  result1_slice_0p[i+1]->SetNameTitle("0.3 < cos(#theta_{#mu}) < 0.6", "0.3 < cos(#theta_{#mu}) < 0.6");
                  break;
               }
      case 2 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.50,0.33,0.75));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.8);
                  result1_slice_0p[i+1]->SetNameTitle("0.6 < cos(#theta_{#mu}) < 0.7", "0.6 < cos(#theta_{#mu}) < 0.7");
                  break;
               }
      case 3 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.33,0.50,0.66,0.75));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.9);
                  result1_slice_0p[i+1]->SetNameTitle("0.7 < cos(#theta_{#mu}) < 0.8", "0.7 < cos(#theta_{#mu}) < 0.8");
                  break;
               }
      case 4 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.66,0.50,0.99,0.75));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.6);
                  result1_slice_0p[i+1]->SetNameTitle("0.8 < cos(#theta_{#mu}) < 0.85", "0.8 < cos(#theta_{#mu}) < 0.85");
                  break;
               }
      case 5 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.25,0.33,0.50));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.6);
                  result1_slice_0p[i+1]->SetNameTitle("0.85 < cos(#theta_{#mu}) < 0.9", "0.85 < cos(#theta_{#mu}) < 0.9");
                  break;
               }
      case 6 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.33,0.25,0.66,0.50));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.4);
                  result1_slice_0p[i+1]->SetNameTitle("0.9 < cos(#theta_{#mu}) < 0.94", "0.9 < cos(#theta_{#mu}) < 0.94");
                  break;
               }
      case 7 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.66,0.25,0.99,0.50));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.4);
                  result1_slice_0p[i+1]->SetNameTitle("0.94 < cos(#theta_{#mu}) < 0.98", "0.94 < cos(#theta_{#mu}) < 0.98");
                  break;
               }
      case 8 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.00,0.33,0.25));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.15);
                  result1_slice_0p[i+1]->SetNameTitle("0.98 < cos(#theta_{#mu}) < 1.0", "0.98 < cos(#theta_{#mu}) < 1.0");
                  break;
               }
      case 9 : { 
                  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.33,0.00,0.66,0.25));
                  pad[i]->cd();
                  leg->Draw();
                  break; 
               }
    }
    if(i<9){
      pad[i]->cd();
      result1_slice_0p[i+1]->SetXTitle("p_{#mu} (GeV)");
      result1_slice_0p[i+1]->SetYTitle("#frac{d#sigma}{d p_{#mu}} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");
      result1_slice_0p[i+1]->Draw("HIST");
      result2_slice_0p[i+1]->Draw("sameHIST");
      data_slice_0p[i+1]->Draw("same");
    }
  }

  compCanv_0p->cd();
  cthmuPad0p->Draw();
  for (int i = 0; i < 10; i++){ 
    //cout << i << endl;
    pad[i]->Draw("same");
  }
  compCanv_0p->Write();
  compCanv_0p->SaveAs(Form("%s_0p.png",outFileTag.Data()));
  compCanv_0p->SaveAs(Form("%s_0p.pdf",outFileTag.Data()));



  cout << "Finished 0p canvas ... " << endl;

  // ********* Now draw the 1p pad *************************************


  TCanvas* compCanv_1p = new TCanvas("compCanv_1p", "compCanv_1p", 1920, 1080);
  compCanv_1p->cd();

  TPad* cthmuPad1p = new TPad("cthmuPad1p","cthmuPad1p",0.0,0.66,0.33,0.99);
  cthmuPad1p->cd();
  data_cthmu_1p->SetNameTitle("result_cthmu","Integrated over all cos(#theta_{p}) and p_{p}");
  data_cthmu_1p->GetYaxis()->SetRangeUser(0,2.6);
  data_cthmu_1p->Draw();
  result1_cthmu_1p->Draw("sameHIST");
  result2_cthmu_1p->Draw("sameHIST");
  data_cthmu_1p->Draw("same");

  TPad* cthmu2 = new TPad("cthmu2","cthmu2",0.33,0.66,0.66,0.99);
  cthmu2->cd();
  data_slice_1p[1]->SetXTitle("cos(#theta_{p})");
  data_slice_1p[1]->GetYaxis()->SetRangeUser(0,2.6);
  data_slice_1p[1]->Draw();
  result1_slice_1p[1]->Draw("sameHIST");
  result2_slice_1p[1]->Draw("sameHIST");
  data_slice_1p[1]->Draw("same");


  TPad* cthmu2_pmom = new TPad("cthmu2_pmom","cthmu2_pmom",0.66,0.66,0.99,0.99);
  cthmu2_pmom->cd();
  data_slice_1p[2]->GetYaxis()->SetRangeUser(0,0.7);
  data_slice_1p[2]->Draw();
  result1_slice_1p[2]->Draw("sameHIST");
  result2_slice_1p[2]->Draw("sameHIST");
  data_slice_1p[2]->Draw("same");

  TPad* cthmu3 = new TPad("cthmu3","cthmu3",0.0,0.33,0.33,0.66);
  cthmu3->cd();
  data_slice_1p[3]->SetXTitle("cos(#theta_{p})");
  data_slice_1p[3]->GetYaxis()->SetRangeUser(0,2.0);
  data_slice_1p[3]->Draw();
  result1_slice_1p[3]->Draw("sameHIST");
  result2_slice_1p[3]->Draw("sameHIST");
  data_slice_1p[3]->Draw("same");

  TPad* cthmu3_pmom1 = new TPad("cthmu3_pmom1","cthmu3_pmom1",0.33,0.33,0.66,0.66);
  cthmu3_pmom1->cd();
  data_slice_1p[4]->GetYaxis()->SetRangeUser(0,2.0);
  data_slice_1p[4]->Draw();
  result1_slice_1p[4]->Draw("sameHIST");
  result2_slice_1p[4]->Draw("sameHIST");
  data_slice_1p[4]->Draw("same");

  TPad* cthmu3_pmom2 = new TPad("cthmu3_pmom2","cthmu3_pmom2",0.66,0.33,0.99,0.66);
  cthmu3_pmom2->cd();
  data_slice_1p[5]->GetYaxis()->SetRangeUser(0,1.2);
  data_slice_1p[5]->Draw();
  result1_slice_1p[5]->Draw("sameHIST");
  result2_slice_1p[5]->Draw("sameHIST");
  data_slice_1p[5]->Draw("same");

  TPad* cthmu4 = new TPad("cthmu4","cthmu4",0.0,0.00,0.33,0.33);
  cthmu4->cd();
  data_slice_1p[6]->SetXTitle("cos(#theta_{p})");
  data_slice_1p[6]->GetYaxis()->SetRangeUser(0,1.0);
  data_slice_1p[6]->Draw();
  result1_slice_1p[6]->Draw("sameHIST");
  result2_slice_1p[6]->Draw("sameHIST");
  data_slice_1p[6]->Draw("same");

  TPad* cthmu4_pmom = new TPad("cthmu4_pmom","cthmu4_pmom",0.33,0.00,0.66,0.33);
  cthmu4_pmom->cd();
  data_slice_1p[7]->GetYaxis()->SetRangeUser(0,1.5);
  data_slice_1p[7]->Draw();
  result1_slice_1p[7]->Draw("sameHIST");
  result2_slice_1p[7]->Draw("sameHIST");
  data_slice_1p[7]->Draw("same");

  TPad* legPad = new TPad("legPad","legPad",0.66,0.00,0.99,0.33);
  legPad->cd();
  leg->Draw();

  compCanv_1p->cd();
  cthmuPad1p->Draw();
  //cthmu1->Draw("same");
  cthmu2->Draw("same");
  cthmu2_pmom->Draw("same");
  cthmu3->Draw("same");
  cthmu3_pmom1->Draw("same");
  cthmu3_pmom2->Draw("same");
  cthmu4->Draw("same");
  cthmu4_pmom->Draw("same");
  legPad->Draw("same");

  compCanv_1p->Write();
  compCanv_1p->SaveAs(Form("%s_1p.png",outFileTag.Data()));
  compCanv_1p->SaveAs(Form("%s_1p.pdf",outFileTag.Data()));
  //char* saveName = Form("%s_multDifComp_0p.pdf",genName);
  //char* saveNamepng = Form("%s_multDifComp_0p.png",genName);
  //compCanv_0p->SaveAs(saveName);
  //compCanv_0p->SaveAs(saveNamepng);

  cout << "Finished :-D" << endl;

  return;

}

//*********************************************************************************************************
//*********************************************************************************************************
//*********************************************************************************************************
//*********************************************************************************************************


void multiComp_multiDif_4comp(TString inFileName1, TString inFileName2, TString inFileName3, TString inFileName4, char* genName1, char* genName2, char* genName3, char* genName4, TString outFileTag, bool ignoreFour=false){

  TFile *inFile1_0p = new TFile(Form("0p_%s",inFileName1.Data()));
  TFile *inFile1_1p = new TFile(Form("Np_%s",inFileName1.Data()));
  TFile *inFile1_comb = new TFile(Form("comb_%s",inFileName1.Data()));

  TFile *inFile2_0p = new TFile(Form("0p_%s",inFileName2.Data()));
  TFile *inFile2_1p = new TFile(Form("Np_%s",inFileName2.Data()));
  TFile *inFile2_comb = new TFile(Form("comb_%s",inFileName2.Data()));

  TFile *inFile3_0p = new TFile(Form("0p_%s",inFileName3.Data()));
  TFile *inFile3_1p = new TFile(Form("Np_%s",inFileName3.Data()));
  TFile *inFile3_comb = new TFile(Form("comb_%s",inFileName3.Data()));

  TFile *inFile4_0p = new TFile(Form("0p_%s",inFileName4.Data()));
  TFile *inFile4_1p = new TFile(Form("Np_%s",inFileName4.Data()));
  TFile *inFile4_comb = new TFile(Form("comb_%s",inFileName4.Data()));

  //inFile1_0p->ls();
  //inFile1_1p->ls();
  //inFile1_comb->ls();
  //inFile2_0p->ls();
  //inFile2_1p->ls();
  //inFile2_comb->ls();
  //inFile3_0p->ls();
  //inFile3_1p->ls();
  //inFile3_comb->ls();
  //inFile4_0p->ls();
  //inFile4_1p->ls();
  //inFile4_comb->ls();

  //******** Collect inputs ******

  TH1D* data_cthmu_0p    = new TH1D();
  TH1D* result1_cthmu_0p = new TH1D();
  TH1D* result2_cthmu_0p = new TH1D();
  TH1D* result3_cthmu_0p = new TH1D();
  TH1D* result4_cthmu_0p = new TH1D();
  TH1D* data_cthmu_1p    = new TH1D();
  TH1D* result1_cthmu_1p = new TH1D();
  TH1D* result2_cthmu_1p = new TH1D();
  TH1D* result3_cthmu_1p = new TH1D();
  TH1D* result4_cthmu_1p = new TH1D();


  data_cthmu_0p    = (TH1D*)  inFile1_0p->Get("result_cthmu");
  result1_cthmu_0p = (TH1D*)  inFile1_0p->Get("MC_cthmu");
  result2_cthmu_0p = (TH1D*)  inFile2_0p->Get("MC_cthmu");
  result3_cthmu_0p = (TH1D*)  inFile3_0p->Get("MC_cthmu");
  result4_cthmu_0p = (TH1D*)  inFile4_0p->Get("MC_cthmu");
  data_cthmu_1p    = (TH1D*)  inFile1_1p->Get("result_cthmu");
  result1_cthmu_1p = (TH1D*)  inFile1_1p->Get("MC_cthmu");
  result2_cthmu_1p = (TH1D*)  inFile2_1p->Get("MC_cthmu");
  result3_cthmu_1p = (TH1D*)  inFile3_1p->Get("MC_cthmu");
  result4_cthmu_1p = (TH1D*)  inFile4_1p->Get("MC_cthmu");

  TH1D* data_NpHist    = new TH1D();
  TH1D* result1_NpHist = new TH1D();
  TH1D* result2_NpHist = new TH1D();
  TH1D* result3_NpHist = new TH1D();
  TH1D* result4_NpHist = new TH1D();
  TH1D* data_LinResult    = new TH1D();
  TH1D* result1_LinResult = new TH1D();
  TH1D* result2_LinResult = new TH1D();
  TH1D* result3_LinResult = new TH1D();
  TH1D* result4_LinResult = new TH1D();

  data_NpHist    = (TH1D*) inFile1_comb->Get("nprotonsResult");
  result1_NpHist = (TH1D*) inFile1_comb->Get("nprotonsResult_MC");
  result2_NpHist = (TH1D*) inFile2_comb->Get("nprotonsResult_MC");
  result3_NpHist = (TH1D*) inFile3_comb->Get("nprotonsResult_MC");
  result4_NpHist = (TH1D*) inFile4_comb->Get("nprotonsResult_MC");
  data_LinResult    = (TH1D*) inFile1_comb->Get("LinResult");
  result1_LinResult = (TH1D*) inFile1_comb->Get("LinResult_MC");
  result2_LinResult = (TH1D*) inFile2_comb->Get("LinResult_MC");
  result3_LinResult = (TH1D*) inFile3_comb->Get("LinResult_MC");
  result4_LinResult = (TH1D*) inFile4_comb->Get("LinResult_MC");

  std::vector<TH1D*> data_slice_0p;
  std::vector<TH1D*> data_slice_1p;
  std::vector<TH1D*> result1_slice_0p;
  std::vector<TH1D*> result1_slice_1p;
  std::vector<TH1D*> result2_slice_0p;
  std::vector<TH1D*> result2_slice_1p;
  std::vector<TH1D*> result3_slice_0p;
  std::vector<TH1D*> result3_slice_1p;
  std::vector<TH1D*> result4_slice_0p;
  std::vector<TH1D*> result4_slice_1p;

  for (int i = 0; i < 10; i++){ 
    data_slice_0p.push_back(new TH1D);
    result1_slice_0p.push_back(new TH1D);
    result2_slice_0p.push_back(new TH1D);
    result3_slice_0p.push_back(new TH1D);
    result4_slice_0p.push_back(new TH1D);

    data_slice_0p[i]    = (TH1D*) inFile1_0p->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_data_Slice%d",i));
    result1_slice_0p[i] = (TH1D*) inFile1_0p->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_MC_Slice%d",i));
    result2_slice_0p[i] = (TH1D*) inFile2_0p->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_MC_Slice%d",i));
    result3_slice_0p[i] = (TH1D*) inFile3_0p->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_MC_Slice%d",i));
    result4_slice_0p[i] = (TH1D*) inFile4_0p->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_MC_Slice%d",i));
  }


  for (int i = 0; i < 8; i++){ 
    data_slice_1p.push_back(new TH1D);
    result1_slice_1p.push_back(new TH1D);
    result2_slice_1p.push_back(new TH1D);
    result3_slice_1p.push_back(new TH1D);
    result4_slice_1p.push_back(new TH1D);
  }

  data_slice_1p[0]    = (TH1D*) inFile1_1p->Get("result_cthmu1");
  result1_slice_1p[0] = (TH1D*) inFile1_1p->Get("MC_cthmu1");
  result2_slice_1p[0] = (TH1D*) inFile2_1p->Get("MC_cthmu1");
  result3_slice_1p[0] = (TH1D*) inFile3_1p->Get("MC_cthmu1");
  result4_slice_1p[0] = (TH1D*) inFile4_1p->Get("MC_cthmu1");  
  data_slice_1p[1]    = (TH1D*) inFile1_1p->Get("result_cthmu2");
  result1_slice_1p[1] = (TH1D*) inFile1_1p->Get("MC_cthmu2");
  result2_slice_1p[1] = (TH1D*) inFile2_1p->Get("MC_cthmu2");
  result3_slice_1p[1] = (TH1D*) inFile3_1p->Get("MC_cthmu2");
  result4_slice_1p[1] = (TH1D*) inFile4_1p->Get("MC_cthmu2");
  data_slice_1p[2]    = (TH1D*) inFile1_1p->Get("result_cthmu2_pmom");
  result1_slice_1p[2] = (TH1D*) inFile1_1p->Get("MC_cthmu2_pmom");
  result2_slice_1p[2] = (TH1D*) inFile2_1p->Get("MC_cthmu2_pmom");
  result3_slice_1p[2] = (TH1D*) inFile3_1p->Get("MC_cthmu2_pmom");
  result4_slice_1p[2] = (TH1D*) inFile4_1p->Get("MC_cthmu2_pmom");  
  data_slice_1p[3]    = (TH1D*) inFile1_1p->Get("result_cthmu3");
  result1_slice_1p[3] = (TH1D*) inFile1_1p->Get("MC_cthmu3");
  result2_slice_1p[3] = (TH1D*) inFile2_1p->Get("MC_cthmu3");
  result3_slice_1p[3] = (TH1D*) inFile3_1p->Get("MC_cthmu3");
  result4_slice_1p[3] = (TH1D*) inFile4_1p->Get("MC_cthmu3");  
  data_slice_1p[4]    = (TH1D*) inFile1_1p->Get("result_cthmu3_pmom1");
  result1_slice_1p[4] = (TH1D*) inFile1_1p->Get("MC_cthmu3_pmom1");
  result2_slice_1p[4] = (TH1D*) inFile2_1p->Get("MC_cthmu3_pmom1");
  result3_slice_1p[4] = (TH1D*) inFile3_1p->Get("MC_cthmu3_pmom1");
  result4_slice_1p[4] = (TH1D*) inFile4_1p->Get("MC_cthmu3_pmom1");  
  data_slice_1p[5]    = (TH1D*) inFile1_1p->Get("result_cthmu3_pmom2");
  result1_slice_1p[5] = (TH1D*) inFile1_1p->Get("MC_cthmu3_pmom2");
  result2_slice_1p[5] = (TH1D*) inFile2_1p->Get("MC_cthmu3_pmom2");
  result3_slice_1p[5] = (TH1D*) inFile3_1p->Get("MC_cthmu3_pmom2");
  result4_slice_1p[5] = (TH1D*) inFile4_1p->Get("MC_cthmu3_pmom2");  
  data_slice_1p[6]    = (TH1D*) inFile1_1p->Get("result_cthmu4");
  result1_slice_1p[6] = (TH1D*) inFile1_1p->Get("MC_cthmu4");
  result2_slice_1p[6] = (TH1D*) inFile2_1p->Get("MC_cthmu4");
  result3_slice_1p[6] = (TH1D*) inFile3_1p->Get("MC_cthmu4");
  result4_slice_1p[6] = (TH1D*) inFile4_1p->Get("MC_cthmu4"); 
  data_slice_1p[7]    = (TH1D*) inFile1_1p->Get("result_cthmu4_pmom");
  result1_slice_1p[7] = (TH1D*) inFile1_1p->Get("MC_cthmu4_pmom");
  result2_slice_1p[7] = (TH1D*) inFile2_1p->Get("MC_cthmu4_pmom");
  result3_slice_1p[7] = (TH1D*) inFile3_1p->Get("MC_cthmu4_pmom");
  result4_slice_1p[7] = (TH1D*) inFile4_1p->Get("MC_cthmu4_pmom");

  TH1D* result1_chi2Hist = new TH1D();
  TH1D* result2_chi2Hist = new TH1D();
  TH1D* result3_chi2Hist = new TH1D();
  TH1D* result4_chi2Hist = new TH1D();
  result1_chi2Hist = (TH1D*)  inFile1_comb->Get("chi2hist");
  result2_chi2Hist = (TH1D*)  inFile2_comb->Get("chi2hist");
  result3_chi2Hist = (TH1D*)  inFile3_comb->Get("chi2hist");
  result4_chi2Hist = (TH1D*)  inFile4_comb->Get("chi2hist");

  double result1_chi2 = result1_chi2Hist->GetBinContent(1);
  double result2_chi2 = result2_chi2Hist->GetBinContent(1);
  double result3_chi2 = result3_chi2Hist->GetBinContent(1);
  double result4_chi2 = result4_chi2Hist->GetBinContent(1);


  //******** Make output plots ******


  cout << "Writting output ... " << endl;

  TFile *outfile = new TFile(Form("%s.root",outFileTag.Data()),"recreate");
  outfile->cd();

  // *******************
  // Original version 
  // *******************
  /*

  result1_cthmu_0p->SetLineColor(kViolet-4);
  result1_cthmu_1p->SetLineColor(kViolet-4);
  result1_NpHist->SetLineColor(kViolet-4);
  result1_LinResult->SetLineColor(kViolet-4);

  result2_cthmu_0p->SetLineColor(kRed-3);
  result2_cthmu_1p->SetLineColor(kRed-3);
  result2_NpHist->SetLineColor(kRed-3);
  result2_LinResult->SetLineColor(kRed-3);

  result3_cthmu_0p->SetLineColor(kBlack);
  result3_cthmu_1p->SetLineColor(kBlack);
  result3_NpHist->SetLineColor(kBlack);
  result3_LinResult->SetLineColor(kBlack);

  result3_cthmu_0p->SetLineStyle(2);
  result3_cthmu_1p->SetLineStyle(2);
  result3_NpHist->SetLineStyle(2);
  result3_LinResult->SetLineStyle(2);

  result4_cthmu_0p->SetLineColor(kGreen-6);
  result4_cthmu_1p->SetLineColor(kGreen-6);
  result4_NpHist->SetLineColor(kGreen-6);
  result4_LinResult->SetLineColor(kGreen-6);

  for (int i = 0; i < 10; i++){ result1_slice_0p[i]->SetLineColor(kViolet-4); }
  for (int i = 0; i < 10; i++){ result2_slice_0p[i]->SetLineColor(kRed-3); }
  for (int i = 0; i < 10; i++){ result3_slice_0p[i]->SetLineColor(kBlack); }
  for (int i = 0; i < 10; i++){ result3_slice_0p[i]->SetLineStyle(2); }    
  for (int i = 0; i < 10; i++){ result4_slice_0p[i]->SetLineColor(kGreen-6); }

  for (int i = 0; i < 8; i++){ result1_slice_1p[i]->SetLineColor(kViolet-4); }
  for (int i = 0; i < 8; i++){ result2_slice_1p[i]->SetLineColor(kRed-3); }
  for (int i = 0; i < 8; i++){ result3_slice_1p[i]->SetLineColor(kBlack); }
  for (int i = 0; i < 8; i++){ result3_slice_1p[i]->SetLineStyle(2); }
  for (int i = 0; i < 8; i++){ result4_slice_1p[i]->SetLineColor(kGreen-6); }

  */
  // *******************
  // Paper Style 
  // *******************

  result1_cthmu_0p->SetLineColor(kAzure+8);
  result1_cthmu_1p->SetLineColor(kAzure+8);
  result1_NpHist->SetLineColor(kAzure+8);
  result1_LinResult->SetLineColor(kAzure+8);

  result2_cthmu_0p->SetLineColor(kAzure-2);
  result2_cthmu_1p->SetLineColor(kAzure-2);
  result2_NpHist->SetLineColor(kAzure-2);
  result2_LinResult->SetLineColor(kAzure-2);

  result3_cthmu_0p->SetLineColor(kRed-4);
  result3_cthmu_1p->SetLineColor(kRed-4);
  result3_NpHist->SetLineColor(kRed-4);
  result3_LinResult->SetLineColor(kRed-4);

  result4_cthmu_0p->SetLineColor(kRed+3);
  result4_cthmu_1p->SetLineColor(kRed+3);
  result4_NpHist->SetLineColor(kRed+3);
  result4_LinResult->SetLineColor(kRed+3);

  result2_cthmu_0p->SetLineStyle(2);
  result2_cthmu_1p->SetLineStyle(2);
  result2_NpHist->SetLineStyle(2);
  result2_LinResult->SetLineStyle(2);

  result4_cthmu_0p->SetLineStyle(3);
  result4_cthmu_1p->SetLineStyle(3);
  result4_NpHist->SetLineStyle(3);
  result4_LinResult->SetLineStyle(3);

  result1_cthmu_0p->SetLineWidth(4);
  result1_cthmu_1p->SetLineWidth(4);
  result1_NpHist->SetLineWidth(4);
  result1_LinResult->SetLineWidth(4);

  result2_cthmu_0p->SetLineWidth(3);
  result2_cthmu_1p->SetLineWidth(3);
  result2_NpHist->SetLineWidth(3);
  result2_LinResult->SetLineWidth(3);

  result3_cthmu_0p->SetLineWidth(2);
  result3_cthmu_1p->SetLineWidth(2);
  result3_NpHist->SetLineWidth(2);
  result3_LinResult->SetLineWidth(2);

  result4_cthmu_0p->SetLineWidth(2);
  result4_cthmu_1p->SetLineWidth(2);
  result4_NpHist->SetLineWidth(2);
  result4_LinResult->SetLineWidth(2);

  data_NpHist->SetMarkerSize(1.5);
  data_cthmu_0p->SetMarkerSize(1.5);
  data_cthmu_1p->SetMarkerSize(1.5);

  for (int i = 0; i < 10; i++){ data_slice_0p[i]->SetMarkerSize(1.5); }

  for (int i = 0; i < 10; i++){ result1_slice_0p[i]->SetLineColor(kAzure+8); }
  for (int i = 0; i < 10; i++){ result2_slice_0p[i]->SetLineColor(kAzure-2); }
  for (int i = 0; i < 10; i++){ result3_slice_0p[i]->SetLineColor(kRed-4); }
  for (int i = 0; i < 10; i++){ result4_slice_0p[i]->SetLineColor(kRed+3); }

  for (int i = 0; i < 10; i++){ result1_slice_0p[i]->SetLineWidth(4); }
  for (int i = 0; i < 10; i++){ result2_slice_0p[i]->SetLineWidth(3); }
  for (int i = 0; i < 10; i++){ result3_slice_0p[i]->SetLineWidth(2); }
  for (int i = 0; i < 10; i++){ result4_slice_0p[i]->SetLineWidth(2); }

  for (int i = 0; i < 10; i++){ result2_slice_0p[i]->SetLineStyle(2); }    
  for (int i = 0; i < 10; i++){ result4_slice_0p[i]->SetLineStyle(3); }

  for (int i = 0; i < 8; i++){ data_slice_1p[i]->SetMarkerSize(1.5); }

  for (int i = 0; i < 8; i++){ result1_slice_1p[i]->SetLineColor(kAzure+8); }
  for (int i = 0; i < 8; i++){ result2_slice_1p[i]->SetLineColor(kAzure-2); }
  for (int i = 0; i < 8; i++){ result3_slice_1p[i]->SetLineColor(kRed-4); }
  for (int i = 0; i < 8; i++){ result4_slice_1p[i]->SetLineColor(kRed+3); }

  for (int i = 0; i < 8; i++){ result1_slice_1p[i]->SetLineWidth(4); }
  for (int i = 0; i < 8; i++){ result2_slice_1p[i]->SetLineWidth(3); }
  for (int i = 0; i < 8; i++){ result3_slice_1p[i]->SetLineWidth(2); }
  for (int i = 0; i < 8; i++){ result4_slice_1p[i]->SetLineWidth(2); }

  for (int i = 0; i < 8; i++){ result2_slice_1p[i]->SetLineStyle(2); }
  for (int i = 0; i < 8; i++){ result4_slice_1p[i]->SetLineStyle(3); }

  if(strstr(outFileTag.Data(),"FSI")){
    result3_cthmu_0p->SetLineColor(kGray+2);
    result3_cthmu_1p->SetLineColor(kGray+2);
    result3_NpHist->SetLineColor(kGray+2);
    result3_LinResult->SetLineColor(kGray+2);

    result4_cthmu_0p->SetLineColor(kGray+3);
    result4_cthmu_1p->SetLineColor(kGray+3);
    result4_NpHist->SetLineColor(kGray+3);
    result4_LinResult->SetLineColor(kGray+3);

    for (int i = 0; i < 10; i++){ result3_slice_0p[i]->SetLineColor(kGray+2); }
    for (int i = 0; i < 10; i++){ result4_slice_0p[i]->SetLineColor(kGray+3); }
    for (int i = 0; i < 8; i++){ result3_slice_1p[i]->SetLineColor(kGray+2); }
    for (int i = 0; i < 8; i++){ result4_slice_1p[i]->SetLineColor(kGray+3); }
  }

  TLegend* leg = new TLegend(0.05, 0.05, 0.95, 0.95);
  leg->AddEntry(data_LinResult,"T2K Fit to Data","ep");
  leg->AddEntry(result1_LinResult, Form("%s, #chi^{2}=%.1f",genName1,result1_chi2), "lf");
  leg->AddEntry(result2_LinResult, Form("%s, #chi^{2}=%.1f",genName2,result2_chi2), "lf");
  if(!ignoreFour) leg->AddEntry(result3_LinResult, Form("%s, #chi^{2}=%.1f",genName3,result3_chi2), "lf");
  leg->AddEntry(result4_LinResult, Form("%s, #chi^{2}=%.1f",genName4,result4_chi2), "lf");
  leg->SetFillColor(kWhite);
  leg->SetFillStyle(0);

  TLegend* leg_small = new TLegend(0.2, 0.3, 0.7, 0.88);
  leg_small->AddEntry(data_LinResult,"T2K Fit to Data","ep");
  leg_small->AddEntry(result1_LinResult, Form("%s, #chi^{2}=%.1f",genName1,result1_chi2), "lf");
  leg_small->AddEntry(result2_LinResult, Form("%s, #chi^{2}=%.1f",genName2,result2_chi2), "lf");
  if(!ignoreFour) leg_small->AddEntry(result3_LinResult, Form("%s, #chi^{2}=%.1f",genName3,result3_chi2), "lf");
  leg_small->AddEntry(result4_LinResult, Form("%s, #chi^{2}=%.1f",genName4,result4_chi2), "lf");
  leg_small->SetFillColor(kWhite);
  leg_small->SetFillStyle(0);

  TLegend* leg_small_right = new TLegend(0.4, 0.3, 0.9, 0.88);
  leg_small_right->AddEntry(data_LinResult,"T2K Fit to Data","ep");
  leg_small_right->AddEntry(result1_LinResult, Form("%s, #chi^{2}=%.1f",genName1,result1_chi2), "lf");
  leg_small_right->AddEntry(result2_LinResult, Form("%s, #chi^{2}=%.1f",genName2,result2_chi2), "lf");
  if(!ignoreFour) leg_small_right->AddEntry(result3_LinResult, Form("%s, #chi^{2}=%.1f",genName3,result3_chi2), "lf");
  leg_small_right->AddEntry(result4_LinResult, Form("%s, #chi^{2}=%.1f",genName4,result4_chi2), "lf");
  leg_small_right->SetFillColor(kWhite);
  leg_small_right->SetFillStyle(0);

  TLegend* leg_NotSoSmall = new TLegend(0.15, 0.15, 0.80, 0.88);
  leg_NotSoSmall->AddEntry(data_LinResult,"T2K Fit to Data","ep");
  leg_NotSoSmall->AddEntry(result1_LinResult, Form("%s, #chi^{2}=%.1f",genName1,result1_chi2), "lf");
  leg_NotSoSmall->AddEntry(result2_LinResult, Form("%s, #chi^{2}=%.1f",genName2,result2_chi2), "lf");
  if(!ignoreFour) leg_NotSoSmall->AddEntry(result3_LinResult, Form("%s, #chi^{2}=%.1f",genName3,result3_chi2), "lf");
  leg_NotSoSmall->AddEntry(result4_LinResult, Form("%s, #chi^{2}=%.1f",genName4,result4_chi2), "lf");
  leg_NotSoSmall->SetFillColor(kWhite);
  leg_NotSoSmall->SetFillStyle(0);

  cout << "Making pretty figures ... " << endl;

  gStyle->SetOptTitle(1);

  data_NpHist->GetXaxis()->SetBinLabel(1, "0");
  data_NpHist->GetXaxis()->SetBinLabel(2, "1");
  data_NpHist->GetXaxis()->SetBinLabel(3, ">1");
  data_NpHist->SetLabelSize(0.07, "X");

  TCanvas* compCanv_comb = new TCanvas("compCanv_comb", "compCanv_comb", 1920, 360);
  compCanv_comb->cd();

  TPad* nprotonsPad = new TPad("nprotonsPad","nprotonsPad",0.0,0.00,0.33,1.00);
  nprotonsPad->cd();
  data_NpHist->Draw();
  result1_NpHist->Draw("sameHIST");
  result2_NpHist->Draw("sameHIST");
  result3_NpHist->Draw("sameHIST");
  result4_NpHist->Draw("sameHIST");
  data_NpHist->Draw("same");

  TPad* lineResultPad = new TPad("lineResultPad","lineResultPad",0.33,0.00,0.66,1.00);
  lineResultPad->cd();
  data_LinResult->GetYaxis()->SetRangeUser(0,0.3);
  data_LinResult->Draw();
  result1_LinResult->Draw("sameHIST");
  result2_LinResult->Draw("sameHIST");
  result3_LinResult->Draw("sameHIST");
  result4_LinResult->Draw("sameHIST");
  data_LinResult->Draw("same");

  TPad* legPadComb = new TPad("legPadComb","legPadComb",0.66,0.00,0.99,1.00);
  legPadComb->cd();
  leg->Draw();
  //result1_LinResult->Draw("histSAME")
  //result2_LinResult->Draw("histSAME")
  //data_LinResult->Draw("same");

  compCanv_comb->cd();

  nprotonsPad->Draw();
  lineResultPad->Draw("same");
  legPadComb->Draw("same");
  compCanv_comb->Write();
  compCanv_comb->SaveAs(Form("%s_comb.png",outFileTag.Data()));
  compCanv_comb->SaveAs(Form("%s_comb.pdf",outFileTag.Data()));

  TCanvas* nprotCanv = new TCanvas();
  nprotCanv->cd();
  data_NpHist->Draw();
  result1_NpHist->Draw("sameHIST");
  result2_NpHist->Draw("sameHIST");
  result3_NpHist->Draw("sameHIST");
  result4_NpHist->Draw("sameHIST");
  data_NpHist->Draw("same");

  nprotCanv->Write("NpCanv");
  nprotCanv->SaveAs(Form("%s_NpOnly.png",outFileTag.Data()));
  nprotCanv->SaveAs(Form("%s_NpOnly.pdf",outFileTag.Data()));



  // 0p canvas ******************************

  TCanvas* compCanv_0p = new TCanvas("compCanv_0p", "compCanv_0p", 1620, 1920);
  compCanv_0p->cd();

  TPad* cthmuPad0p = new TPad("cthmuPad0p","cthmuPad0p",0.00,0.80,0.50,1.00);
  cthmuPad0p->cd();
  data_cthmu_0p->GetYaxis()->SetRangeUser(0,10);
  data_cthmu_0p->SetXTitle("cos(#theta_{#mu})");
  data_cthmu_0p->SetYTitle("#frac{d#sigma}{d cos(#theta_{#mu})} (10^{-39} cm^{2} Nucleon^{-1})");
  data_cthmu_0p->SetNameTitle("result_cthmu","Integrated over all p_{#mu}");
  data_cthmu_0p->Draw();
  result1_cthmu_0p->Draw("sameHIST");
  result2_cthmu_0p->Draw("sameHIST");
  result3_cthmu_0p->Draw("sameHIST");
  result4_cthmu_0p->Draw("sameHIST");
  data_cthmu_0p->Draw("same");
  leg_small->Draw("same");

  std::vector<TPad*> pad;

  for (int i = 0; i < 10; i++){ 
    //cout << i << endl;
    switch(i){
      case 0 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.50,0.80,1.00,1.00));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 1.6);
                  result1_slice_0p[i+1]->SetNameTitle("-0.3 < cos(#theta_{#mu}) < 0.3", "-0.3 < cos(#theta_{#mu}) < 0.3");
                  break;
               }
      case 1 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.60,0.50,0.80));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 1.6);
                  result1_slice_0p[i+1]->SetNameTitle("0.3 < cos(#theta_{#mu}) < 0.6", "0.3 < cos(#theta_{#mu}) < 0.6");
                  break;
               }
      case 2 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.50,0.60,1.00,0.80));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.8);
                  result1_slice_0p[i+1]->SetNameTitle("0.6 < cos(#theta_{#mu}) < 0.7", "0.6 < cos(#theta_{#mu}) < 0.7");
                  break;
               }
      case 3 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.40,0.50,0.60));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 1.0);
                  result1_slice_0p[i+1]->SetNameTitle("0.7 < cos(#theta_{#mu}) < 0.8", "0.7 < cos(#theta_{#mu}) < 0.8");
                  break;
               }
      case 4 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.50,0.40,1.00,0.60));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.6);
                  result1_slice_0p[i+1]->SetNameTitle("0.8 < cos(#theta_{#mu}) < 0.85", "0.8 < cos(#theta_{#mu}) < 0.85");
                  break;
               }
      case 5 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.20,0.50,0.40));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.6);
                  result1_slice_0p[i+1]->SetNameTitle("0.85 < cos(#theta_{#mu}) < 0.9", "0.85 < cos(#theta_{#mu}) < 0.9");
                  break;
               }
      case 6 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.50,0.20,1.00,0.40));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.5);
                  result1_slice_0p[i+1]->SetNameTitle("0.9 < cos(#theta_{#mu}) < 0.94", "0.9 < cos(#theta_{#mu}) < 0.94");
                  break;
               }
      case 7 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.00,0.50,0.20));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.4);
                  result1_slice_0p[i+1]->SetNameTitle("0.94 < cos(#theta_{#mu}) < 0.98", "0.94 < cos(#theta_{#mu}) < 0.98");
                  break;
               }
      case 8 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.50,0.00,1.00,0.20));
                  result1_slice_0p[i+1]->GetYaxis()->SetRangeUser(0, 0.15);
                  result1_slice_0p[i+1]->SetNameTitle("0.98 < cos(#theta_{#mu}) < 1.0", "0.98 < cos(#theta_{#mu}) < 1.0");
                  break;
               }
      case 9 : { 
                  //pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.33,0.00,0.66,0.25));
                  //pad[i]->cd();
                  //leg->Draw();
                  break; 
               }
    }
    if(i<9){
      pad[i]->cd();
      result1_slice_0p[i+1]->SetXTitle("p_{#mu} (GeV)");
      result1_slice_0p[i+1]->SetYTitle("#frac{d#sigma}{d p_{#mu}} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");
      result1_slice_0p[i+1]->Draw("HIST");
      result2_slice_0p[i+1]->Draw("sameHIST");
      if(!ignoreFour) result3_slice_0p[i+1]->Draw("sameHIST");
      result4_slice_0p[i+1]->Draw("sameHIST");
      data_slice_0p[i+1]->Draw("same");
    }
  }

  compCanv_0p->cd();
  cthmuPad0p->Draw();
  for (int i = 0; i < 9/*10*/; i++){ 
    //cout << i << endl;
    pad[i]->Draw("same");
  }
  compCanv_0p->Write();
  compCanv_0p->SaveAs(Form("%s_0p.png",outFileTag.Data()));
  compCanv_0p->SaveAs(Form("%s_0p.pdf",outFileTag.Data()));



  cout << "Finished 0p canvas ... " << endl;

  // ********* Now draw the 1p pad *************************************


  TCanvas* compCanv_1p = new TCanvas("compCanv_1p", "compCanv_1p", 1920, 1080);
  compCanv_1p->cd();

  TPad* cthmuPad1p = new TPad("cthmuPad1p","cthmuPad1p",0.5,0.80,1.00,1.00);
  cthmuPad1p->cd();
  data_cthmu_1p->SetNameTitle("result_cthmu","Integrated over all cos(#theta_{p}) and p_{p}");
  data_cthmu_1p->GetYaxis()->SetRangeUser(0,3.0);
  data_cthmu_1p->Draw();
  result1_cthmu_1p->Draw("sameHIST");
  result2_cthmu_1p->Draw("sameHIST");
  if(!ignoreFour)  result3_cthmu_1p->Draw("sameHIST");
  result4_cthmu_1p->Draw("sameHIST");
  data_cthmu_1p->Draw("same");

  TPad* cthmu1 = new TPad("cthmu1","cthmu1",0.00,0.60,0.50,0.80);
  cthmu1->cd();
  data_slice_1p[0]->SetXTitle("cos(#theta_{p})");
  data_slice_1p[0]->GetYaxis()->SetRangeUser(0,7.0);
  data_slice_1p[0]->Draw();
  result1_slice_1p[0]->Draw("sameHIST");
  result2_slice_1p[0]->Draw("sameHIST");
  if(!ignoreFour)  result3_slice_1p[0]->Draw("sameHIST");
  result4_slice_1p[0]->Draw("sameHIST");
  data_slice_1p[0]->Draw("same");
  leg_NotSoSmall->Draw("same");

  TPad* cthmu2 = new TPad("cthmu2","cthmu2",0.50,0.60,1.00,0.80);
  cthmu2->cd();
  data_slice_1p[1]->SetXTitle("cos(#theta_{p})");
  data_slice_1p[1]->GetYaxis()->SetRangeUser(0,3.0);
  data_slice_1p[1]->Draw();
  result1_slice_1p[1]->Draw("sameHIST");
  result2_slice_1p[1]->Draw("sameHIST");
  if(!ignoreFour)  result3_slice_1p[1]->Draw("sameHIST");
  result4_slice_1p[1]->Draw("sameHIST");
  data_slice_1p[1]->Draw("same");


  TPad* cthmu2_pmom = new TPad("cthmu2_pmom","cthmu2_pmom",0.0,0.40,0.50,0.60);
  cthmu2_pmom->cd();
  data_slice_1p[2]->GetYaxis()->SetRangeUser(0,0.7);
  data_slice_1p[2]->Draw();
  result1_slice_1p[2]->Draw("sameHIST");
  result2_slice_1p[2]->Draw("sameHIST");
  if(!ignoreFour)  result3_slice_1p[2]->Draw("sameHIST");
  result4_slice_1p[2]->Draw("sameHIST");
  data_slice_1p[2]->Draw("same");

  TPad* cthmu3 = new TPad("cthmu3","cthmu3",0.50,0.40,1.00,0.60);
  cthmu3->cd();
  data_slice_1p[3]->SetXTitle("cos(#theta_{p})");
  data_slice_1p[3]->GetYaxis()->SetRangeUser(0,2.0);
  data_slice_1p[3]->Draw();
  result1_slice_1p[3]->Draw("sameHIST");
  result2_slice_1p[3]->Draw("sameHIST");
  if(!ignoreFour)  result3_slice_1p[3]->Draw("sameHIST");
  result4_slice_1p[3]->Draw("sameHIST");
  data_slice_1p[3]->Draw("same");

  TPad* cthmu3_pmom1 = new TPad("cthmu3_pmom1","cthmu3_pmom1",0.0,0.20,0.50,0.40);
  cthmu3_pmom1->cd();
  data_slice_1p[4]->GetYaxis()->SetRangeUser(0,2.0);
  data_slice_1p[4]->Draw();
  result1_slice_1p[4]->Draw("sameHIST");
  result2_slice_1p[4]->Draw("sameHIST");
  if(!ignoreFour)  result3_slice_1p[4]->Draw("sameHIST");
  result4_slice_1p[4]->Draw("sameHIST");
  data_slice_1p[4]->Draw("same");

  TPad* cthmu3_pmom2 = new TPad("cthmu3_pmom2","cthmu3_pmom2",0.50,0.20,1.00,0.40);
  cthmu3_pmom2->cd();
  data_slice_1p[5]->GetYaxis()->SetRangeUser(0,1.2);
  data_slice_1p[5]->Draw();
  result1_slice_1p[5]->Draw("sameHIST");
  result2_slice_1p[5]->Draw("sameHIST");
  if(!ignoreFour)  result3_slice_1p[5]->Draw("sameHIST");
  result4_slice_1p[5]->Draw("sameHIST");
  data_slice_1p[5]->Draw("same");

  TPad* cthmu4 = new TPad("cthmu4","cthmu4",0.0,0.00,0.50,0.20);
  cthmu4->cd();
  data_slice_1p[6]->SetXTitle("cos(#theta_{p})");
  data_slice_1p[6]->GetYaxis()->SetRangeUser(0,1.0);
  data_slice_1p[6]->Draw();
  result1_slice_1p[6]->Draw("sameHIST");
  result2_slice_1p[6]->Draw("sameHIST");
  if(!ignoreFour)  result3_slice_1p[6]->Draw("sameHIST");
  result4_slice_1p[6]->Draw("sameHIST");
  data_slice_1p[6]->Draw("same");

  TPad* cthmu4_pmom = new TPad("cthmu4_pmom","cthmu4_pmom",0.50,0.00,1.00,0.20);
  cthmu4_pmom->cd();
  data_slice_1p[7]->GetYaxis()->SetRangeUser(0,1.6);
  data_slice_1p[7]->Draw();
  result1_slice_1p[7]->Draw("sameHIST");
  result2_slice_1p[7]->Draw("sameHIST");
  if(!ignoreFour)  result3_slice_1p[7]->Draw("sameHIST");
  result4_slice_1p[7]->Draw("sameHIST");
  data_slice_1p[7]->Draw("same");

  TPad* legPad = new TPad("legPad","legPad",0.00,0.00,0.50,0.20);
  legPad->cd();
  leg->Draw();

  compCanv_1p->cd();
  cthmuPad1p->Draw();
  //cthmu1->Draw("same");
  cthmu2->Draw("same");
  cthmu2_pmom->Draw("same");
  cthmu3->Draw("same");
  cthmu3_pmom1->Draw("same");
  cthmu3_pmom2->Draw("same");
  cthmu4->Draw("same");
  cthmu4_pmom->Draw("same");
  legPad->Draw("same");

  compCanv_1p->Write();
  compCanv_1p->SaveAs(Form("%s_1p.png",outFileTag.Data()));
  compCanv_1p->SaveAs(Form("%s_1p.pdf",outFileTag.Data()));

  TCanvas* compCanv_1p_sansleg = new TCanvas("compCanv_1p_sansleg", "compCanv_1p_sansleg", 1620, 1920);
  compCanv_1p_sansleg->cd();

  TPad* nprotonsPad_sansleg = new TPad("nprotonsPad_sansleg","nprotonsPad_sansleg",0.00,0.80,0.50,1.00);
  nprotonsPad_sansleg->cd();
  data_NpHist->Draw();
  result1_NpHist->Draw("sameHIST");
  result2_NpHist->Draw("sameHIST");
  if(!ignoreFour)  result3_NpHist->Draw("sameHIST");
  result4_NpHist->Draw("sameHIST");
  data_NpHist->Draw("same");

  compCanv_1p_sansleg->cd();
  cthmuPad1p->Draw();
  cthmu1->Draw("same");
  cthmu2->Draw("same");
  cthmu2_pmom->Draw("same");
  cthmu3->Draw("same");
  cthmu3_pmom1->Draw("same");
  cthmu3_pmom2->Draw("same");
  cthmu4->Draw("same");
  cthmu4_pmom->Draw("same");
  //legPad->Draw("same");
  nprotonsPad_sansleg->Draw("same");

  compCanv_1p_sansleg->Write();
  compCanv_1p_sansleg->SaveAs(Form("%s_1p_noLeg.png",outFileTag.Data()));
  compCanv_1p_sansleg->SaveAs(Form("%s_1p_noLeg.pdf",outFileTag.Data()));

  data_NpHist->Write("resultNpHist");
  data_cthmu_0p->Write("resultCthmu_0p");
  data_cthmu_1p->Write("resultCthmu_1p");
  for (int i = 0; i < 10; i++){ data_slice_0p[i]->Write(Form("dataslice_0p_%d",i)); }
  for (int i = 0; i < 8; i++){ data_slice_1p[i]->Write(Form("dataslice_1p_%d",i)); }

  //char* saveName = Form("%s_multDifComp_0p.pdf",genName);
  //char* saveNamepng = Form("%s_multDifComp_0p.png",genName);
  //compCanv_0p->SaveAs(saveName);
  //compCanv_0p->SaveAs(saveNamepng);

  cout << "Finished :-D" << endl;

  return;

}
