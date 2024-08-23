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


using namespace std;


void convertPierre(TString inFileName, TString outFileName){
  TFile *infile = new TFile(inFileName);
  TVectorD* result = (TVectorD*)infile->Get("Fitted_xsec");
  TMatrixD* cov_syst = (TMatrixD*)infile->Get("RelCov_syst");
  TMatrixD* cov_stat = (TMatrixD*)infile->Get("RelCov_stat");
  TMatrixD* cov = (TMatrixD*)infile->Get("RelCov_stat");

  (*cov)+=(*cov_syst);

  cov->Print();

  TH1D* linearResult = new TH1D("LinResult", "LinResult",32,0,32);
  TH2D* covhist = new TH2D(cov->GetSub(60,91,60,91));

  std::vector<TH2Poly*> result_slice;
  for (int i = 0; i < 4; i++){ 
    result_slice.push_back(new TH2Poly());
    result_slice[i]->SetNameTitle(Form("dataslice_%i",i),Form("dataslice_%i",i));
  }

  // -1.00<cth_mu<-0.30    cth_p   mom_p  cth_p  mom_p
  result_slice[0]->AddBin( -1.00,  0.50,  0.87,  30.0);
  result_slice[0]->AddBin(  0.87,  0.50,  0.94,  30.0);
  result_slice[0]->AddBin(  0.94,  0.50,  0.97,  30.0);
  result_slice[0]->AddBin(  0.97,  0.50,  1.00,  30.0);
  // -0.30<cth_mu<0.30     cth_p   mom_p  cth_p  mom_p
  result_slice[1]->AddBin( -1.00,  0.50,  0.75,  30.0);
  result_slice[1]->AddBin(  0.75,  0.50,  0.85,  30.0);
  result_slice[1]->AddBin(  0.85,  0.50,  0.94,  0.68);//-|-
  result_slice[1]->AddBin(  0.85,  0.68,  0.94,  0.78);//-|
  result_slice[1]->AddBin(  0.85,  0.78,  0.94,  0.90);//-|
  result_slice[1]->AddBin(  0.85,  0.90,  0.94,  30.0);//-|-
  result_slice[1]->AddBin(  0.94,  0.50,  1.00,  30.0);
  //  0.30<cth_mu<0.80    cth_p   mom_p  cth_p  mom_p
  result_slice[2]->AddBin( -1.00,  0.50,  0.30,  30.0);
  result_slice[2]->AddBin(  0.30,  0.50,  0.50,  30.0);
  result_slice[2]->AddBin(  0.50,  0.50,  0.80,  0.60);//-|-
  result_slice[2]->AddBin(  0.50,  0.60,  0.80,  0.70);//-|
  result_slice[2]->AddBin(  0.50,  0.70,  0.80,  0.80);//-|
  result_slice[2]->AddBin(  0.50,  0.80,  0.80,  0.90);//-|
  result_slice[2]->AddBin(  0.50,  0.90,  0.80,  30.0);//-|-
  result_slice[2]->AddBin(  0.80,  0.50,  1.00,  0.60);//-|-
  result_slice[2]->AddBin(  0.80,  0.60,  1.00,  0.70);//-|
  result_slice[2]->AddBin(  0.80,  0.70,  1.00,  0.80);//-|
  result_slice[2]->AddBin(  0.80,  0.80,  1.00,  1.00);//-|
  result_slice[2]->AddBin(  0.80,  1.00,  1.00,  30.0);//-|-
  //  0.80<cth_mu<1.00    cth_p   mom_p  cth_p  mom_p
  result_slice[3]->AddBin( -1.00,  0.50,  0.00,  30.0);
  result_slice[3]->AddBin(  0.00,  0.50,  0.30,  30.0);
  result_slice[3]->AddBin(  0.30,  0.50,  0.80,  0.60);//-|-
  result_slice[3]->AddBin(  0.30,  0.60,  0.80,  0.70);//-|
  result_slice[3]->AddBin(  0.30,  0.70,  0.80,  0.80);//-|
  result_slice[3]->AddBin(  0.30,  0.80,  0.80,  0.90);//-|
  result_slice[3]->AddBin(  0.30,  0.90,  0.80,  1.10);//-|
  result_slice[3]->AddBin(  0.30,  1.10,  0.80,  30.0);//-|-
  result_slice[3]->AddBin(  0.80,  0.50,  1.00,  30.0);

  int binIndex=60;
  for (int i = 0; i < 4; i++){ 
    for(int b = 0; b < result_slice[i]->GetNumberOfBins(); b++){
      result_slice[i]->SetBinContent(b+1, (*result)[binIndex]);
      binIndex++;
    }
  }
  binIndex=60;
  for (int bin = 1; bin < 33; bin++){ 
    linearResult->SetBinContent(bin, (*result)[binIndex]);
    binIndex++;
  }

  //Get 1D slices as well: 
  //Integrated over muon angle: 
  Double_t result_cthmu_bins[5] = {-1.0, -0.3, 0.3, 0.8, 1.00};
  TH1D* result_cthmu = new TH1D("result_cthmu","result_cthmu",4,result_cthmu_bins);
  int globalBinCount=0;
  for (int i = 0; i < 4; i++){ 
    for(int b = 0; b < result_slice[i]->GetNumberOfBins(); b++){
      result_cthmu->SetBinContent(i+1, result_cthmu->GetBinContent(i+1)+result_slice[i]->GetBinContent(b+1));
      //cout << "Bin content in bin " << 60 + globalBinCount << " is " << result_slice[i]->GetBinContent(b+1) << endl;
      //cout << "Bin content in vector element " << 60 + globalBinCount << " is " << (*result)[60 + globalBinCount] << endl;
      //globalBinCount++;
    }
  }
  result_cthmu->Scale(1, "width");

  //First muon anglular bin:
  Double_t result_cthmu1_bins[5] = {-1.00,  0.87,  0.94,  0.97, 1.00};
  TH1D* result_cthmu1 = new TH1D("result_cthmu1","result_cthmu1",4,result_cthmu1_bins);
  cout << "Number of bins in slice 0 is " << result_slice[0]->GetNumberOfBins() << endl;
  for(int b = 0; b < result_slice[0]->GetNumberOfBins(); b++){
    result_cthmu1->SetBinContent(b+1, result_cthmu1->GetBinContent(b+1)+result_slice[0]->GetBinContent(b+1));
  }
  result_cthmu1->Scale(1, "width");

  //Second muon anglular bin:
  int pAngBin=1;
  int pMomBin=1;
  Double_t result_cthmu2_bins[5] = {-1.00,  0.75,  0.85,  0.94, 1.00};
  Double_t result_cthmu2_pmom_bins[5] = {0.50,  0.68,  0.78,  0.90, 1.50};
  TH1D* result_cthmu2 = new TH1D("result_cthmu2","result_cthmu2",4,result_cthmu2_bins);
  TH1D* result_cthmu2_pmom = new TH1D("result_cthmu2_pmom","result_cthmu2_pmom",4,result_cthmu2_pmom_bins);
  cout << "Number of bins in slice 1 is " << result_slice[1]->GetNumberOfBins() << endl;
  for(int b = 0; b < result_slice[1]->GetNumberOfBins(); b++){
    result_cthmu2->SetBinContent(pAngBin, result_cthmu2->GetBinContent(pAngBin)+result_slice[1]->GetBinContent(b+1));
    if(b==0 || b==1 || b==5) pAngBin++;
    else{
      result_cthmu2_pmom->SetBinContent(pMomBin, result_cthmu2_pmom->GetBinContent(pMomBin)+result_slice[1]->GetBinContent(b+1));
      pMomBin++;
    }
  }
  result_cthmu2->Scale(1, "width");
  result_cthmu2_pmom->Scale(1, "width");
  //result_cthmu2_pmom->SetBinContent(4, result_cthmu2_pmom->GetBinContent(4)*(result_cthmu2_pmom_bins[4]-result_cthmu2_pmom_bins[3])/(30.0-result_cthmu2_pmom_bins[4]));   // Scale for the true width of the last bin


  //Third muon angular bin:
  pAngBin=1;
  pMomBin=1;
  Double_t result_cthmu3_bins[5] = {-1.00,  0.30,  0.50,  0.80, 1.00};
  Double_t result_cthmu3_pmom1_bins[6] = {0.50, 0.60, 0.70, 0.80, 0.90, 1.5};
  Double_t result_cthmu3_pmom2_bins[6] = {0.50, 0.60, 0.70, 0.80, 1.00, 1.5};
  TH1D* result_cthmu3 = new TH1D("result_cthmu3","result_cthmu3",4,result_cthmu3_bins);
  TH1D* result_cthmu3_pmom1 = new TH1D("result_cthmu3_pmom1","result_cthmu3_pmom1",5,result_cthmu3_pmom1_bins);
  TH1D* result_cthmu3_pmom2 = new TH1D("result_cthmu3_pmom2","result_cthmu3_pmom2",5,result_cthmu3_pmom2_bins);
  cout << "Number of bins in slice 2 is " << result_slice[2]->GetNumberOfBins() << endl;
  for(int b = 0; b < result_slice[2]->GetNumberOfBins(); b++){
    result_cthmu3->SetBinContent(pAngBin, result_cthmu3->GetBinContent(pAngBin)+result_slice[2]->GetBinContent(b+1));
    if(b==0 || b==1 || b==6) pAngBin++;
    if(b>=2 && b<=6){
      result_cthmu3_pmom1->SetBinContent(pMomBin, result_cthmu3_pmom1->GetBinContent(pMomBin)+result_slice[2]->GetBinContent(b+1));
      pMomBin++;
    }
    if(b==6)pMomBin=1;
    else if(b>6){
      result_cthmu3_pmom2->SetBinContent(pMomBin, result_cthmu3_pmom2->GetBinContent(pMomBin)+result_slice[2]->GetBinContent(b+1));
      pMomBin++;
    }
  }
  result_cthmu3->Scale(1, "width");
  result_cthmu3_pmom1->Scale(1, "width");
  //result_cthmu3_pmom1->SetBinContent(5, result_cthmu3_pmom1->GetBinContent(5)*(result_cthmu3_pmom1_bins[5]-result_cthmu3_pmom1_bins[4])/(30.0-result_cthmu3_pmom1_bins[5]));   // Scale for the true width of the last bin
  result_cthmu3_pmom2->Scale(1, "width");
  //result_cthmu3_pmom2->SetBinContent(5, result_cthmu3_pmom2->GetBinContent(5)*(result_cthmu3_pmom2_bins[5]-result_cthmu3_pmom2_bins[4])/(30.0-result_cthmu3_pmom2_bins[5]));   // Scale for the true width of the last bin

  //Fourth muon angular bin:
  pAngBin=1;
  pMomBin=1;
  Double_t result_cthmu4_bins[5] = {-1.00,  0.00,  0.30,  0.80, 1.00};
  Double_t result_cthmu4_pmom_bins[7] = {0.50, 0.60, 0.70, 0.80, 0.90, 1.10, 1.5};
  TH1D* result_cthmu4 = new TH1D("result_cthmu4","result_cthmu4",4,result_cthmu4_bins);
  TH1D* result_cthmu4_pmom = new TH1D("result_cthmu4_pmom","result_cthmu4_pmom",6,result_cthmu4_pmom_bins);
  cout << "Number of bins in slice 3 is " << result_slice[3]->GetNumberOfBins() << endl;
  for(int b = 0; b < result_slice[3]->GetNumberOfBins(); b++){
    result_cthmu4->SetBinContent(pAngBin, result_cthmu4->GetBinContent(pAngBin)+result_slice[3]->GetBinContent(b+1));
    if(b==0 || b==1 || b==7) pAngBin++;
    else{
      result_cthmu4_pmom->SetBinContent(pMomBin, result_cthmu4_pmom->GetBinContent(pMomBin)+result_slice[3]->GetBinContent(b+1));
      pMomBin++;
    }
  }
  result_cthmu4->Scale(1, "width");
  result_cthmu4_pmom->Scale(1, "width");
  //result_cthmu4_pmom->SetBinContent(6, result_cthmu4_pmom->GetBinContent(6)*(result_cthmu4_pmom_bins[6]-result_cthmu4_pmom_bins[5])/(30.0-result_cthmu4_pmom_bins[6]));   // Scale for the true width of the last bin


  TFile *outfile = new TFile(outFileName,"recreate");
  outfile->cd();

  linearResult->Write("LinResult");
  covhist->Write("CovMatrix");

  for (int i = 0; i < 4; i++){ 
    result_slice[i]->Write();
  }

  result_cthmu->Write();
  result_cthmu1->Write();
  result_cthmu2->Write();
  result_cthmu3->Write();
  result_cthmu4->Write();
  result_cthmu2_pmom->Write();
  result_cthmu3_pmom1->Write();
  result_cthmu3_pmom2->Write();
  result_cthmu4_pmom->Write();
  return;
}


