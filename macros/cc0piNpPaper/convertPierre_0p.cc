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


void convertPierre_0p(TString inFileName, TString outFileName){
  TFile *infile = new TFile(inFileName);
  TVectorD* result = (TVectorD*)infile->Get("Fitted_xsec");
  TMatrixD* cov_syst = (TMatrixD*)infile->Get("RelCov_syst");
  TMatrixD* cov_stat = (TMatrixD*)infile->Get("RelCov_stat");
  TMatrixD* cov = (TMatrixD*)infile->Get("RelCov_stat");

  (*cov)+=(*cov_syst);

  TH1D* linearResult = new TH1D("LinResult", "LinResult",60,0,60);
  TH2D* covhist = new TH2D(cov->GetSub(0,59,0,59));

  TH2Poly* datapoly = new TH2Poly();
  datapoly->SetNameTitle("datapoly","datapoly");

  //                 cth_mu mom_mu cth_mu mom_mu
  datapoly->AddBin( -1.00,  0.00, -0.30,  30.0);//0
//
  datapoly->AddBin( -0.30,  0.00,  0.30,  0.30);//1
  datapoly->AddBin( -0.30,  0.30,  0.30,  0.40);//2
  datapoly->AddBin( -0.30,  0.40,  0.30,  30.0);//3
//
  datapoly->AddBin(  0.30,  0.00,  0.60,  0.30);//4
  datapoly->AddBin(  0.30,  0.30,  0.60,  0.40);//5
  datapoly->AddBin(  0.30,  0.40,  0.60,  0.50);//6
  datapoly->AddBin(  0.30,  0.50,  0.60,  0.60);//7
  datapoly->AddBin(  0.30,  0.60,  0.60,  30.0);//8
//
  datapoly->AddBin(  0.60,  0.00,  0.70,  0.30);//9
  datapoly->AddBin(  0.60,  0.30,  0.70,  0.40);//10
  datapoly->AddBin(  0.60,  0.40,  0.70,  0.50);//11
  datapoly->AddBin(  0.60,  0.50,  0.70,  0.60);//12
  datapoly->AddBin(  0.60,  0.60,  0.70,  30.0);//13
//
  datapoly->AddBin(  0.70,  0.00,  0.80,  0.30);//14
  datapoly->AddBin(  0.70,  0.30,  0.80,  0.40);//15
  datapoly->AddBin(  0.70,  0.40,  0.80,  0.50);//16
  datapoly->AddBin(  0.70,  0.50,  0.80,  0.60);//17
  datapoly->AddBin(  0.70,  0.60,  0.80,  0.70);//18
  datapoly->AddBin(  0.70,  0.70,  0.80,  0.80);//19
  datapoly->AddBin(  0.70,  0.80,  0.80,  30.0);//20
//
  //datapoly->AddBin(  0.80,  0.00,  0.85,  0.30);//21 // PROBLEM HERE!!!!
  //datapoly->AddBin(  0.80,  0.30,  0.85,  0.40);//22
  //datapoly->AddBin(  0.80,  0.40,  0.85,  0.50);//23
  //datapoly->AddBin(  0.80,  0.50,  0.85,  0.60);//24
  //datapoly->AddBin(  0.80,  0.60,  0.85,  0.70);//25
  //datapoly->AddBin(  0.80,  0.70,  0.85,  0.80);//26
  //datapoly->AddBin(  0.80,  0.80,  0.85,  30.0);//27
  datapoly->AddBin(  0.80,  0.00,  0.85,  0.40);//21 
  datapoly->AddBin(  0.80,  0.40,  0.85,  0.50);//22
  datapoly->AddBin(  0.80,  0.50,  0.85,  0.60);//23
  datapoly->AddBin(  0.80,  0.60,  0.85,  0.70);//24
  datapoly->AddBin(  0.80,  0.70,  0.85,  0.80);//25
  datapoly->AddBin(  0.80,  0.80,  0.85,  30.0);//26
//
  datapoly->AddBin(  0.85,  0.00,  0.90,  0.30);//27
  datapoly->AddBin(  0.85,  0.30,  0.90,  0.40);//28
  datapoly->AddBin(  0.85,  0.40,  0.90,  0.50);//29
  datapoly->AddBin(  0.85,  0.50,  0.90,  0.60);//30
  datapoly->AddBin(  0.85,  0.60,  0.90,  0.70);//31
  datapoly->AddBin(  0.85,  0.70,  0.90,  0.80);//32
  datapoly->AddBin(  0.85,  0.80,  0.90,  1.00);//33
  datapoly->AddBin(  0.85,  1.00,  0.90,  30.0);//34
//
  datapoly->AddBin(  0.90,  0.00,  0.94,  0.40);//35
  datapoly->AddBin(  0.90,  0.40,  0.94,  0.50);//36
  datapoly->AddBin(  0.90,  0.50,  0.94,  0.60);//37
  datapoly->AddBin(  0.90,  0.60,  0.94,  0.70);//38
  datapoly->AddBin(  0.90,  0.70,  0.94,  0.80);//39
  datapoly->AddBin(  0.90,  0.80,  0.94,  1.25);//40
  datapoly->AddBin(  0.90,  1.25,  0.94,  30.0);//41
//
  datapoly->AddBin(  0.94,  0.00,  0.98,  0.40);//42
  datapoly->AddBin(  0.94,  0.40,  0.98,  0.50);//43
  datapoly->AddBin(  0.94,  0.50,  0.98,  0.60);//44
  datapoly->AddBin(  0.94,  0.60,  0.98,  0.70);//45
  datapoly->AddBin(  0.94,  0.70,  0.98,  0.80);//46
  datapoly->AddBin(  0.94,  0.80,  0.98,  1.00);//47
  datapoly->AddBin(  0.94,  1.00,  0.98,  1.25);//48
  datapoly->AddBin(  0.94,  1.25,  0.98,  1.50);//49
  datapoly->AddBin(  0.94,  1.50,  0.98,  2.00);//50
  datapoly->AddBin(  0.94,  2.00,  0.98,  30.0);//51
//
  datapoly->AddBin(  0.98,  0.00,  1.00,  0.50);//52
  datapoly->AddBin(  0.98,  0.50,  1.00,  0.65);//53
  datapoly->AddBin(  0.98,  0.65,  1.00,  0.80);//54
  datapoly->AddBin(  0.98,  0.80,  1.00,  1.25);//55
  datapoly->AddBin(  0.98,  1.25,  1.00,  2.00);//56
  datapoly->AddBin(  0.98,  2.00,  1.00,  3.00);//57
  datapoly->AddBin(  0.98,  3.00,  1.00,  5.00);//58
  datapoly->AddBin(  0.98,  5.00,  1.00,  30.0);//59

  for(int b = 0; b < datapoly->GetNumberOfBins(); b++){
    datapoly->SetBinContent(b+1, (*result)[b]);
    linearResult->SetBinContent(b+1, (*result)[b]);
  }

  //Get 1D slices as well: 
  //Integrated over muon angle: 
  Double_t result_cthmu_bins[11] = {-1.0, -0.3, 0.3, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.98, 1.00};
  TH1D* result_cthmu = new TH1D("result_cthmu","result_cthmu",10,result_cthmu_bins);
  for(int b = 0; b < datapoly->GetNumberOfBins(); b++){
    if(b==0) result_cthmu->SetBinContent(1, result_cthmu->GetBinContent(1)+datapoly->GetBinContent(b+1));  //-1 -0.3 
    else if(b<4)  result_cthmu->SetBinContent(2, result_cthmu->GetBinContent(2)+datapoly->GetBinContent(b+1)); //-0.3
    else if(b<9)  result_cthmu->SetBinContent(3, result_cthmu->GetBinContent(3)+datapoly->GetBinContent(b+1)); // 0.3
    else if(b<14) result_cthmu->SetBinContent(4, result_cthmu->GetBinContent(4)+datapoly->GetBinContent(b+1));// 0.6
    else if(b<21) result_cthmu->SetBinContent(5, result_cthmu->GetBinContent(5)+datapoly->GetBinContent(b+1)); // 0.7
    else if(b<28) result_cthmu->SetBinContent(6, result_cthmu->GetBinContent(6)+datapoly->GetBinContent(b+1)); // 0.8
    else if(b<36) result_cthmu->SetBinContent(7, result_cthmu->GetBinContent(7)+datapoly->GetBinContent(b+1)); // 0.85
    else if(b<43) result_cthmu->SetBinContent(8, result_cthmu->GetBinContent(8)+datapoly->GetBinContent(b+1)); // 0.9
    else if(b<53) result_cthmu->SetBinContent(9, result_cthmu->GetBinContent(9)+datapoly->GetBinContent(b+1)); // 0.94
    else          result_cthmu->SetBinContent(10,result_cthmu->GetBinContent(10)+datapoly->GetBinContent(b+1));// 0.98
  }
  result_cthmu->Scale(1, "width");

  //Zeroth muon anglular bin -1 -0.3:
  int globalBinCount=1;
  Double_t result_cthmu0_bins[5] = {0.00, 30.0};
  TH1D* result_cthmu0 = new TH1D("result_cthmu0","result_cthmu0",1,result_cthmu0_bins);
  for(int b = 0; b < result_cthmu0->GetNbinsX(); b++){
    result_cthmu0->SetBinContent(b+1, result_cthmu0->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu0->Scale(1, "width");

  //First muon anglular bin -0.3 0.3:
  Double_t result_cthmu1_bins[4] = {0.00,  0.30,  0.40, 1.0};
  TH1D* result_cthmu1 = new TH1D("result_cthmu1","result_cthmu1",3,result_cthmu1_bins);
  for(int b = 0; b < result_cthmu1->GetNbinsX(); b++){
    result_cthmu1->SetBinContent(b+1, result_cthmu1->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu1->Scale(1, "width");
  //result_cthmu1->SetBinContent(3, result_cthmu1->GetBinContent(3)*(result_cthmu1_bins[3]-result_cthmu1_bins[2])/(30.0-result_cthmu1_bins[3]));   // Scale for the true width of the last bin

  //Second muon anglular bin 0.3 0.6:
  Double_t result_cthmu2_bins[6] = {0.00,  0.30,  0.40,  0.5, 0.6, 1.0};
  TH1D* result_cthmu2 = new TH1D("result_cthmu2","result_cthmu2",5,result_cthmu2_bins);
  for(int b = 0; b < result_cthmu2->GetNbinsX(); b++){
    result_cthmu2->SetBinContent(b+1, result_cthmu2->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu2->Scale(1, "width");
  //result_cthmu2->SetBinContent(5, result_cthmu2->GetBinContent(5)*(result_cthmu2_bins[5]-result_cthmu2_bins[4])/(30.0-result_cthmu2_bins[5]));   // Scale for the true width of the last bin

  //Third muon anglular bin 0.6 0.7:
  Double_t result_cthmu3_bins[6] = {0.00,  0.30,  0.40,  0.5, 0.6, 1.0};
  TH1D* result_cthmu3 = new TH1D("result_cthmu3","result_cthmu3",5,result_cthmu3_bins);
  for(int b = 0; b < result_cthmu3->GetNbinsX(); b++){
    result_cthmu3->SetBinContent(b+1, result_cthmu3->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu3->Scale(1, "width");
  //result_cthmu3->SetBinContent(5, result_cthmu3->GetBinContent(5)*(result_cthmu3_bins[5]-result_cthmu3_bins[4])/(30.0-result_cthmu3_bins[5]));   // Scale for the true width of the last bin

  //Fourth muon anglular bin 0.7-0.8:
  Double_t result_cthmu4_bins[8] = {0.00, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0};
  TH1D* result_cthmu4 = new TH1D("result_cthmu4","result_cthmu4",7,result_cthmu4_bins);
  for(int b = 0; b < result_cthmu4->GetNbinsX(); b++){
    result_cthmu4->SetBinContent(b+1, result_cthmu4->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu4->Scale(1, "width");
  //result_cthmu4->SetBinContent(7, result_cthmu4->GetBinContent(7)*(result_cthmu4_bins[7]-result_cthmu4_bins[6])/(30.0-result_cthmu4_bins[7]));   // Scale for the true width of the last bin

  //Fifth muon anglular bin 0.8-0.85:
  Double_t result_cthmu5_bins[7] = {0.00, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0};
  TH1D* result_cthmu5 = new TH1D("result_cthmu5","result_cthmu5",6,result_cthmu5_bins);
  for(int b = 0; b < result_cthmu5->GetNbinsX(); b++){
    result_cthmu5->SetBinContent(b+1, result_cthmu5->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu5->Scale(1, "width");
  //result_cthmu5->SetBinContent(6, result_cthmu5->GetBinContent(6)*(result_cthmu5_bins[6]-result_cthmu5_bins[5])/(30.0-result_cthmu5_bins[6]));   // Scale for the true width of the last bin

  //Sixth muon anglular bin 0.85-0.90:
  Double_t result_cthmu6_bins[9] = {0.00, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0, 1.5};
  TH1D* result_cthmu6 = new TH1D("result_cthmu6","result_cthmu6",8,result_cthmu6_bins);
  for(int b = 0; b < result_cthmu6->GetNbinsX(); b++){
    result_cthmu6->SetBinContent(b+1, result_cthmu6->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu6->Scale(1, "width");
  //result_cthmu6->SetBinContent(8, result_cthmu6->GetBinContent(8)*(result_cthmu6_bins[8]-result_cthmu6_bins[7])/(30.0-result_cthmu6_bins[8]));   // Scale for the true width of the last bin

  //Seventh muon anglular bin 0.90-0.94:
  Double_t result_cthmu7_bins[8] = {0.00, 0.40, 0.50, 0.60, 0.70, 0.80, 1.25, 2.0};
  TH1D* result_cthmu7 = new TH1D("result_cthmu7","result_cthmu7",7,result_cthmu7_bins);
  for(int b = 0; b < result_cthmu7->GetNbinsX(); b++){
    result_cthmu7->SetBinContent(b+1, result_cthmu7->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu7->Scale(1, "width");
  //result_cthmu7->SetBinContent(7, result_cthmu7->GetBinContent(7)*(result_cthmu7_bins[7]-result_cthmu7_bins[6])/(30.0-result_cthmu7_bins[7]));   // Scale for the true width of the last bin

  //Eigth muon anglular bin 0.94-0.98:
  Double_t result_cthmu8_bins[11] = {0.00, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0, 1.25, 1.5, 2.0, 3.0};
  TH1D* result_cthmu8 = new TH1D("result_cthmu8","result_cthmu8",10,result_cthmu8_bins);
  for(int b = 0; b < result_cthmu8->GetNbinsX(); b++){
    result_cthmu8->SetBinContent(b+1, result_cthmu8->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu8->Scale(1, "width");
  //result_cthmu8->SetBinContent(10, result_cthmu8->GetBinContent(10)*(result_cthmu8_bins[10]-result_cthmu8_bins[9])/(30.0-result_cthmu8_bins[10]));   // Scale for the true width of the last bin

  //Ninth muon anglular bin 0.98-1.00:
  Double_t result_cthmu9_bins[9] = {0.00, 0.50, 0.65, 0.80, 1.25, 2.0, 3.0, 5.0, 8.0};
  TH1D* result_cthmu9 = new TH1D("result_cthmu9","result_cthmu9",8,result_cthmu9_bins);
  for(int b = 0; b < result_cthmu9->GetNbinsX(); b++){
    result_cthmu9->SetBinContent(b+1, result_cthmu9->GetBinContent(b+1)+datapoly->GetBinContent(globalBinCount));
    globalBinCount++;
  }
  result_cthmu9->Scale(1, "width");
  //result_cthmu9->SetBinContent(8, result_cthmu9->GetBinContent(8)*(result_cthmu9_bins[8]-result_cthmu9_bins[7])/(30.0-result_cthmu9_bins[8]));   // Scale for the true width of the last bin


  TFile *outfile = new TFile(outFileName,"recreate");
  outfile->cd();

  linearResult->Write("LinResult");
  covhist->Write("CovMatrix");

  datapoly->Write();

  result_cthmu->Write();
  result_cthmu0->Write("dataslice_0");
  result_cthmu1->Write("dataslice_1");
  result_cthmu2->Write("dataslice_2");
  result_cthmu3->Write("dataslice_3");
  result_cthmu4->Write("dataslice_4");
  result_cthmu5->Write("dataslice_5");
  result_cthmu6->Write("dataslice_6");
  result_cthmu7->Write("dataslice_7");
  result_cthmu8->Write("dataslice_8");
  result_cthmu9->Write("dataslice_9");

  return;
}


