/******************************************************

Code to take output of the fitter and produce a corrected
Nevents spectrum compared to the MC and fake data truth

Author: Stephen Dolan
Date Created: Mar 2018

******************************************************/


#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <assert.h>

#include <TCanvas.h>
#include <TH1F.h>
#include <TTree.h>
#include <TString.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TMath.h>

//#define OVERIDEBINNING


using namespace std;



void calcXsecWithErrors (TString mcFilename, TString fitResultFilename, TString fakeDataFilename, TString outputFileName, double potRatio){
  TFile* mcFile = new TFile(mcFilename);
  TFile* fitResultFile = new TFile(fitResultFilename);
  TFile* fakeDataFile = new TFile(fakeDataFilename); 
  TFile* outputFile = new TFile(outputFileName, "RECREATE"); 
 

  TTree* selEvtsTree = (TTree*) mcFile->Get("selectedEvents");
  TTree* trueEvtsTree = (TTree*) mcFile->Get("trueEvents");
  TTree* selEvtsFakeDataTree = (TTree*) fakeDataFile->Get("selectedEvents");
  TTree* trueEvtsFakeDataTree = (TTree*) fakeDataFile->Get("trueEvents");

  TH1D* fitParamValues = (TH1D*) fitResultFile->Get("paramhist_parpar_ccqe_result");
  TH1D* fitParamErrors = (TH1D*) fitResultFile->Get("paramerrhist_parpar_ccqe_result");

  TH1D* histForBinning = (TH1D*) fitResultFile->Get("MuTPCpTPC_RecD1_cc0pi1p_nominal");


  #ifdef OVERIDEBINNING
    const int nbinsAuto = 8;
    const Double_t binsAutoAr[nbinsAuto+1] =  {0.0, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 1.1};
    const TArrayD* binsAuto = new TArrayD(nbinsAuto+1, binsAutoAr);
  #else
    const int nbinsAuto = histForBinning->GetXaxis()->GetNbins();
    const TArrayD* binsAuto = histForBinning->GetXaxis()->GetXbins();  
  #endif

    
  TH1D* histForBinningSample = (TH1D*) fitResultFile->Get("evhist_sam0_iter0_mc");
  const int nbinsAutoSample = histForBinningSample->GetXaxis()->GetNbins();
  const TArrayD* binsAutoSample = histForBinningSample->GetXaxis()->GetXbins();  
  
  TH1D* totalSpectMC_allSam = new TH1D("totalSpectMC_allSam","totalSpectMC_allSam",nbinsAutoSample,binsAutoSample->GetArray());
  TH1D* totalSpectFit_allSam = new TH1D("totalSpectFit_allSam","totalSpectFit_allSam",nbinsAutoSample,binsAutoSample->GetArray());
  TH1D* totalSpectFD_allSam = new TH1D("totalSpectFD_allSam","totalSpectFD_allSam",nbinsAutoSample,binsAutoSample->GetArray());

  TH1D* totalSpectMC_sam0 = (TH1D*) fitResultFile->Get("evhist_sam0_iter0_mc");
  //TH1D* totalSpectFit_sam0 = (TH1D*) fitResultFile->Get("evhist_sam0_iter"+finalIter+"_pred");  //Need to insert correct iter number
  TH1D* totalSpectFit_sam0 = (TH1D*) fitResultFile->Get("evhist_sam0_finaliter_pred");  //Need to insert correct iter number
  TH1D* totalSpectFD_sam0 = (TH1D*) fitResultFile->Get("evhist_sam0_iter0_data");

  TH1D* totalSpectMC_sam1 = (TH1D*) fitResultFile->Get("evhist_sam1_iter0_mc");
  TH1D* totalSpectFit_sam1 = (TH1D*) fitResultFile->Get("evhist_sam1_finaliter_pred");
  TH1D* totalSpectFD_sam1 = (TH1D*) fitResultFile->Get("evhist_sam1_iter0_data");
  if(!totalSpectMC_sam1) totalSpectMC_sam1 = new TH1D("NullSam1","NullSam1",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFit_sam1) totalSpectFit_sam1 = new TH1D("NullSam1","NullSam1",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFD_sam1) totalSpectFD_sam1 = new TH1D("NullSam1","NullSam1",nbinsAutoSample,binsAutoSample->GetArray());

  totalSpectMC_allSam->Add(totalSpectMC_sam0,totalSpectMC_sam1);
  totalSpectFit_allSam->Add(totalSpectFit_sam0,totalSpectFit_sam1);
  totalSpectFD_allSam->Add(totalSpectFD_sam0,totalSpectFD_sam1);

  TH1D* totalSpectMC_sam2 = (TH1D*) fitResultFile->Get("evhist_sam2_iter0_mc");
  TH1D* totalSpectFit_sam2 = (TH1D*) fitResultFile->Get("evhist_sam2_finaliter_pred");
  TH1D* totalSpectFD_sam2 = (TH1D*) fitResultFile->Get("evhist_sam2_iter0_data");
  if(totalSpectMC_sam2) totalSpectMC_allSam->Add(totalSpectMC_sam2);
  if(totalSpectFit_sam2)totalSpectFit_allSam->Add(totalSpectFit_sam2);
  if(totalSpectFD_sam2) totalSpectFD_allSam->Add(totalSpectFD_sam2);
  if(!totalSpectMC_sam2) totalSpectMC_sam2 = new TH1D("NullSam2","NullSam2",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFit_sam2) totalSpectFit_sam2 = new TH1D("NullSam2","NullSam2",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFD_sam2) totalSpectFD_sam2 = new TH1D("NullSam2","NullSam2",nbinsAutoSample,binsAutoSample->GetArray());

  TH1D* totalSpectMC_sam3 = (TH1D*) fitResultFile->Get("evhist_sam3_iter0_mc");
  TH1D* totalSpectFit_sam3 = (TH1D*) fitResultFile->Get("evhist_sam3_finaliter_pred");
  TH1D* totalSpectFD_sam3 = (TH1D*) fitResultFile->Get("evhist_sam3_iter0_data");
  if(totalSpectMC_sam3) totalSpectMC_allSam->Add(totalSpectMC_sam3);
  if(totalSpectFit_sam3)totalSpectFit_allSam->Add(totalSpectFit_sam3);
  if(totalSpectFD_sam3) totalSpectFD_allSam->Add(totalSpectFD_sam3);
  if(!totalSpectMC_sam3) totalSpectMC_sam3 = new TH1D("NullSam3","NullSam3",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFit_sam3) totalSpectFit_sam3 = new TH1D("NullSam3","NullSam3",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFD_sam3) totalSpectFD_sam3 = new TH1D("NullSam3","NullSam3",nbinsAutoSample,binsAutoSample->GetArray());

  TH1D* totalSpectMC_sam4 = (TH1D*) fitResultFile->Get("evhist_sam4_iter0_mc");
  TH1D* totalSpectFit_sam4 = (TH1D*) fitResultFile->Get("evhist_sam4_finaliter_pred");
  TH1D* totalSpectFD_sam4 = (TH1D*) fitResultFile->Get("evhist_sam4_iter0_data");
  if(totalSpectMC_sam4) totalSpectMC_allSam->Add(totalSpectMC_sam4);
  if(totalSpectFit_sam4)totalSpectFit_allSam->Add(totalSpectFit_sam4);
  if(totalSpectFD_sam4) totalSpectFD_allSam->Add(totalSpectFD_sam4);
  if(!totalSpectMC_sam4) totalSpectMC_sam4 = new TH1D("NullSam4","NullSam4",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFit_sam4) totalSpectFit_sam4 = new TH1D("NullSam4","NullSam4",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFD_sam4) totalSpectFD_sam4 = new TH1D("NullSam4","NullSam4",nbinsAutoSample,binsAutoSample->GetArray());

  TH1D* totalSpectMC_sam5 = (TH1D*) fitResultFile->Get("evhist_sam5_iter0_mc");
  TH1D* totalSpectFit_sam5 = (TH1D*) fitResultFile->Get("evhist_sam5_finaliter_pred");
  TH1D* totalSpectFD_sam5 = (TH1D*) fitResultFile->Get("evhist_sam5_iter0_data");
  if(totalSpectMC_sam5) totalSpectMC_allSam->Add(totalSpectMC_sam5);
  if(totalSpectFit_sam5)totalSpectFit_allSam->Add(totalSpectFit_sam5);
  if(totalSpectFD_sam5) totalSpectFD_allSam->Add(totalSpectFD_sam5);
  if(!totalSpectMC_sam5) totalSpectMC_sam5 = new TH1D("NullSam5","NullSam5",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFit_sam5) totalSpectFit_sam5 = new TH1D("NullSam5","NullSam5",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFD_sam5) totalSpectFD_sam5 = new TH1D("NullSam5","NullSam5",nbinsAutoSample,binsAutoSample->GetArray());

  TH1D* totalSpectMC_sam6 = (TH1D*) fitResultFile->Get("evhist_sam6_iter0_mc");
  TH1D* totalSpectFit_sam6 = (TH1D*) fitResultFile->Get("evhist_sam6_finaliter_pred");
  TH1D* totalSpectFD_sam6 = (TH1D*) fitResultFile->Get("evhist_sam6_iter0_data");
  if(totalSpectMC_sam6) totalSpectMC_allSam->Add(totalSpectMC_sam6);
  if(totalSpectFit_sam6)totalSpectFit_allSam->Add(totalSpectFit_sam6);
  if(totalSpectFD_sam6) totalSpectFD_allSam->Add(totalSpectFD_sam6);
  if(!totalSpectMC_sam6) totalSpectMC_sam6 = new TH1D("NullSam6","NullSam6",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFit_sam6) totalSpectFit_sam6 = new TH1D("NullSam6","NullSam6",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFD_sam6) totalSpectFD_sam6 = new TH1D("NullSam6","NullSam6",nbinsAutoSample,binsAutoSample->GetArray());

  TH1D* totalSpectMC_sam7 = (TH1D*) fitResultFile->Get("evhist_sam7_iter0_mc");
  TH1D* totalSpectFit_sam7 = (TH1D*) fitResultFile->Get("evhist_sam7_finaliter_pred");
  TH1D* totalSpectFD_sam7 = (TH1D*) fitResultFile->Get("evhist_sam7_iter0_data");
  if(totalSpectMC_sam7) totalSpectMC_allSam->Add(totalSpectMC_sam7);
  if(totalSpectFit_sam7)totalSpectFit_allSam->Add(totalSpectFit_sam7);
  if(totalSpectFD_sam7) totalSpectFD_allSam->Add(totalSpectFD_sam7);
  if(!totalSpectMC_sam7) totalSpectMC_sam7 = new TH1D("NullSam7","NullSam7",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFit_sam7) totalSpectFit_sam7 = new TH1D("NullSam7","NullSam7",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFD_sam7) totalSpectFD_sam7 = new TH1D("NullSam7","NullSam7",nbinsAutoSample,binsAutoSample->GetArray());

  TH1D* totalSpectMC_sam8 = (TH1D*) fitResultFile->Get("evhist_sam8_iter0_mc");
  TH1D* totalSpectFit_sam8 = (TH1D*) fitResultFile->Get("evhist_sam8_finaliter_pred");
  TH1D* totalSpectFD_sam8 = (TH1D*) fitResultFile->Get("evhist_sam8_iter0_data");
  if(totalSpectMC_sam8) totalSpectMC_allSam->Add(totalSpectMC_sam8);
  if(totalSpectFit_sam8)totalSpectFit_allSam->Add(totalSpectFit_sam8);
  if(totalSpectFD_sam8) totalSpectFD_allSam->Add(totalSpectFD_sam8);
  if(!totalSpectMC_sam8) totalSpectMC_sam8 = new TH1D("NullSam8","NullSam8",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFit_sam8) totalSpectFit_sam8 = new TH1D("NullSam8","NullSam8",nbinsAutoSample,binsAutoSample->GetArray());
  if(!totalSpectFD_sam8) totalSpectFD_sam8 = new TH1D("NullSam8","NullSam8",nbinsAutoSample,binsAutoSample->GetArray());


  TH1D* mcSpect = new TH1D("mcSpect","mcSpect",nbinsAuto,binsAuto->GetArray());
  TH1D* fakeDataTrueSpect = new TH1D("fakeDataTrueSpect","fakeDataTrueSpect",nbinsAuto,binsAuto->GetArray());
  TH1D* mcSpect_T = new TH1D("mcSpect_T","mcSpect_T",nbinsAuto,binsAuto->GetArray());
  TH1D* fakeDataTrueSpect_T = new TH1D("fakeDataTrueSpect_T","fakeDataTrueSpect_T",nbinsAuto,binsAuto->GetArray());

  TH1D* mcSpect_ingrid = new TH1D("mcSpect","mcSpect",nbinsAuto,binsAuto->GetArray());
  TH1D* fakeDataTrueSpect_ingrid = new TH1D("fakeDataTrueSpect","fakeDataTrueSpect",nbinsAuto,binsAuto->GetArray());
  TH1D* mcSpect_ingrid_T = new TH1D("mcSpect_T","mcSpect_T",nbinsAuto,binsAuto->GetArray());
  TH1D* fakeDataTrueSpect_ingrid_T = new TH1D("fakeDataTrueSpect_T","fakeDataTrueSpect_T",nbinsAuto,binsAuto->GetArray());


  selEvtsTree->Draw("(D1True)>>mcSpect", "weight*(( mectopology<3 ) && D2True==0 && ((cutBranch!=0) && (cutBranch!=4)))");
  selEvtsFakeDataTree->Draw("(D1True)>>fakeDataTrueSpect", "weight*(( mectopology<3 ) && D2True==0 && ((cutBranch!=0) && (cutBranch!=4)))");

  selEvtsTree->Draw("(D1True)>>mcSpect_ingrid", "weight*(( mectopology<3 ) && D2True==1 && ((cutBranch!=0) && (cutBranch!=4)))");
  selEvtsFakeDataTree->Draw("(D1True)>>fakeDataTrueSpect_ingrid", "weight*(( mectopology<3 ) && D2True==1 && ((cutBranch!=0) && (cutBranch!=4)))");

  mcSpect->Scale(potRatio);


  TH1D* fitSpect = new TH1D("fitSpect","fitSpect",nbinsAuto,binsAuto->GetArray());
  TH1D* fitSpect_T = new TH1D("fitSpect_T","fitSpect_T",nbinsAuto,binsAuto->GetArray());

  std::cout << "Filling fitSpect" << std::endl;

  for(int i=1;i<=nbinsAuto;i++){
    fitSpect->SetBinContent( i,( (mcSpect->GetBinContent(i))*(fitParamValues->GetBinContent(i)) ) );
    fitSpect->SetBinError( i,( (fitSpect->GetBinContent(i))*(fitParamErrors->GetBinContent(i)) ) );
    cout << "Error for bin " << i << " is " << ( (fitSpect->GetBinContent(i))*(fitParamErrors->GetBinContent(i)) ) << endl;
    cout << " fitParamValues " << fitParamValues->GetBinContent(i)<< endl;
    cout << " fitParamErrors " << fitParamErrors->GetBinContent(i)<< endl;
    cout << " mcSpect " << mcSpect->GetBinContent(i)<< endl;
    cout << " fitSpect " << fitSpect->GetBinContent(i)<< endl;
  }
  std::cout << "Filled fitSpect" << std::endl;




  fitSpect->SetLineColor(kRed);
  fitSpect->SetLineWidth(2);
  fitSpect->SetLineStyle(2);
  mcSpect->SetLineColor(kBlue);
  mcSpect->SetLineWidth(2);
  mcSpect->SetLineStyle(2);
  fakeDataTrueSpect->SetLineColor(kGreen+2);
  fakeDataTrueSpect->SetLineWidth(2);
  fakeDataTrueSpect->SetLineStyle(2);

  fitSpect->Write();
  mcSpect->Write();
  fakeDataTrueSpect->Write();

  fitSpect_T->SetLineColor(kRed);
  fitSpect_T->SetLineWidth(2);
  fitSpect_T->SetLineStyle(2);
  mcSpect_T->SetLineColor(kBlue);
  mcSpect_T->SetLineWidth(2);
  mcSpect_T->SetLineStyle(2);
  fakeDataTrueSpect_T->SetLineColor(kGreen+2);
  fakeDataTrueSpect_T->SetLineWidth(2);
  fakeDataTrueSpect_T->SetLineStyle(2);

  fitSpect_T->Write();
  mcSpect_T->Write();
  fakeDataTrueSpect_T->Write();

  totalSpectFit_sam0->SetLineColor(kRed);
  totalSpectFit_sam0->SetLineWidth(2);
  totalSpectMC_sam0->SetLineColor(kBlue);
  totalSpectMC_sam0->SetLineWidth(2);
  totalSpectFD_sam0->SetLineColor(kGreen+2);
  totalSpectFD_sam0->SetLineWidth(2);

  totalSpectFit_sam1->SetLineColor(kRed);
  totalSpectFit_sam1->SetLineWidth(2);
  totalSpectMC_sam1->SetLineColor(kBlue);
  totalSpectMC_sam1->SetLineWidth(2);
  totalSpectFD_sam1->SetLineColor(kGreen+2);
  totalSpectFD_sam1->SetLineWidth(2);

  totalSpectFit_sam2->SetLineColor(kRed);
  totalSpectFit_sam2->SetLineWidth(2);
  totalSpectMC_sam2->SetLineColor(kBlue);
  totalSpectMC_sam2->SetLineWidth(2);
  totalSpectFD_sam2->SetLineColor(kGreen+2);
  totalSpectFD_sam2->SetLineWidth(2);

  totalSpectFit_sam3->SetLineColor(kRed);
  totalSpectFit_sam3->SetLineWidth(2);
  totalSpectMC_sam3->SetLineColor(kBlue);
  totalSpectMC_sam3->SetLineWidth(2);
  totalSpectFD_sam3->SetLineColor(kGreen+2);
  totalSpectFD_sam3->SetLineWidth(2);

  totalSpectFit_sam4->SetLineColor(kRed);
  totalSpectFit_sam4->SetLineWidth(2);
  totalSpectMC_sam4->SetLineColor(kBlue);
  totalSpectMC_sam4->SetLineWidth(2);
  totalSpectFD_sam4->SetLineColor(kGreen+2);
  totalSpectFD_sam4->SetLineWidth(2);

  totalSpectFit_sam5->SetLineColor(kRed);
  totalSpectFit_sam5->SetLineWidth(2);
  totalSpectMC_sam5->SetLineColor(kBlue);
  totalSpectMC_sam5->SetLineWidth(2);
  totalSpectFD_sam5->SetLineColor(kGreen+2);
  totalSpectFD_sam5->SetLineWidth(2);

  totalSpectFit_sam6->SetLineColor(kRed);
  totalSpectFit_sam6->SetLineWidth(2);
  totalSpectMC_sam6->SetLineColor(kBlue);
  totalSpectMC_sam6->SetLineWidth(2);
  totalSpectFD_sam6->SetLineColor(kGreen+2);
  totalSpectFD_sam6->SetLineWidth(2);


  totalSpectFit_sam8->SetLineColor(kRed);
  totalSpectFit_sam8->SetLineWidth(2);
  totalSpectMC_sam8->SetLineColor(kBlue);
  totalSpectMC_sam8->SetLineWidth(2);
  totalSpectFD_sam8->SetLineColor(kGreen+2);
  totalSpectFD_sam8->SetLineWidth(2);


  totalSpectFit_allSam->SetLineColor(kRed);
  totalSpectFit_allSam->SetLineWidth(2);
  totalSpectMC_allSam->SetLineColor(kBlue);
  totalSpectMC_allSam->SetLineWidth(2);
  totalSpectFD_allSam->SetLineColor(kGreen+2);
  totalSpectFD_allSam->SetLineWidth(2);

  totalSpectMC_sam0 ->Write();
  totalSpectFit_sam0->Write();  
  totalSpectFD_sam0 ->Write();
  totalSpectMC_sam1 ->Write();
  totalSpectFit_sam1->Write();
  totalSpectFD_sam1 ->Write();


  TCanvas* canv = new TCanvas("Signal Comp","Signal Comp");
  fitSpect->Draw();
  mcSpect->Draw("same");
  fakeDataTrueSpect->Draw("same");
  canv->Write();

  TCanvas* canv_T = new TCanvas("SigCompT","SigCompT");
  fitSpect_T->Draw();
  mcSpect_T->Draw("same");
  fakeDataTrueSpect_T->Draw("same");
  canv_T->Write();

  TCanvas* canv2 = new TCanvas("Sample0 Comp","Sample0 Comp");
  totalSpectFit_sam0->Draw();
  totalSpectMC_sam0->Draw("same");  
  totalSpectFD_sam0->Draw("sameE"); 
  canv2->Write();

  TCanvas* canv3 = new TCanvas("Sample1 Comp","Sample1 Comp");
  totalSpectFit_sam1->Draw();
  totalSpectMC_sam1->Draw("same");  
  totalSpectFD_sam1->Draw("sameE"); 
  canv3->Write();

  TCanvas* canv32= new TCanvas("Sample2 Comp","Sample2 Comp");
  totalSpectFit_sam2->Draw();
  totalSpectMC_sam2->Draw("same");  
  totalSpectFD_sam2->Draw("sameE"); 
  canv32->Write();
  TCanvas* canv33= new TCanvas("Sample3 Comp","Sample3 Comp");
  totalSpectFit_sam3->Draw();
  totalSpectMC_sam3->Draw("same");  
  totalSpectFD_sam3->Draw("sameE"); 
  canv33->Write();
  TCanvas* canv34= new TCanvas("Sample4 Comp","Sample4 Comp");
  totalSpectFit_sam4->Draw();
  totalSpectMC_sam4->Draw("same");  
  totalSpectFD_sam4->Draw("sameE"); 
  canv34->Write();
  TCanvas* canv35= new TCanvas("Sample5 Comp","Sample5 Comp");
  totalSpectFit_sam5->Draw();
  totalSpectMC_sam5->Draw("same");  
  totalSpectFD_sam5->Draw("sameE"); 
  canv35->Write();
  TCanvas* canv36= new TCanvas("Sample6 Comp","Sample6 Comp");
  totalSpectFit_sam6->Draw();
  totalSpectMC_sam6->Draw("same");  
  totalSpectFD_sam6->Draw("sameE"); 
  canv36->Write();


  TCanvas* canv4 = new TCanvas("All Sample Comp","All Sample Comp");
  totalSpectFit_allSam->Draw();
  totalSpectMC_allSam->Draw("same");  
  totalSpectFD_allSam->Draw("sameE"); 
  canv4->Write();

  TCanvas* canv5 = new TCanvas("All Truth and Rec","All Truth and Rec");
  fitSpect->Draw();
  mcSpect->Draw("same");
  fakeDataTrueSpect->Draw("same");
  totalSpectFit_allSam->Draw("same");
  totalSpectMC_allSam->Draw("same");  
  totalSpectFD_allSam->Draw("sameE"); 
  canv5->Write();


}
