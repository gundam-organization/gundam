/******************************************************

Code to take output of the fitter and produce a corrected
Nevents spectrum compared to the MC and fake data truth

Author: Stephen Dolan
Date Created: Jan 2016

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


  TH1D* mcSpect = new TH1D("mcSpect","mcSpect",nbinsAuto,binsAuto->GetArray());
  TH1D* fakeDataTrueSpect = new TH1D("fakeDataTrueSpect","fakeDataTrueSpect",nbinsAuto,binsAuto->GetArray());
  TH1D* mcSpect_T = new TH1D("mcSpect_T","mcSpect_T",nbinsAuto,binsAuto->GetArray());
  TH1D* fakeDataTrueSpect_T = new TH1D("fakeDataTrueSpect_T","fakeDataTrueSpect_T",nbinsAuto,binsAuto->GetArray());

  //Super DEBUG option:
  //selEvtsTree->Draw("D2True>>mcSpect"                  , "( ( (mectopology==1)||(mectopology==2) ) && (cutBranch==1) )");
  //selEvtsFakeDataTree->Draw("D2True>>fakeDataTrueSpect", "( ( (mectopology==1)||(mectopology==2) ) && (cutBranch==1) )");

  //selEvtsTree->Draw("D2True>>mcSpect"                  , "( (mectopology==1)||(mectopology==2) ) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch!=5) && (cutBranch!=6))");
  //selEvtsFakeDataTree->Draw("D2True>>fakeDataTrueSpect", "( (mectopology==1)||(mectopology==2) ) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch!=5) && (cutBranch!=6))");

  selEvtsTree->Draw("(D1True)>>mcSpect", "weight*((( (mectopology==1)||(mectopology==2) ) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) )) && ((cutBranch!=0) && (cutBranch!=4)))");// && (cutBranch!=5) && (cutBranch!=6))");
  selEvtsFakeDataTree->Draw("(D1True)>>fakeDataTrueSpect", "weight*((( (mectopology==1)||(mectopology==2) ) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) )) && ((cutBranch!=0) && (cutBranch!=4)))");// && (cutBranch!=5) && (cutBranch!=6))");
  mcSpect->Scale(potRatio);

  //The _T implies the result is post eff correction
  trueEvtsTree->Draw("(D1True)>>mcSpect_T", "(( (mectopology==1)||(mectopology==2) ) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) )) ");
  trueEvtsFakeDataTree->Draw("D1True>>fakeDataTrueSpect_T", "(( (mectopology==1)||(mectopology==2) ) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) ))");
  mcSpect_T->Scale(potRatio);


  TH1D* fitSpect = new TH1D("fitSpect","fitSpect",nbinsAuto,binsAuto->GetArray());
  TH1D* fitSpect_T = new TH1D("fitSpect_T","fitSpect_T",nbinsAuto,binsAuto->GetArray());

  std::cout << "Filling fitSpect" << std::endl;

  for(int i=1;i<=nbinsAuto;i++){
    fitSpect->SetBinContent( i,( (mcSpect->GetBinContent(i))*(fitParamValues->GetBinContent(i)) ) );
    fitSpect->SetBinError( i,( (fitSpect->GetBinContent(i))*(fitParamErrors->GetBinContent(i)) ) );
    cout << "Error for bin " << i << " is " << ( (fitSpect->GetBinContent(i))*(fitParamErrors->GetBinContent(i)) ) << endl;
    fitSpect_T->SetBinContent( i,( (mcSpect_T->GetBinContent(i))*(fitParamValues->GetBinContent(i)) ) );
    fitSpect_T->SetBinError( i,( (fitSpect_T->GetBinContent(i))*(fitParamErrors->GetBinContent(i)) ) );
  }
  std::cout << "Filled fitSpect" << std::endl;


//temp stat error fudge:
 //for(int i=1;i<=nbinsAuto;i++){
 //  double systRelErr = fitSpect->GetBinError(i)/fitSpect->GetBinContent(i);
 //  double statRelErr = 0.06;
 //  double totalRelErr = sqrt((systRelErr*systRelErr)+(statRelErr*statRelErr));
 //  fitSpect->SetBinError( i, totalRelErr*fitSpect->GetBinContent(i) );
 //  cout << "Total error for bin " << i << " is " << totalRelErr*fitSpect->GetBinContent(i) << endl;
 //  fitSpect_T->SetBinError( i, totalRelErr*fitSpect_T->GetBinContent(i) );
 //}

  TH1D* dif_xSecFit = new TH1D("dif_xSecFit","dif_xSecFit",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_xSecFD = new TH1D("dif_xSecFitFD","dif_xSecFitFD",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_xSecMC = new TH1D("dif_xSecFitMC","dif_xSecFitMC",nbinsAuto,binsAuto->GetArray());

  double nneutrons = 2.75e29;
  double flux = 1.92e13; //cm^2 per 10^21 POT
  double pot = 3.2; //10^21 
  for(int i=1;i<=nbinsAuto;i++){
    dif_xSecFit->SetBinContent(i, fitSpect_T->GetBinContent(i)/(nneutrons*flux*pot*dif_xSecFit->GetBinWidth(i)));
    dif_xSecFit->SetBinError(i, dif_xSecFit->GetBinContent(i)*(fitSpect_T->GetBinError(i)/fitSpect_T->GetBinContent(i)) );

    dif_xSecMC->SetBinContent(i, mcSpect_T->GetBinContent(i)/(nneutrons*flux*pot*dif_xSecMC->GetBinWidth(i)));
    //dif_xSecMC->SetBinError(i, dif_xSecMC->GetBinContent(i)*(mcSpect_T->GetBinError(i)/mcSpect_T->GetBinContent(i)) );
    dif_xSecMC->SetBinError(i, 0);

    dif_xSecFD->SetBinContent(i, fakeDataTrueSpect_T->GetBinContent(i)/(nneutrons*flux*pot*dif_xSecFD->GetBinWidth(i)));
    //dif_xSecFD->SetBinError(i, dif_xSecFD->GetBinContent(i)*(fakeDataTrueSpect_T->GetBinError(i)/fakeDataTrueSpect_T->GetBinContent(i)) );
    dif_xSecFD->SetBinError(i, 0);
  }

  //Errors from stats pulls study for proceedings:
  // dif_xSecFit->SetBinError(1, dif_xSecFit->GetBinContent(1)*(0.11));
  // dif_xSecFit->SetBinError(2, dif_xSecFit->GetBinContent(2)*(0.13));
  // dif_xSecFit->SetBinError(3, dif_xSecFit->GetBinContent(3)*(0.157));
  // dif_xSecFit->SetBinError(4, dif_xSecFit->GetBinContent(4)*(0.147));
  // dif_xSecFit->SetBinError(5, dif_xSecFit->GetBinContent(5)*(0.144));
  // dif_xSecFit->SetBinError(6, dif_xSecFit->GetBinContent(6)*(0.26));
  // dif_xSecFit->SetBinError(7, dif_xSecFit->GetBinContent(7)*(0.29));
  // dif_xSecFit->SetBinError(8, dif_xSecFit->GetBinContent(8)*(0.35));

  //Stat errors for Neutrino poster:
  // dif_xSecFit->SetBinError(1, sqrt( (dif_xSecFit->GetBinContent(1)*(0.11*0.825)*dif_xSecFit->GetBinContent(1)*(0.11*0.825)) +   (dif_xSecFit->GetBinError(1) * dif_xSecFit->GetBinError(1)) ) );
  // dif_xSecFit->SetBinError(2, sqrt( (dif_xSecFit->GetBinContent(2)*(0.13*0.825)*dif_xSecFit->GetBinContent(2)*(0.13*0.825)) +   (dif_xSecFit->GetBinError(2) * dif_xSecFit->GetBinError(2)) ) );
  // dif_xSecFit->SetBinError(3, sqrt( (dif_xSecFit->GetBinContent(3)*(0.157*0.825)*dif_xSecFit->GetBinContent(3)*(0.157*0.825)) + (dif_xSecFit->GetBinError(3) * dif_xSecFit->GetBinError(3)) ) );
  // dif_xSecFit->SetBinError(4, sqrt( (dif_xSecFit->GetBinContent(4)*(0.147*0.825)*dif_xSecFit->GetBinContent(4)*(0.147*0.825)) + (dif_xSecFit->GetBinError(4) * dif_xSecFit->GetBinError(4)) ) );
  // dif_xSecFit->SetBinError(5, sqrt( (dif_xSecFit->GetBinContent(5)*(0.144*0.825)*dif_xSecFit->GetBinContent(5)*(0.144*0.825)) + (dif_xSecFit->GetBinError(5) * dif_xSecFit->GetBinError(5)) ) );
  // dif_xSecFit->SetBinError(6, sqrt( (dif_xSecFit->GetBinContent(6)*(0.26*0.825)*dif_xSecFit->GetBinContent(6)*(0.26*0.825)) +   (dif_xSecFit->GetBinError(6) * dif_xSecFit->GetBinError(6)) ) );
  // dif_xSecFit->SetBinError(7, sqrt( (dif_xSecFit->GetBinContent(7)*(0.29*0.825)*dif_xSecFit->GetBinContent(7)*(0.29*0.825)) +   (dif_xSecFit->GetBinError(7) * dif_xSecFit->GetBinError(7)) ) );
  // dif_xSecFit->SetBinError(8, sqrt( (dif_xSecFit->GetBinContent(8)*(0.35*0.825)*dif_xSecFit->GetBinContent(8)*(0.35*0.825)) +   (dif_xSecFit->GetBinError(8) * dif_xSecFit->GetBinError(8)) ) );



  dif_xSecFit->SetLineColor(kRed);
  dif_xSecFit->SetLineWidth(2);

  dif_xSecMC->SetLineColor(kBlue);
  dif_xSecMC->SetLineWidth(2);

  dif_xSecFD->SetLineColor(kGreen+2);
  dif_xSecFD->SetLineWidth(2);


  dif_xSecFit->Write();
  dif_xSecMC->Write();
  dif_xSecFD->Write();


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

  cout << "pre eff true chi2 hist" << endl;
  fitSpect->Chi2Test(fakeDataTrueSpect, "UUP");


  TCanvas* canv5_T = new TCanvas("AllTruthandRec_T","AllTruthandRec_T");
  fitSpect_T->Draw();
  mcSpect_T->Draw("same");
  fakeDataTrueSpect_T->Draw("same");
  totalSpectFit_allSam->Draw("same");
  totalSpectMC_allSam->Draw("same");  
  totalSpectFD_allSam->Draw("sameE"); 
  canv5_T->Write();

  TCanvas* canvxs = new TCanvas("xsecCanv","xsecCanv");
  dif_xSecMC->Draw();
  dif_xSecFD->Draw("same");
  dif_xSecFit->Draw("same");
  canvxs->Write();


  cout << "xsec chi2 hist with current errors" << endl;
  dif_xSecFit->Chi2Test(dif_xSecFD, "WWP");

  for(int i=1;i<=nbinsAuto;i++){
    dif_xSecFit->SetBinError(i, sqrt(dif_xSecFit->GetBinError(i)*dif_xSecFit->GetBinError(i) + (0.2*dif_xSecFit->GetBinContent(i))*(0.2*dif_xSecFit->GetBinContent(i))) );
  }

  cout << "xsec chi2 hist with exp errors" << endl;
  dif_xSecFit->Chi2Test(dif_xSecFD, "WWP");

}
