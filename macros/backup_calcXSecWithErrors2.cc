/******************************************************

Code to take output of the fitter and produce a corrected
Nevents spectrum compared to the MC and fake data truth

Now takes result from propError code to include full errors
on the fit result. 

Author: Stephen Dolan
Date Created: Jun 2016

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

#include <TMatrixD.h>
#include <TMatrixDSym.h>

//#define OVERIDEBINNING


using namespace std;

//bool isRealData = false;
//bool TPCOnlySig= true;

double calcChi2(TH1D* h1, TH1D* h2, TMatrixD covar){
  double chi2=0;
  cout << "calcChi2::Determinant of covar is: " << covar.Determinant() << endl;
  //covar.Print();
  covar.SetTol(1e-1000);
  covar.Invert();
  //covar.Print();
  //cout << "Inverted covariance matrix" << endl;
  for(int i=0; i<h1->GetNbinsX(); i++){
    for(int j=0; j<h1->GetNbinsX(); j++){
      chi2+= ((h1->GetBinContent(i+1)) - (h2->GetBinContent(i+1)))*covar[i][j]*((h1->GetBinContent(j+1)) - (h2->GetBinContent(j+1)));
    }
  }
  cout << "calcChi2::chi2 is: " << chi2 << endl;
  return chi2;
}

TH1D* makeShapeOnly(TH1D* h1, TH1D* histToFill, int nbinsAuto){
  for(int i=1;i<=nbinsAuto;i++){
    histToFill->SetBinContent(i, h1->GetBinContent(i)/h1->Integral());
    histToFill->SetBinError(i, histToFill->GetBinContent(i)*(h1->GetBinError(i)/h1->GetBinContent(i)) );
  }
  return histToFill;
}

TH1D* makeShapeOnlyNoError(TH1D* h1, TH1D* histToFill, int nbinsAuto){
  for(int i=1;i<=nbinsAuto;i++){
    histToFill->SetBinContent(i, h1->GetBinContent(i)/h1->Integral());
  }
  return histToFill;
}



//pAngleRestrict - how restrict proton angle?
// 0 Normal phase space restrictions
// 1 Low angle compononent (cosTheta: 0.8-1)
// 2 High angle compononent (cosTheta: 0.4-0.8)


//Example running:
//calcXsecWithErrors2("propErrorOut_reg0p8.root", "$XSLLHFITTER/inputs/NeutAir5_2DV2.root", "../lcurveStudyOut_reg0.8.root", "$XSLLHFITTER/inputs/GenieAir2_2DV2.root", "xsecOut_reg0p8.root", 1.12385)
void calcXsecWithErrors2 (TString propErrFilename, TString mcFilename, TString fitResultFilename, TString fakeDataFilename, TString outputFileName, double potRatio, double MCPOT=3.316, bool isRealData=false, bool TPCOnlySig=false, int pAngleRestrict=0){
  TFile* propErrFile = new TFile(propErrFilename);
  TFile* mcFile = new TFile(mcFilename);
  TFile* fitResultFile = new TFile(fitResultFilename);
  TFile* fakeDataFile = new TFile(fakeDataFilename); 
  TFile* outputFile = new TFile(outputFileName, "RECREATE"); 
 

  TTree* selEvtsTree = (TTree*) mcFile->Get("selectedEvents");
  TTree* trueEvtsTree = (TTree*) mcFile->Get("trueEvents");
  TTree* selEvtsFakeDataTree = (TTree*) fakeDataFile->Get("selectedEvents");
  TTree* trueEvtsFakeDataTree = (TTree*) fakeDataFile->Get("trueEvents");

  TH1D* histForBinning = (TH1D*) fitResultFile->Get("MuTPCpTPC_RecD1_cc0pi1p_nominal");
 
  TH1D* histForAnyBinning = (TH1D*) fitResultFile->Get("MuTPCpTPC_RecD2_cc0pi1p_nominal");


  #ifdef OVERIDEBINNING
    const int nbinsAuto = 8;
    const int nbinsAutoAB = 8;
    const Double_t binsAutoAr[nbinsAuto+1] = {0.00001, 0.47, 1.02, 1.54, 1.98, 2.34, 2.64, 2.89, 3.14159};
    const TArrayD* binsAuto = new TArrayD(nbinsAuto+1, binsAutoAr);
    const Double_t binsAutoABAr[nbinsAuto+1] = {0.00001, 0.47, 1.02, 1.54, 1.98, 2.34, 2.64, 2.89, 3.14159};
    const TArrayD* binsAutoAB = new TArrayD(nbinsAuto+1, binsAutoAr);
  #else
    const int nbinsAuto = histForBinning->GetXaxis()->GetNbins();
    const TArrayD* binsAuto = histForBinning->GetXaxis()->GetXbins();  
    const int nbinsAutoAB = histForAnyBinning->GetXaxis()->GetNbins();
    const TArrayD* binsAutoAB = histForAnyBinning->GetXaxis()->GetXbins();  
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

  TH1D* mcSpectAB = new TH1D("mcSpectAB","mcSpectAB",nbinsAutoAB,binsAutoAB->GetArray());
  TH1D* fakeDataTrueSpectAB = new TH1D("fakeDataTrueSpectAB","fakeDataTrueSpectAB",nbinsAutoAB,binsAutoAB->GetArray());
  TH1D* fitSpectAB = new TH1D("fitSpectAB","fitSpectAB",nbinsAutoAB,binsAutoAB->GetArray());

  TH1D* mcSpectOOPS_T = new TH1D("mcSpectOOPS_T","mcSpectOOPS_T",1 ,0 ,10000);
  TH1D* fakeDataTrueSpectOOPS_T = new TH1D("fakeDataTrueSpectOOPS_T","fakeDataTrueSpectOOPS_T",1 ,0 ,10000);
  TH1D* fitSpectOOPS_T = new TH1D("fitSpectOOPS_T","fitSpectOOPS_T",1 ,0 ,10000);

  TH1D* mcSpect_sigInOOPS = new TH1D("mcSpect_sigInOOPS","mcSpect_sigInOOPS",nbinsAuto,binsAuto->GetArray());
  TH1D* fakeDataTrueSpect_sigInOOPS = new TH1D("fakeDataTrueSpect_sigInOOPS","fakeDataTrueSpect_sigInOOPS",nbinsAuto,binsAuto->GetArray());
  TH1D* mcSpect_OOPSInSig = new TH1D("mcSpect_OOPSInSig","mcSpect_OOPSInSig",nbinsAuto,binsAuto->GetArray());
  TH1D* fakeDataTrueSpect_OOPSInSig = new TH1D("fakeDataTrueSpect_OOPSInSig","fakeDataTrueSpect_OOPSInSig",nbinsAuto,binsAuto->GetArray());

  //selEvtsTree->Draw("(D1True)>>mcSpect", "weight*((( (mectopology==1)||(mectopology==2) ) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) )) && ((cutBranch!=0) && (cutBranch!=4)))");// && (cutBranch!=5) && (cutBranch!=6))");
  //selEvtsFakeDataTree->Draw("(D1True)>>fakeDataTrueSpect", "weight*((( (mectopology==1)||(mectopology==2) ) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) )) && ((cutBranch!=0) && (cutBranch!=4)))");// && (cutBranch!=5) && (cutBranch!=6))");
  selEvtsTree->Draw("(D1True)>>mcSpect", "weight*(( (mectopology==1)||(mectopology==2) ) && (D2True<0.1) && (D2True>-0.1) && (D1True>0.000001) && (D1Rec>0.000001) && (D1Rec<100.0) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch<8)))");// && (cutBranch!=5) && (cutBranch!=6))");
  selEvtsFakeDataTree->Draw("(D1True)>>fakeDataTrueSpect", "weight*(( (mectopology==1)||(mectopology==2) ) && (D2True<0.1)  && (D2True>-0.1) && (D1True>0.000001) && (D1Rec>0.000001) && (D1Rec<100.0) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch<8) ))");// && (cutBranch!=5) && (cutBranch!=6))");
  mcSpect->Scale(potRatio);

  selEvtsTree->Draw("(D1True)>>mcSpect_sigInOOPS", "weight*(( (mectopology==1)||(mectopology==2) ) && (D2Rec!=0) && (D2True==0) && (D1True>0.000001) && (D1Rec>0.000001) && (D1Rec<100.0) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch<8)))");// && (cutBranch!=5) && (cutBranch!=6))");
  selEvtsFakeDataTree->Draw("(D1True)>>fakeDataTrueSpect_sigInOOPS", "weight*(( (mectopology==1)||(mectopology==2) ) && (D2Rec!=0) && (D2True==0) && (D1True>0.000001) && (D1Rec>0.000001) && (D1Rec<100.0) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch<8) ))");// && (cutBranch!=5) && (cutBranch!=6))");
  mcSpect_sigInOOPS->Scale(potRatio);

  selEvtsTree->Draw("(D1True)>>mcSpect_OOPSInSig", "weight*(( (mectopology==1)||(mectopology==2) ) && (D2Rec==0) && (D2True!=0) && (D1True>0.000001) && (D1Rec>0.000001) && (D1Rec<100.0) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch<8)))");// && (cutBranch!=5) && (cutBranch!=6))");
  selEvtsFakeDataTree->Draw("(D1True)>>fakeDataTrueSpect_OOPSInSig", "weight*(( (mectopology==1)||(mectopology==2) ) && (D2Rec==0) && (D2True!=0) && (D1True>0.000001) && (D1Rec>0.000001) && (D1Rec<100.0) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch<8) ))");// && (cutBranch!=5) && (cutBranch!=6))");
  mcSpect_OOPSInSig->Scale(potRatio);

  selEvtsTree->Draw("(D2True)>>mcSpectAB", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && (D1True<100.0) && (D1Rec>0.000001) && (D1Rec<100.0) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch<8)))");// && (cutBranch!=5) && (cutBranch!=6))");
  selEvtsFakeDataTree->Draw("(D2True)>>fakeDataTrueSpectAB", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && (D1True<100.0) && (D1Rec>0.000001) && (D1Rec<100.0) && ((cutBranch!=0) && (cutBranch!=4) && (cutBranch<8) ))");// && (cutBranch!=5) && (cutBranch!=6))");
  mcSpectAB->Scale(potRatio);

  if(TPCOnlySig==false && pAngleRestrict==0){
    //The _T implies the result is post eff correction
    trueEvtsTree->Draw("(D1True)>>mcSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) )) ");
    trueEvtsFakeDataTree->Draw("D1True>>fakeDataTrueSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) ))");
    //trueEvtsTree->Draw("(D1True)>>mcSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D2True<0.1) && (D2True>-0.1))");
    //trueEvtsFakeDataTree->Draw("(D1True)>>fakeDataTrueSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D2True<0.1) && (D2True>-0.1))");
   
    trueEvtsTree->Draw("(D1True)>>mcSpectOOPS_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue<450)||(pMomTrue>1000)||(muMomTrue<250)||(muCosThetaTrue<-0.6)||(pCosThetaTrue<0.4) )) ");
    trueEvtsFakeDataTree->Draw("D1True>>fakeDataTrueSpectOOPS_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue<450)||(pMomTrue>1000)||(muMomTrue<250)||(muCosThetaTrue<-0.6)||(pCosThetaTrue<0.4) )) ");

  }
  else if (TPCOnlySig==true && pAngleRestrict==0){
    //The _T implies the result is post eff correction
    trueEvtsTree->Draw("(D1True)>>mcSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue>500)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>0.6)&&(pCosThetaTrue>0.6) )) ");
    trueEvtsFakeDataTree->Draw("D1True>>fakeDataTrueSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue>500)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>0.6)&&(pCosThetaTrue>0.6) ))");
  }
  else if (TPCOnlySig==false && pAngleRestrict==1){
    //The _T implies the result is post eff correction
    trueEvtsTree->Draw("(D1True)>>mcSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.8) )) ");
    trueEvtsFakeDataTree->Draw("D1True>>fakeDataTrueSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.8) ))");
  }
  else if (TPCOnlySig==false && pAngleRestrict==2){
    //The _T implies the result is post eff correction
    trueEvtsTree->Draw("(D1True)>>mcSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4)&&(pCosThetaTrue<0.8) )) ");
    trueEvtsFakeDataTree->Draw("D1True>>fakeDataTrueSpect_T", "weight*(( (mectopology==1)||(mectopology==2) ) && (D1True>0.000001) && ( (pMomTrue>450)&&(pMomTrue<1000)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4)&&(pCosThetaTrue<0.8) ))");
  }

  mcSpect_T->Scale(potRatio);


  TH1D* propErrHisto = (TH1D*) propErrFile->Get("SigHistoFinal");
  TMatrixDSym *covarNevtsInp = (TMatrixDSym*) propErrFile->Get("covar_xsec");
  TMatrixDSym *covarNevts = (TMatrixDSym*) propErrFile->Get("covar_xsec");

  TH1D* effHisto = new TH1D("effHisto","effHisto",nbinsAuto,binsAuto->GetArray());
  TH1D* effHisto_fd = new TH1D("effHisto_fd","effHisto_fd",nbinsAuto,binsAuto->GetArray());

  TH1D* effHistoOOPS = new TH1D("effHistoOOPS","effHistoOOPS",1 , 0, 10000 );
  TH1D* effHistoOOPS_fd = new TH1D("effHistoOOPS_fd","effHistoOOPS_fd",1 , 0, 10000 );

  TH1D* fitSpect = new TH1D("fitSpect","fitSpect",nbinsAuto,binsAuto->GetArray());
  TH1D* fitSpect_T = new TH1D("fitSpect_T","fitSpect_T",nbinsAuto,binsAuto->GetArray());


  std::cout << "Filling fitSpect" << std::endl;

  for(int i=1;i<=nbinsAuto;i++){
    fitSpect->SetBinContent( i, propErrHisto->GetBinContent(i) );
    fitSpect->SetBinError( i, propErrHisto->GetBinError(i) );
    double eff = mcSpect->GetBinContent(i)/mcSpect_T->GetBinContent(i);
    effHisto->SetBinContent(i, eff);
    double eff_fd = fakeDataTrueSpect->GetBinContent(i)/fakeDataTrueSpect_T->GetBinContent(i);
    effHisto_fd->SetBinContent(i, eff_fd);
    fitSpect_T->SetBinContent(i, fitSpect->GetBinContent(i)/eff);
    fitSpect_T->SetBinError(i, fitSpect->GetBinError(i)/eff);
  }
  for(int i=1;i<=nbinsAutoAB;i++){
    fitSpectAB->SetBinContent( i, propErrHisto->GetBinContent(i+nbinsAuto) );
    fitSpectAB->SetBinError( i, propErrHisto->GetBinError(i+nbinsAuto) );
  }
  std::cout << "Filled fitSpect" << std::endl;

  double effOOPS = mcSpectAB->GetBinContent(2)/mcSpectOOPS_T->GetBinContent(1);
  double effOOPS_fd = fakeDataTrueSpectAB->GetBinContent(2)/fakeDataTrueSpectOOPS_T->GetBinContent(1);
  effHistoOOPS->SetBinContent(1, effOOPS);
  effHistoOOPS_fd->SetBinContent(1, effOOPS_fd);

  fitSpectOOPS_T->SetBinContent(1, fitSpectAB->GetBinContent(2)/effOOPS);
  fitSpectAB->SetBinError( 1, fitSpectAB->GetBinError(2)/effOOPS);

//temp stat error fudge:
 //for(int i=1;i<=nbinsAuto;i++){
 //  double systRelErr = fitSpect->GetBinError(i)/fitSpect->GetBinContent(i);
 //  double statRelErr = 0.06;
 //  double totalRelErr = sqrt((systRelErr*systRelErr)+(statRelErr*statRelErr));
 //  fitSpect->SetBinError( i, totalRelErr*fitSpect->GetBinContent(i) );
 //  cout << "Total error for bin " << i << " is " << totalRelErr*fitSpect->GetBinContent(i) << endl;
 //  fitSpect_T->SetBinError( i, totalRelErr*fitSpect_T->GetBinContent(i) );
 //}

  TH1D* dif_xSecFit = new TH1D("Result","Result",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_xSecFD = new TH1D("Fake Data","Fake Data",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_xSecMC = new TH1D("Monte Carlo","Monte Carlo",nbinsAuto,binsAuto->GetArray());

  TH1D* dif_xSecNeut = new TH1D("NEUT Prior","NEUT Prior",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_xSecGenie = new TH1D("GENIE","GENIE",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_xSecNiwgNeut = new TH1D("NEUT Nominal","NEUT Nominal",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_xSecNuwro = new TH1D("NuWro (prod)","NuWro (prod)",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_xSecNuwroFix = new TH1D("NuWro (scale fix)","NuWro (scale fix)",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_xSecNeutNo2p2h = new TH1D("NEUT No 2p2h","NEUT No 2p2h",nbinsAuto,binsAuto->GetArray());

  TH1D* dif_shapeOnly_xSecFit = new TH1D("Result Shape Only","Result Shape Only",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_shapeOnly_xSecFD = new TH1D("Fake Data Shape Only","Fake Data Shape Only",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_shapeOnly_xSecMC = new TH1D("Monte Carlo Shape Only","Monte Carlo Shape Only",nbinsAuto,binsAuto->GetArray());

  TH1D* dif_shapeOnly_xSecNeut = new TH1D("NEUT Prior Shape Only","NEUT Prior Shape Only",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_shapeOnly_xSecGenie = new TH1D("GENIE Shape Only","GENIE Shape Only",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_shapeOnly_xSecNiwgNeut = new TH1D("NEUT Nominal Shape Only","NEUT Nominal Shape Only",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_shapeOnly_xSecNuwro = new TH1D("NuWro (prod) Shape Only","NuWro (prod) Shape Only",nbinsAuto,binsAuto->GetArray());
  TH1D* dif_shapeOnly_xSecNeutNo2p2h = new TH1D("NEUT No 2p2h Shape Only","NEUT No 2p2h Shape Only",nbinsAuto,binsAuto->GetArray());  

  //Get new integrated flux
  //Flux Integral is:    19.273
  //Flux binning is:     0.0-0.4, 0.4-0.5, 0.5-0.6, 0.6-0.7, 0.7-1.0, 1.0-1.5, 1.5-2.5, 2.5-3.5, 3.5-5.0, 5.0-7.0, 7.0-30
  //Integral over bin:   4.178  , 4.379  , 3.385  , 4.758  , 5.890  , 1.363  , 0.783  , 0.358  , 0.324  , 0.165  , 0.069
  //Sum:                 25.652                
  // Note the itegrals don't add up because I allowed each bin to overlap with it's adjacent bin by one flux histo bin
  //Get Flux Params:
  double fluxBinWeight[11] = {4.178/25.652  , 4.379/25.652  , 3.385/25.652  , 4.758/25.652  , 5.890/25.652  , 1.363/25.652  , 0.783/25.652  , 0.358/25.652  , 0.324/25.652  , 0.165/25.652  , 0.069/25.652}; 
  TH1D* fittedFluxPar = (TH1D*) fitResultFile->Get("paramhist_parpar_flux_result");
  double intFluxWeight = 0;
  for(int b=0; b<11; b++){
    intFluxWeight+=fluxBinWeight[b]*fittedFluxPar->GetBinContent(b+1);
  }
  cout << "Integrated flux weight following fitted flux is: " << intFluxWeight << endl;
  TH1D* intFluxWeightHisto = new TH1D("intFluxWeightHisto","intFluxWeightHisto",100,0,2);
  intFluxWeightHisto->Fill(intFluxWeight);


  double nneutrons = 2.75e29;
  double nnucleons = 5.5373e29;
  double tgtScale = nneutrons/nnucleons;
  double flux = 1.927347e13 * intFluxWeight; //cm^2 per 10^21 POT
  double potMC = MCPOT; //10^21 
  double potData = potMC*potRatio; 
  for(int i=1;i<=nbinsAuto;i++){
    dif_xSecFit->SetBinContent(i, fitSpect_T->GetBinContent(i)/(nnucleons*flux*potData*dif_xSecFit->GetBinWidth(i)));
    dif_xSecFit->SetBinError(i, dif_xSecFit->GetBinContent(i)*(fitSpect_T->GetBinError(i)/fitSpect_T->GetBinContent(i)) );

    dif_xSecMC->SetBinContent(i, mcSpect_T->GetBinContent(i)/(nnucleons*flux*potData*dif_xSecMC->GetBinWidth(i)));
    //dif_xSecMC->SetBinError(i, dif_xSecMC->GetBinContent(i)*(mcSpect_T->GetBinError(i)/mcSpect_T->GetBinContent(i)) );
    dif_xSecMC->SetBinError(i, 0);

    dif_xSecFD->SetBinContent(i, fakeDataTrueSpect_T->GetBinContent(i)/(nnucleons*flux*potData*dif_xSecFD->GetBinWidth(i)));
    //dif_xSecFD->SetBinError(i, dif_xSecFD->GetBinContent(i)*(fakeDataTrueSpect_T->GetBinError(i)/fakeDataTrueSpect_T->GetBinContent(i)) );
    dif_xSecFD->SetBinError(i, 0);
  }

  //dpt MC histos:

  //NEUT (Production):
  dif_xSecNeut->SetBinContent(1,tgtScale*5.08254e-39 );
  dif_xSecNeut->SetBinContent(2,tgtScale*1.04649e-38 );
  dif_xSecNeut->SetBinContent(3,tgtScale*9.89072e-39 );
  dif_xSecNeut->SetBinContent(4,tgtScale*7.53286e-39 );
  dif_xSecNeut->SetBinContent(5,tgtScale*4.19225e-39 );
  dif_xSecNeut->SetBinContent(6,tgtScale*1.82197e-39 );
  dif_xSecNeut->SetBinContent(7,tgtScale*1.11108e-39 );
  dif_xSecNeut->SetBinContent(8,tgtScale*2.33204e-40 );

  //GENIE:
  dif_xSecGenie->SetBinContent(1,tgtScale*8.59418e-39 );
  dif_xSecGenie->SetBinContent(2,tgtScale*9.06280e-39 );
  dif_xSecGenie->SetBinContent(3,tgtScale*8.95103e-39 );
  dif_xSecGenie->SetBinContent(4,tgtScale*7.22271e-39 );
  dif_xSecGenie->SetBinContent(5,tgtScale*2.50477e-39 );
  dif_xSecGenie->SetBinContent(6,tgtScale*1.12144e-39 );
  dif_xSecGenie->SetBinContent(7,tgtScale*4.84653e-40 );
  dif_xSecGenie->SetBinContent(8,tgtScale*9.11695e-41 );

  //NEUT NIWG RW:
  dif_xSecNiwgNeut->SetBinContent(1,tgtScale*0.484025E-38 );
  dif_xSecNiwgNeut->SetBinContent(2,tgtScale*0.993608E-38 );
  dif_xSecNiwgNeut->SetBinContent(3,tgtScale*0.935381E-38 );
  dif_xSecNiwgNeut->SetBinContent(4,tgtScale*0.708487E-38 );
  dif_xSecNiwgNeut->SetBinContent(5,tgtScale*0.381605E-38 );
  dif_xSecNiwgNeut->SetBinContent(6,tgtScale*0.151857E-38 );
  dif_xSecNiwgNeut->SetBinContent(7,tgtScale*0.083522E-38 );
  dif_xSecNiwgNeut->SetBinContent(8,tgtScale*0.015037E-38 );

  //NuWro (production):
  dif_xSecNuwro->SetBinContent(1,tgtScale*1.09702E-38 );
  dif_xSecNuwro->SetBinContent(2,tgtScale*1.60394E-38 );
  dif_xSecNuwro->SetBinContent(3,tgtScale*1.47253E-38 );
  dif_xSecNuwro->SetBinContent(4,tgtScale*1.10474E-38 );
  dif_xSecNuwro->SetBinContent(5,tgtScale*0.523762E-38 );
  dif_xSecNuwro->SetBinContent(6,tgtScale*0.131239E-38 );
  dif_xSecNuwro->SetBinContent(7,tgtScale*0.09246E-38 );
  dif_xSecNuwro->SetBinContent(8,tgtScale*0.0223159E-38 );

  //NuWro (xsec fix):
  dif_xSecNuwroFix->SetBinContent(1,tgtScale*0.575*1.09702E-38 );
  dif_xSecNuwroFix->SetBinContent(2,tgtScale*0.575*1.60394E-38 );
  dif_xSecNuwroFix->SetBinContent(3,tgtScale*0.575*1.47253E-38 );
  dif_xSecNuwroFix->SetBinContent(4,tgtScale*0.575*1.10474E-38 );
  dif_xSecNuwroFix->SetBinContent(5,tgtScale*0.575*0.523762E-38 );
  dif_xSecNuwroFix->SetBinContent(6,tgtScale*0.575*0.131239E-38 );
  dif_xSecNuwroFix->SetBinContent(7,tgtScale*0.575*0.09246E-38 );
  dif_xSecNuwroFix->SetBinContent(8,tgtScale*0.575*0.0223159E-38 );

  //NEUT No 2p2h
  dif_xSecNeutNo2p2h->SetBinContent(1 ,2.52105e-39);
  dif_xSecNeutNo2p2h->SetBinContent(2 ,5.20801e-39);
  dif_xSecNeutNo2p2h->SetBinContent(3 ,4.99469e-39);
  dif_xSecNeutNo2p2h->SetBinContent(4 ,3.67613e-39);
  dif_xSecNeutNo2p2h->SetBinContent(5 ,1.91540e-39);
  dif_xSecNeutNo2p2h->SetBinContent(6 ,7.54291e-40);
  dif_xSecNeutNo2p2h->SetBinContent(7 ,3.74448e-40);
  dif_xSecNeutNo2p2h->SetBinContent(8 ,5.98077e-41);



  //dat:

  TFile* datGenieXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/genie/dat_MCTruth.root");
  TH1D*  datGenieXSecHist = (TH1D*) datGenieXSecFile->Get("truth");
  TFile* datNEUTRWXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/neut/RW/dat_MCTruth.root");
  TH1D*  datNEUTRWXSecHist = (TH1D*) datNEUTRWXSecFile->Get("truth");
  TFile* datNuWroXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/nuwro/dat_MCTruth.root");
  TH1D*  datNuWroXSecHist = (TH1D*) datNuWroXSecFile->Get("truth");
  TH1D*  datNuWroXSecHist_ScaleFix = (TH1D*) datNuWroXSecFile->Get("truth");
  datNuWroXSecHist_ScaleFix->Scale(0.575);

  TFile* datNEUTPriorXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/neut_prior/dat_MCTruth.root");
  TH1D* datNEUTPriorXSecHist = (TH1D*) datNEUTPriorXSecFile->Get("dif_xSecFitFD");
  TFile* datNEUTNo2p2hXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/neut_no2p2h/dat_MCTruth.root");
  TH1D* datNEUTNo2p2hXSecHist = (TH1D*) datNEUTNo2p2hXSecFile->Get("dif_xSecFitFD");

  datGenieXSecHist->SetNameTitle("GENIE", "GENIE");
  datNEUTRWXSecHist->SetNameTitle("NEUT Nominal", "NEUT Nominal");
  datNuWroXSecHist->SetNameTitle("NuWro (prod)", "NuWro (prod)");
  datNuWroXSecHist_ScaleFix->SetNameTitle("NuWro (Scale Fix)", "NuWro (Scale Fix)");
  datNEUTPriorXSecHist->SetNameTitle("NEUT Prior", "NEUT Prior");
  datNEUTNo2p2hXSecHist->SetNameTitle("NEUT No2p2h", "NEUT no2ph2h");

  TH1D* datGenieXSecHist_shapeOnly = (TH1D*)datGenieXSecHist->Clone();
  TH1D* datNEUTRWXSecHist_shapeOnly = (TH1D*)datNEUTRWXSecHist->Clone();
  TH1D* datNuWroXSecHist_shapeOnly = (TH1D*)datNuWroXSecHist->Clone();
  TH1D* datNEUTPriorXSecHist_shapeOnly = (TH1D*)datNEUTPriorXSecHist->Clone();
  TH1D* datNEUTNo2p2hXSecHist_shapeOnly = (TH1D*)datNEUTNo2p2hXSecHist->Clone();

  //dphit:

  TFile* dphitGenieXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/genie/dphit_MCTruth.root");
  TH1D*  dphitGenieXSecHist = (TH1D*) dphitGenieXSecFile->Get("truth");
  TFile* dphitNEUTRWXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/neut/RW/dphit_MCTruth.root");
  TH1D*  dphitNEUTRWXSecHist = (TH1D*) dphitNEUTRWXSecFile->Get("truth");
  TFile* dphitNuWroXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/nuwro/dphit_MCTruth.root");
  TH1D*  dphitNuWroXSecHist = (TH1D*) dphitNuWroXSecFile->Get("truth");
  TH1D*  dphitNuWroXSecHist_ScaleFix = (TH1D*) dphitNuWroXSecFile->Get("truth");
  dphitNuWroXSecHist_ScaleFix->Scale(0.575);

  TFile* dphitNEUTPriorXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/neut_prior/dphit_MCTruth.root");
  TH1D* dphitNEUTPriorXSecHist = (TH1D*) dphitNEUTPriorXSecFile->Get("dif_xSecFitFD");
  TFile* dphitNEUTNo2p2hXSecFile = new TFile("/data/t2k/dolan/fitting/xsecFromMC/neut_no2p2h/dphit_MCTruth.root");
  TH1D* dphitNEUTNo2p2hXSecHist = (TH1D*) dphitNEUTNo2p2hXSecFile->Get("dif_xSecFitFD");

  dphitGenieXSecHist->SetNameTitle("GENIE", "GENIE");
  dphitNEUTRWXSecHist->SetNameTitle("NEUT Nominal", "NEUT Nominal");
  dphitNuWroXSecHist->SetNameTitle("NuWro (prod)", "NuWro (prod)");
  dphitNuWroXSecHist_ScaleFix->SetNameTitle("NuWro (Scale Fix)", "NuWro (Scale Fix)");
  dphitNEUTPriorXSecHist->SetNameTitle("NEUT Prior", "NEUT Prior");
  dphitNEUTNo2p2hXSecHist->SetNameTitle("NEUT No2p2h", "NEUT no2ph2h");

  TH1D* dphitGenieXSecHist_shapeOnly = (TH1D*)dphitGenieXSecHist->Clone();
  TH1D* dphitNEUTRWXSecHist_shapeOnly = (TH1D*)dphitNEUTRWXSecHist->Clone();
  TH1D* dphitNuWroXSecHist_shapeOnly = (TH1D*)dphitNuWroXSecHist->Clone();
  TH1D* dphitNEUTPriorXSecHist_shapeOnly = (TH1D*)dphitNEUTPriorXSecHist->Clone();
  TH1D* dphitNEUTNo2p2hXSecHist_shapeOnly = (TH1D*)dphitNEUTNo2p2hXSecHist->Clone();


  //Shape only stuff:

  for(int i=1;i<=nbinsAuto;i++){
    dif_shapeOnly_xSecFit = makeShapeOnly(dif_xSecFit ,dif_shapeOnly_xSecFit, nbinsAuto);
    dif_shapeOnly_xSecFD = makeShapeOnlyNoError(dif_xSecFD ,dif_shapeOnly_xSecFD, nbinsAuto);
    dif_shapeOnly_xSecMC = makeShapeOnlyNoError(dif_xSecMC ,dif_shapeOnly_xSecMC, nbinsAuto);
    dif_shapeOnly_xSecNeut = makeShapeOnlyNoError(dif_xSecNeut ,dif_shapeOnly_xSecNeut, nbinsAuto);
    dif_shapeOnly_xSecGenie = makeShapeOnlyNoError(dif_xSecGenie ,dif_shapeOnly_xSecGenie, nbinsAuto);
    dif_shapeOnly_xSecNiwgNeut = makeShapeOnlyNoError(dif_xSecNiwgNeut ,dif_shapeOnly_xSecNiwgNeut, nbinsAuto);
    dif_shapeOnly_xSecNuwro = makeShapeOnlyNoError(dif_xSecNuwro ,dif_shapeOnly_xSecNuwro, nbinsAuto);
    dif_shapeOnly_xSecNeutNo2p2h = makeShapeOnlyNoError(dif_xSecNeutNo2p2h ,dif_shapeOnly_xSecNeutNo2p2h, nbinsAuto);

    datGenieXSecHist_shapeOnly = makeShapeOnlyNoError(datGenieXSecHist_shapeOnly, datGenieXSecHist_shapeOnly, nbinsAuto);
    datNEUTRWXSecHist_shapeOnly = makeShapeOnlyNoError(datNEUTRWXSecHist_shapeOnly, datNEUTRWXSecHist_shapeOnly, nbinsAuto);
    datNuWroXSecHist_shapeOnly = makeShapeOnlyNoError(datNuWroXSecHist_shapeOnly, datNuWroXSecHist_shapeOnly, nbinsAuto);
    datNEUTPriorXSecHist_shapeOnly = makeShapeOnlyNoError(datNEUTPriorXSecHist_shapeOnly, datNEUTPriorXSecHist_shapeOnly, nbinsAuto);
    datNEUTNo2p2hXSecHist_shapeOnly = makeShapeOnlyNoError(datNEUTNo2p2hXSecHist_shapeOnly, datNEUTNo2p2hXSecHist_shapeOnly, nbinsAuto);
    dphitGenieXSecHist_shapeOnly = makeShapeOnlyNoError(dphitGenieXSecHist, dphitGenieXSecHist_shapeOnly, nbinsAuto);
    dphitNEUTRWXSecHist_shapeOnly = makeShapeOnlyNoError(dphitNEUTRWXSecHist, dphitNEUTRWXSecHist_shapeOnly, nbinsAuto);
    dphitNuWroXSecHist_shapeOnly = makeShapeOnlyNoError(dphitNuWroXSecHist, dphitNuWroXSecHist_shapeOnly, nbinsAuto);
    dphitNEUTPriorXSecHist_shapeOnly = makeShapeOnlyNoError(dphitNEUTPriorXSecHist, dphitNEUTPriorXSecHist_shapeOnly, nbinsAuto);
    dphitNEUTNo2p2hXSecHist_shapeOnly = makeShapeOnlyNoError(dphitNEUTNo2p2hXSecHist, dphitNEUTNo2p2hXSecHist_shapeOnly, nbinsAuto);
  }

  TH1D* effFluxErr = (TH1D*) propErrFile->Get("effFluxErr");

  TH1D* dif_xSecFit_allError = new TH1D(*dif_xSecFit);
  double newErrXsec;
  double newErrNevts;
  for(int i=1;i<=nbinsAuto;i++){
    newErrXsec = sqrt( pow(dif_xSecFit_allError->GetBinError(i),2) + pow(effFluxErr->GetBinContent(i)*dif_xSecFit_allError->GetBinContent(i),2) );
    newErrNevts = sqrt( pow(fitSpect->GetBinError(i),2) + pow(effFluxErr->GetBinContent(i)*fitSpect->GetBinContent(i),2) );
    dif_xSecFit_allError->SetBinError(i, newErrXsec);
    (*covarNevts)(i-1,i-1) = newErrNevts*newErrNevts;
  }

  //Calculate Corrolation Matrix
  TMatrixD cormatrix(nbinsAuto,nbinsAuto);
  for(int r=0;r<nbinsAuto;r++){
    for(int c=0;c<nbinsAuto;c++){
      cormatrix[r][c]= (*covarNevts)[r][c]/sqrt((((*covarNevts)[r][r])*((*covarNevts))[c][c]));
    }
  }

  //Calculate XSec Covar Matrix
  TMatrixD covarXsec(nbinsAuto,nbinsAuto);
  for(int r=0;r<nbinsAuto;r++){
    for(int c=0;c<nbinsAuto;c++){
      covarXsec[r][c]= (cormatrix)[r][c]*(dif_xSecFit_allError->GetBinError(r+1))*(dif_xSecFit_allError->GetBinError(c+1)) ;
    }
  }

  double selfChi2 = calcChi2(dif_xSecFit_allError, dif_xSecFit_allError, covarXsec);
  double mcChi2 = calcChi2(dif_xSecFit_allError, dif_xSecMC, covarXsec);
  double fdChi2 = calcChi2(dif_xSecFit_allError, dif_xSecFD, covarXsec);

  cout << "Chi2 tests for self, MC, FD are " << selfChi2 << ", " << mcChi2 << " and " << fdChi2 << endl; 

  TH1D* chi2Hist = new TH1D("chi2Hist", "chi2Hist", 3, 0, 2);
  chi2Hist->GetXaxis()->SetBinLabel(1, "selfchi2");
  chi2Hist->GetXaxis()->SetBinLabel(2, "mcChi2");
  chi2Hist->GetXaxis()->SetBinLabel(3, "fdChi2");

  chi2Hist->SetBinContent(1, selfChi2);
  chi2Hist->SetBinContent(2, mcChi2);
  chi2Hist->SetBinContent(3, fdChi2);

  TH1D* chi2Hist_realdata = new TH1D("chi2Hist_realdata", "chi2Hist_realdata", 6, 0, 6);
  TH1D* chi2Hist_realdata_dat = new TH1D("chi2Hist_realdata_dat", "chi2Hist_realdata_dat", 6, 0, 6);
  TH1D* chi2Hist_realdata_dphit = new TH1D("chi2Hist_realdata_dphit", "chi2Hist_realdata_dphit", 6, 0, 6);

  TH1D* chi2Hist_realdata_noNuWroProd = new TH1D("chi2Hist_realdata_noNuWroProd", "chi2Hist_realdata_noNuWroProd", 5, 0, 5);
  TH1D* chi2Hist_realdata_dat_noNuWroProd = new TH1D("chi2Hist_realdata_dat_noNuWroProd", "chi2Hist_realdata_dat_noNuWroProd", 5, 0, 5);
  TH1D* chi2Hist_realdata_dphit_noNuWroProd = new TH1D("chi2Hist_realdata_dphit_noNuWroProd", "chi2Hist_realdata_dphit_noNuWroProd", 5, 0, 5);

  double neutChi2 = 0, neutChi2_dat = 0, neutChi2_dphit = 0;
  double neutRWChi2 = 0, neutRWChi2_dat = 0, neutRWChi2_dphit = 0;
  double genieChi2 = 0, genieChi2_dat = 0, genieChi2_dphit = 0;
  double nuWroChi2 = 0, nuWroChi2_dat = 0, nuWroChi2_dphit = 0;
  double nuWroFixChi2 = 0, nuWroFixChi2_dat = 0, nuWroFixChi2_dphit = 0;
  double neutNo2p2hChi2 = 0, neutNo2p2hChi2_dat = 0, neutNo2p2hChi2_dphit = 0;

  if(dif_xSecFit_allError->GetNbinsX() == dif_xSecNeut->GetNbinsX()){
    cout << "calc dpt chi2" << endl;
    neutChi2 = calcChi2(dif_xSecFit_allError, dif_xSecNeut, covarXsec);
    neutRWChi2 = calcChi2(dif_xSecFit_allError, dif_xSecNiwgNeut, covarXsec);
    genieChi2 = calcChi2(dif_xSecFit_allError, dif_xSecGenie, covarXsec);
    nuWroChi2 = calcChi2(dif_xSecFit_allError, dif_xSecNuwro, covarXsec);
    nuWroFixChi2 = calcChi2(dif_xSecFit_allError, dif_xSecNuwroFix, covarXsec);
    neutNo2p2hChi2 = calcChi2(dif_xSecFit_allError, dif_xSecNeutNo2p2h, covarXsec);
  }

  if(dif_xSecFit_allError->GetNbinsX() == datGenieXSecHist->GetNbinsX()){
    cout << "calc dat chi2" << endl;
    neutChi2_dat = calcChi2(dif_xSecFit_allError, datNEUTPriorXSecHist, covarXsec);
    neutRWChi2_dat = calcChi2(dif_xSecFit_allError, datNEUTRWXSecHist, covarXsec);
    genieChi2_dat = calcChi2(dif_xSecFit_allError, datGenieXSecHist, covarXsec);
    nuWroChi2_dat = calcChi2(dif_xSecFit_allError, datNuWroXSecHist, covarXsec);
    nuWroFixChi2_dat = calcChi2(dif_xSecFit_allError, datNuWroXSecHist_ScaleFix, covarXsec);
    neutNo2p2hChi2_dat = calcChi2(dif_xSecFit_allError, datNEUTNo2p2hXSecHist, covarXsec);
  }

  if(dif_xSecFit_allError->GetNbinsX() == dphitGenieXSecHist->GetNbinsX()){
    cout << "calc dphit chi2" << endl;
    neutChi2_dphit = calcChi2(dif_xSecFit_allError, dphitNEUTPriorXSecHist, covarXsec);
    neutRWChi2_dphit = calcChi2(dif_xSecFit_allError, dphitNEUTRWXSecHist, covarXsec);
    genieChi2_dphit = calcChi2(dif_xSecFit_allError, dphitGenieXSecHist, covarXsec);
    nuWroChi2_dphit = calcChi2(dif_xSecFit_allError, dphitNuWroXSecHist, covarXsec);
    nuWroFixChi2_dphit = calcChi2(dif_xSecFit_allError, dphitNuWroXSecHist_ScaleFix, covarXsec);
    neutNo2p2hChi2_dphit = calcChi2(dif_xSecFit_allError, dphitNEUTNo2p2hXSecHist, covarXsec);
  }

  cout << endl << "Real data chi2 results: " << endl;
  cout << "NEUT Prior:  " << neutChi2 << endl;
  cout << "Neut Nominal:  " << neutRWChi2 << endl;
  cout << "GENIE:  " << genieChi2 << endl;
  cout << "NuWro:  " << nuWroChi2 << endl;
  cout << "NuWro Scale Fix:  " << nuWroFixChi2 << endl << endl;
  cout << "Neut No 2p2h:  " << neutNo2p2hChi2 << endl << endl;

  chi2Hist_realdata->SetBinContent(1, neutChi2);
  chi2Hist_realdata->SetBinContent(2, neutRWChi2);
  chi2Hist_realdata->SetBinContent(3, genieChi2);
  chi2Hist_realdata->SetBinContent(4, nuWroChi2);
  chi2Hist_realdata->SetBinContent(5, nuWroFixChi2);
  chi2Hist_realdata->SetBinContent(6, neutNo2p2hChi2);

  chi2Hist_realdata_dat->SetBinContent(1, neutChi2_dat);
  chi2Hist_realdata_dat->SetBinContent(2, neutRWChi2_dat);
  chi2Hist_realdata_dat->SetBinContent(3, genieChi2_dat);
  chi2Hist_realdata_dat->SetBinContent(4, nuWroChi2_dat);
  chi2Hist_realdata_dat->SetBinContent(5, nuWroFixChi2_dat);
  chi2Hist_realdata_dat->SetBinContent(6, neutNo2p2hChi2_dat);

  chi2Hist_realdata_dphit->SetBinContent(1, neutChi2_dphit);
  chi2Hist_realdata_dphit->SetBinContent(2, neutRWChi2_dphit);
  chi2Hist_realdata_dphit->SetBinContent(3, genieChi2_dphit);
  chi2Hist_realdata_dphit->SetBinContent(4, nuWroChi2_dphit);
  chi2Hist_realdata_dphit->SetBinContent(5, nuWroFixChi2_dphit);
  chi2Hist_realdata_dphit->SetBinContent(6, neutNo2p2hChi2_dphit);

  chi2Hist_realdata->GetXaxis()->SetBinLabel(1, "NEUT Prior");
  chi2Hist_realdata->GetXaxis()->SetBinLabel(2, "NEUT Nominal");
  chi2Hist_realdata->GetXaxis()->SetBinLabel(3, "GENIE");
  chi2Hist_realdata->GetXaxis()->SetBinLabel(4, "NuWro (prod)");
  chi2Hist_realdata->GetXaxis()->SetBinLabel(5, "NuWro (rescale)");
  chi2Hist_realdata->GetXaxis()->SetBinLabel(6, "NEUT No 2p2h");

  chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(1, "NEUT Prior");
  chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(2, "NEUT Nominal");
  chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(3, "GENIE");
  chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(4, "NuWro (prod)");
  chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(5, "NuWro (rescale)");
  chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(6, "NEUT No 2p2h");

  chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(1, "NEUT Prior");
  chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(2, "NEUT Nominal");
  chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(3, "GENIE");
  chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(4, "NuWro (prod)");
  chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(5, "NuWro (rescale)");
  chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(6, "NEUT No 2p2h");


  chi2Hist_realdata_noNuWroProd->SetBinContent(1, neutChi2);
  chi2Hist_realdata_noNuWroProd->SetBinContent(2, neutRWChi2);
  chi2Hist_realdata_noNuWroProd->SetBinContent(3, genieChi2);
  chi2Hist_realdata_noNuWroProd->SetBinContent(4, nuWroFixChi2);
  chi2Hist_realdata_noNuWroProd->SetBinContent(5, neutNo2p2hChi2);

  chi2Hist_realdata_dat_noNuWroProd->SetBinContent(1, neutChi2_dat);
  chi2Hist_realdata_dat_noNuWroProd->SetBinContent(2, neutRWChi2_dat);
  chi2Hist_realdata_dat_noNuWroProd->SetBinContent(3, genieChi2_dat);
  chi2Hist_realdata_dat_noNuWroProd->SetBinContent(4, nuWroFixChi2_dat);
  chi2Hist_realdata_dat_noNuWroProd->SetBinContent(5, neutNo2p2hChi2_dat);

  chi2Hist_realdata_dphit_noNuWroProd->SetBinContent(1, neutChi2_dphit);
  chi2Hist_realdata_dphit_noNuWroProd->SetBinContent(2, neutRWChi2_dphit);
  chi2Hist_realdata_dphit_noNuWroProd->SetBinContent(3, genieChi2_dphit);
  chi2Hist_realdata_dphit_noNuWroProd->SetBinContent(4, nuWroFixChi2_dphit);
  chi2Hist_realdata_dphit_noNuWroProd->SetBinContent(5, neutNo2p2hChi2_dphit);

  chi2Hist_realdata_noNuWroProd->GetXaxis()->SetBinLabel(1, "NEUT Prior");
  chi2Hist_realdata_noNuWroProd->GetXaxis()->SetBinLabel(2, "NEUT Nominal");
  chi2Hist_realdata_noNuWroProd->GetXaxis()->SetBinLabel(3, "GENIE");
  chi2Hist_realdata_noNuWroProd->GetXaxis()->SetBinLabel(4, "NuWro (rescale)");
  chi2Hist_realdata_noNuWroProd->GetXaxis()->SetBinLabel(5, "NEUT No 2p2h");

  chi2Hist_realdata_dat_noNuWroProd->GetXaxis()->SetBinLabel(1, "NEUT Prior");
  chi2Hist_realdata_dat_noNuWroProd->GetXaxis()->SetBinLabel(2, "NEUT Nominal");
  chi2Hist_realdata_dat_noNuWroProd->GetXaxis()->SetBinLabel(3, "GENIE");
  chi2Hist_realdata_dat_noNuWroProd->GetXaxis()->SetBinLabel(4, "NuWro (rescale)");
  chi2Hist_realdata_dat_noNuWroProd->GetXaxis()->SetBinLabel(5, "NEUT No 2p2h");

  chi2Hist_realdata_dphit_noNuWroProd->GetXaxis()->SetBinLabel(1, "NEUT Prior");
  chi2Hist_realdata_dphit_noNuWroProd->GetXaxis()->SetBinLabel(2, "NEUT Nominal");
  chi2Hist_realdata_dphit_noNuWroProd->GetXaxis()->SetBinLabel(3, "GENIE");
  chi2Hist_realdata_dphit_noNuWroProd->GetXaxis()->SetBinLabel(4, "NuWro (rescale)");
  chi2Hist_realdata_dphit_noNuWroProd->GetXaxis()->SetBinLabel(5, "NEUT No 2p2h");


  //Errors from stats pulls study for proceedings:
  // dif_xSecFit->SetBinError(1, dif_xSecFit->GetBinContent(1)*(0.11));
  // dif_xSecFit->SetBinError(2, dif_xSecFit->GetBinContent(2)*(0.13));
  // dif_xSecFit->SetBinError(3, dif_xSecFit->GetBinContent(3)*(0.157));
  // dif_xSecFit->SetBinError(4, dif_xSecFit->GetBinContent(4)*(0.147));
  // dif_xSecFit->SetBinError(5, dif_xSecFit->GetBinContent(5)*(0.144));
  // dif_xSecFit->SetBinError(6, dif_xSecFit->GetBinContent(6)*(0.26));
  // dif_xSecFit->SetBinError(7, dif_xSecFit->GetBinContent(7)*(0.29));
  // dif_xSecFit->SetBinError(8, dif_xSecFit->GetBinContent(8)*(0.35));

  outputFile->cd();

  chi2Hist->Write();
  chi2Hist_realdata->Write();
  chi2Hist_realdata_dat->Write();
  chi2Hist_realdata_dphit->Write();

  chi2Hist_realdata_noNuWroProd->Write();
  chi2Hist_realdata_dat_noNuWroProd->Write();
  chi2Hist_realdata_dphit_noNuWroProd->Write();


  effHisto->Write();
  effHisto_fd->Write();

  dif_xSecFit->SetLineColor(kRed);
  dif_xSecFit->SetLineWidth(2);

  dif_xSecFit_allError->SetLineColor(kRed);
  dif_xSecFit_allError->SetLineWidth(2);

  dif_xSecMC->SetLineColor(kBlue);
  dif_xSecMC->SetLineWidth(2);
  dif_xSecMC->SetXTitle("#deltap_{T}(GeV/c)");
  dif_xSecMC->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (Nucleon^{-1} cm^{2} GeV^{-1})");


  dif_xSecFD->SetLineColor(kGreen+2);
  dif_xSecFD->SetLineWidth(2);

  dif_xSecFit_allError->Write("dif_xSecFit_allError");
  dif_xSecFit->Write();
  dif_xSecMC->Write();
  dif_xSecFD->Write();

  if(isRealData){
    dif_xSecNeut->SetLineColor(kRed);
    dif_xSecNeut->SetLineWidth(2);
    datNEUTPriorXSecHist->SetLineColor(kRed);
    datNEUTPriorXSecHist->SetLineWidth(2);
    dphitNEUTPriorXSecHist->SetLineColor(kRed);
    dphitNEUTPriorXSecHist->SetLineWidth(2);

    dif_xSecNiwgNeut->SetLineColor(kRed);
    dif_xSecNiwgNeut->SetLineWidth(2);
    dif_xSecNiwgNeut->SetLineStyle(2);
    datNEUTRWXSecHist->SetLineColor(kRed);
    datNEUTRWXSecHist->SetLineWidth(2);
    datNEUTRWXSecHist->SetLineStyle(2);
    dphitNEUTRWXSecHist->SetLineColor(kRed);
    dphitNEUTRWXSecHist->SetLineWidth(2);
    dphitNEUTRWXSecHist->SetLineStyle(2);

    dif_xSecNuwro->SetLineColor(kBlue);
    dif_xSecNuwro->SetLineWidth(2);
    dif_xSecNuwro->SetXTitle("#deltap_{T}(GeV/c)");
    dif_xSecNuwro->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (Nucleon^{-1} cm^{2} GeV^{-1})");
    datNuWroXSecHist->SetLineColor(kBlue);
    datNuWroXSecHist->SetLineWidth(2);
    datNuWroXSecHist->SetXTitle("#delta#alpha_{T}(rads)");
    datNuWroXSecHist->SetYTitle("#frac{d#sigma}{d#delta#alpha_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");
    dphitNuWroXSecHist->SetLineColor(kBlue);
    dphitNuWroXSecHist->SetLineWidth(2);
    dphitNuWroXSecHist->SetXTitle("#delta#phi_{T}(rads)");
    dphitNuWroXSecHist->SetYTitle("#frac{d#sigma}{d#delta#phi_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");

    dif_xSecNuwroFix->SetLineColor(kBlue);
    dif_xSecNuwroFix->SetLineWidth(2);
    dif_xSecNuwroFix->SetLineStyle(2);
    datNuWroXSecHist_ScaleFix->SetLineColor(kBlue);
    datNuWroXSecHist_ScaleFix->SetLineWidth(2);
    datNuWroXSecHist_ScaleFix->SetLineStyle(2);
    dphitNuWroXSecHist_ScaleFix->SetLineColor(kBlue);
    dphitNuWroXSecHist_ScaleFix->SetLineWidth(2);
    dphitNuWroXSecHist_ScaleFix->SetLineStyle(2);


    dif_xSecGenie->SetLineColor(kGreen-2);
    dif_xSecGenie->SetLineWidth(2);
    datGenieXSecHist->SetLineColor(kGreen-2);
    datGenieXSecHist->SetLineWidth(2);
    dphitGenieXSecHist->SetLineColor(kGreen-2);
    dphitGenieXSecHist->SetLineWidth(2);

    dif_xSecNeutNo2p2h->SetLineColor(kRed+2);
    dif_xSecNeutNo2p2h->SetLineWidth(2);
    dif_xSecNeutNo2p2h->SetLineStyle(3);
    datNEUTNo2p2hXSecHist->SetLineColor(kRed+2);
    datNEUTNo2p2hXSecHist->SetLineWidth(2);
    datNEUTNo2p2hXSecHist->SetLineStyle(3);
    dphitNEUTNo2p2hXSecHist->SetLineColor(kRed+2);
    dphitNEUTNo2p2hXSecHist->SetLineWidth(2);
    dphitNEUTNo2p2hXSecHist->SetLineStyle(3);

  }


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

  fitSpectAB->SetLineColor(kRed);
  fitSpectAB->SetLineWidth(2);
  fitSpectAB->SetLineStyle(2);
  mcSpectAB->SetLineColor(kBlue);
  mcSpectAB->SetLineWidth(2);
  mcSpectAB->SetLineStyle(2);
  fakeDataTrueSpectAB->SetLineColor(kGreen+2);
  fakeDataTrueSpectAB->SetLineWidth(2);
  fakeDataTrueSpectAB->SetLineStyle(2);
  
  fitSpectAB->Write();
  mcSpectAB->Write();
  fakeDataTrueSpectAB->Write();

  mcSpect_sigInOOPS->SetLineColor(kBlue);
  mcSpect_sigInOOPS->SetLineWidth(2);
  mcSpect_sigInOOPS->SetLineStyle(2);
  fakeDataTrueSpect_sigInOOPS->SetLineColor(kGreen+2);
  fakeDataTrueSpect_sigInOOPS->SetLineWidth(2);
  fakeDataTrueSpect_sigInOOPS->SetLineStyle(2);
  mcSpect_OOPSInSig->SetLineColor(kBlue);
  mcSpect_OOPSInSig->SetLineWidth(2);
  mcSpect_OOPSInSig->SetLineStyle(2);
  fakeDataTrueSpect_OOPSInSig->SetLineColor(kGreen+2);
  fakeDataTrueSpect_OOPSInSig->SetLineWidth(2);
  fakeDataTrueSpect_OOPSInSig->SetLineStyle(2);

  mcSpect_sigInOOPS->Write();
  fakeDataTrueSpect_sigInOOPS->Write();
  mcSpect_OOPSInSig->Write();
  fakeDataTrueSpect_OOPSInSig->Write();

  mcSpect_sigInOOPS->Divide(mcSpect);
  fakeDataTrueSpect_sigInOOPS->Divide(fakeDataTrueSpect);
  mcSpect_OOPSInSig->Divide(mcSpect);
  fakeDataTrueSpect_OOPSInSig->Divide(fakeDataTrueSpect);

  mcSpect_sigInOOPS->Write("mcSpect_sigInOOPS_norm");
  fakeDataTrueSpect_sigInOOPS->Write("fakeDataTrueSpect_sigInOOPS_norm");
  mcSpect_OOPSInSig->Write("mcSpect_OOPSInSig_norm");
  fakeDataTrueSpect_OOPSInSig->Write("fakeDataTrueSpect_OOPSInSig_norm");

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



  totalSpectFit_allSam->SetLineColor(kRed);
  totalSpectFit_allSam->SetLineWidth(2);
  totalSpectMC_allSam->SetLineColor(kBlue);
  totalSpectMC_allSam->SetLineWidth(2);
  totalSpectFD_allSam->SetLineColor(kGreen+2);
  totalSpectFD_allSam->SetLineWidth(2);



  TCanvas* canv = new TCanvas("Signal Comp","Signal Comp");
  fitSpect->Draw();
  mcSpect->Draw("same");
  fakeDataTrueSpect->Draw("same");
  canv->Write();

  TCanvas* canvAB = new TCanvas("Signal Comp AB","Signal Comp AB");
  fitSpectAB->Draw();
  mcSpectAB->Draw("same");
  fakeDataTrueSpectAB->Draw("same");
  canvAB->Write();

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

  TCanvas* canvxs_allError = new TCanvas("xsecCanv_allError","xsecCanv_allError");
  dif_xSecMC->Draw();
  dif_xSecFD->Draw("same");
  dif_xSecFit_allError->Draw("same");
  canvxs_allError->Write();

  if(isRealData){
    TCanvas* canvxs_genComp = new TCanvas("xsecCanv_genComp","xsecCanv_genComp");
    dif_xSecNuwro->Draw();
    dif_xSecNuwroFix->Draw("same");
    dif_xSecNeut->Draw("same");
    dif_xSecNiwgNeut->Draw("same");
    dif_xSecNeutNo2p2h->Draw("same");
    dif_xSecGenie->Draw("same");
    dif_xSecFit_allError->Draw("same");
    canvxs_genComp->Write();

    TCanvas* canvxs_genComp_dat = new TCanvas("xsecCanv_genComp_dat","xsecCanv_genComp_dat");
    datNuWroXSecHist->Draw("HIST");
    datNuWroXSecHist_ScaleFix->Draw("sameHIST");
    datNEUTPriorXSecHist->Draw("sameHIST");
    datNEUTRWXSecHist->Draw("sameHIST");
    datNEUTNo2p2hXSecHist->Draw("sameHIST");
    datGenieXSecHist->Draw("sameHIST");
    dif_xSecFit_allError->Draw("same");
    canvxs_genComp_dat->Write();

    TCanvas* canvxs_genComp_dphit = new TCanvas("xsecCanv_genComp_dphit","xsecCanv_genComp_dphit");
    dphitNuWroXSecHist->Draw("HIST");
    dphitNuWroXSecHist_ScaleFix->Draw("sameHIST");
    dphitNEUTPriorXSecHist->Draw("sameHIST");
    dphitNEUTRWXSecHist->Draw("sameHIST");
    dphitNEUTNo2p2hXSecHist->Draw("sameHIST");
    dphitGenieXSecHist->Draw("sameHIST");
    dif_xSecFit_allError->Draw("same");
    canvxs_genComp_dphit->Write();
  }

  dif_xSecNuwro->Write();
  dif_xSecNuwroFix->Write();
  dif_xSecNeut->Write();
  dif_xSecNiwgNeut->Write();
  dif_xSecNeutNo2p2h->Write();
  dif_xSecGenie->Write();
  datNuWroXSecHist->Write();
  datNuWroXSecHist_ScaleFix->Write();
  datNEUTPriorXSecHist->Write();
  datNEUTRWXSecHist->Write();
  datNEUTNo2p2hXSecHist->Write();
  datGenieXSecHist->Write();
  dphitNuWroXSecHist->Write();
  dphitNuWroXSecHist_ScaleFix->Write();
  dphitNEUTPriorXSecHist->Write();
  dphitNEUTRWXSecHist->Write();
  dphitNEUTNo2p2hXSecHist->Write();
  dphitGenieXSecHist->Write();


  dif_shapeOnly_xSecFit->Write();
  dif_shapeOnly_xSecFD->Write();
  dif_shapeOnly_xSecMC->Write();
  dif_shapeOnly_xSecNeut->Write();
  dif_shapeOnly_xSecGenie->Write();
  dif_shapeOnly_xSecNiwgNeut->Write();
  dif_shapeOnly_xSecNuwro->Write();
  dif_shapeOnly_xSecNeutNo2p2h->Write();
  datGenieXSecHist_shapeOnly->Write();
  datNEUTRWXSecHist_shapeOnly->Write();
  datNuWroXSecHist_shapeOnly->Write();
  datNEUTPriorXSecHist_shapeOnly->Write();
  datNEUTNo2p2hXSecHist_shapeOnly->Write();
  dphitGenieXSecHist_shapeOnly->Write();
  dphitNEUTRWXSecHist_shapeOnly->Write();
  dphitNuWroXSecHist_shapeOnly->Write();
  dphitNEUTPriorXSecHist_shapeOnly->Write();
  dphitNEUTNo2p2hXSecHist_shapeOnly->Write();

  mcSpectOOPS_T->Write();
  fakeDataTrueSpectOOPS_T->Write();
  fitSpectOOPS_T->Write();
  effHistoOOPS->Write();
  effHistoOOPS_fd->Write();

  cout << "xsec chi2 hist with current errors" << endl;
  dif_xSecFit->Chi2Test(dif_xSecFD, "WWP");

  // for(int i=1;i<=nbinsAuto;i++){
  //   dif_xSecFit->SetBinError(i, sqrt(dif_xSecFit->GetBinError(i)*dif_xSecFit->GetBinError(i) + (0.2*dif_xSecFit->GetBinContent(i))*(0.2*dif_xSecFit->GetBinContent(i))) );
  // }

  cout << "xsec chi2 hist with exp errors" << endl;
  dif_xSecFit->Chi2Test(dif_xSecFD, "WWP");

  covarNevts->Write("covarNevts");
  cormatrix.Write("cormatrix");
  covarXsec.Write("covarXsec");

  //cout << "Original cov matrix: " << endl;
  //covarNevtsInp->Print();

  //cout << "Adjusted cov matrix: " << endl;
  //covarNevts->Print();

  //cout << "Cor matrix: " << endl;
  //cormatrix.Print();

  dif_xSecFit_allError->Print("all");

  //cout << "Covar Xsec matrix: " << endl;
  //covarXsec.Print();

  intFluxWeightHisto->Write("intFluxWeightHisto");

  datGenieXSecFile->Close();
  datNEUTRWXSecFile->Close();
  datNuWroXSecFile->Close();
  datNEUTPriorXSecFile->Close();
  datNEUTPriorXSecFile->Close();
  dphitGenieXSecFile->Close();
  dphitNEUTRWXSecFile->Close();
  dphitNuWroXSecFile->Close();
  dphitNEUTPriorXSecFile->Close();
  dphitNEUTPriorXSecFile->Close();

  cout << "Finished calcXSecWithErrors2" << endl;

}
