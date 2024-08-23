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
#include <TLegend.h>
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
  cout << "making shape only on histo: " << h1->GetTitle() << endl;
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
void calcXsecWithErrors2 (TString propErrFilename, TString mcFilename, TString fitResultFilename, TString fakeDataFilename, TString outputFileName, double potRatio, double MCPOT=3.316, bool isRealData=false, TString var="dpt", bool useInputEff=false, bool TPCOnlySig=false, int pAngleRestrict=0){
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
  mcSpectOOPS_T->Scale(potRatio);


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
    // if the input eff fromt he FD may be incorrect (e.g. basket only) use the input MC eff
    if(useInputEff) fakeDataTrueSpect_T->SetBinContent(i, fakeDataTrueSpect->GetBinContent(i)/eff);
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
  TH1D* dif_xSecNeutBask = new TH1D("NEUT 6D","NEUT 6D",nbinsAuto,binsAuto->GetArray());

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

  TFile* dptNEUTPriorXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dptref/dptv3/asimov/5_1reg/quickFtX_xsecOut.root");
  TFile* dptGENIEXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dptref/dptv3/genie/5_1reg/quickFtX_xsecOut.root");
  TFile* dptNuWroXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dptref/dptv3/nuWro/5_1reg/quickFtX_xsecOut.root");
  TFile* dptNEUTNo2p2hXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dptref/dptv3/neutNo2p2h_noSB/5_1reg/quickFtX_xsecOut.root");
  TFile* dptNEUTBaskXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dptref/dptv3/neutBask/5_1reg/quickFtX_xsecOut.root");

  if(dptNEUTPriorXSecFile->Get("Fake Data") && dptGENIEXSecFile->Get("Fake Data") && dptNEUTBaskXSecFile->Get("Fake Data")
     && dptNuWroXSecFile->Get("Fake Data") && dptNEUTNo2p2hXSecFile->Get("Fake Data") && var=="dpt"){
    dif_xSecNeut = (TH1D*) dptNEUTPriorXSecFile->Get("Fake Data");
    dif_xSecNeut->SetNameTitle("Neut6B", "Neut6B");
    dif_xSecGenie = (TH1D*) dptGENIEXSecFile->Get("Fake Data");
    dif_xSecGenie->SetNameTitle("Genie", "Genie");
    dif_xSecNuwro = (TH1D*) dptNuWroXSecFile->Get("Fake Data");
    dif_xSecNuwro->SetNameTitle("NuWro", "NuWro");
    dif_xSecNeutNo2p2h = (TH1D*) dptNEUTNo2p2hXSecFile->Get("Fake Data");
    dif_xSecNeutNo2p2h->SetNameTitle("Neut6BNo2p2h", "Neut6BNo2p2h");    
    dif_xSecNeutBask = (TH1D*) dptNEUTBaskXSecFile->Get("Fake Data");
    dif_xSecNeutBask->SetNameTitle("Neut6D", "Neut6D");   
  }
  else{
    cout << "Pre generated MC histos for dpt are not ready, run calcXsecWithErrors2 again to fill them" << endl;
    if(!dptNEUTPriorXSecFile->Get("Fake Data")) cout << "Issue is with dptNEUTPriorXSecFile" << endl;
    if(!dptGENIEXSecFile->Get("Fake Data")) cout << "Issue is with dptGENIEXSecFile" << endl;
    if(!dptNuWroXSecFile->Get("Fake Data")) cout << "Issue is with dptNuWroXSecFile" << endl;
    if(!dptNEUTNo2p2hXSecFile->Get("Fake Data")) cout << "Issue is with dptNEUTNo2p2hXSecFile" << endl;
  }
  //dat:

  TFile* datNEUTPriorXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/datref/datv3/asimov/5_1reg/quickFtX_xsecOut.root");
  TFile* datGENIEXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/datref/datv3/genie/5_1reg/quickFtX_xsecOut.root");
  TFile* datNuWroXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/datref/datv3/nuWro/5_1reg/quickFtX_xsecOut.root");
  TFile* datNEUTNo2p2hXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/datref/datv3/neutNo2p2h_noSB/5_1reg/quickFtX_xsecOut.root");
  TFile* datNEUTBaskXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/datref/datv3/neutBask/5_1reg/quickFtX_xsecOut.root");

  TH1D* datGenieXSecHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());
  TH1D* datNuWroXSecHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());
  TH1D* datNEUTPriorXSecHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());
  TH1D* datNEUTNo2p2hXSecHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());
  TH1D* datNEUTBaskHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());

  if(datNEUTPriorXSecFile->Get("Fake Data") && datGENIEXSecFile->Get("Fake Data") && datNEUTBaskXSecFile->Get("Fake Data")
     && datNuWroXSecFile->Get("Fake Data") && datNEUTNo2p2hXSecFile->Get("Fake Data") && var=="dat"){
    datGenieXSecHist = (TH1D*) datGENIEXSecFile->Get("Fake Data");
    datGenieXSecHist->SetNameTitle("Genie", "Genie");
    datNuWroXSecHist = (TH1D*) datNuWroXSecFile->Get("Fake Data");
    datNuWroXSecHist->SetNameTitle("NuWro", "NuWro");
    datNEUTPriorXSecHist = (TH1D*) datNEUTPriorXSecFile->Get("Fake Data");
    datNEUTPriorXSecHist->SetNameTitle("Neut6B", "Neut6B");
    datNEUTNo2p2hXSecHist = (TH1D*) datNEUTNo2p2hXSecFile->Get("Fake Data");
    datNEUTNo2p2hXSecHist->SetNameTitle("Neut6BNo2p2h", "Neut6BNo2p2h");    
    datNEUTBaskHist = (TH1D*) datNEUTBaskXSecFile->Get("Fake Data");
    datNEUTBaskHist->SetNameTitle("Neut6D", "Neut6D"); 
  }
  else{
    cout << "Pre generated MC histos for dat are not ready, run calcXsecWithErrors2 again to fill them" << endl;
    if(!datNEUTPriorXSecFile->Get("Fake Data")) cout << "Issue is with datNEUTPriorXSecFile" << endl;
    if(!datGENIEXSecFile->Get("Fake Data")) cout << "Issue is with datGENIEXSecFile" << endl;
    if(!datNuWroXSecFile->Get("Fake Data")) cout << "Issue is with datNuWroXSecFile" << endl;
    if(!datNEUTNo2p2hXSecFile->Get("Fake Data")) cout << "Issue is with datNEUTNo2p2hXSecFile" << endl;
  }

  datGenieXSecHist->SetNameTitle("GENIE", "GENIE");
  datNuWroXSecHist->SetNameTitle("NuWro (prod)", "NuWro (prod)");
  datNEUTPriorXSecHist->SetNameTitle("NEUT Prior", "NEUT Prior");
  datNEUTNo2p2hXSecHist->SetNameTitle("NEUT No2p2h", "NEUT no2ph2h");
  datNEUTBaskHist->SetNameTitle("NEUT 6D", "NEUT 6D");

  TH1D* datGenieXSecHist_shapeOnly = (TH1D*)datGenieXSecHist->Clone();
  TH1D* datNuWroXSecHist_shapeOnly = (TH1D*)datNuWroXSecHist->Clone();
  TH1D* datNEUTPriorXSecHist_shapeOnly = (TH1D*)datNEUTPriorXSecHist->Clone();
  TH1D* datNEUTNo2p2hXSecHist_shapeOnly = (TH1D*)datNEUTNo2p2hXSecHist->Clone();

  //dphit:

  TFile* dphitNEUTPriorXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dphitref/dphitv3/asimov/3_1reg/quickFtX_xsecOut.root");
  TFile* dphitGENIEXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dphitref/dphitv3/genie/5_1reg/quickFtX_xsecOut.root");
  TFile* dphitNuWroXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dphitref/dphitv3/nuWro/5_1reg/quickFtX_xsecOut.root");
  TFile* dphitNEUTNo2p2hXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dphitref/dphitv3/neutNo2p2h_noSB/5_1reg/quickFtX_xsecOut.root");
  TFile* dphitNEUTBaskXSecFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/dphitref/dphitv3/neutBask/5_1reg/quickFtX_xsecOut.root");

  TH1D* dphitGenieXSecHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());
  TH1D* dphitNuWroXSecHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());
  TH1D* dphitNEUTPriorXSecHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());
  TH1D* dphitNEUTNo2p2hXSecHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());
  TH1D* dphitNEUTBaskHist= new TH1D("h","h",nbinsAuto,binsAuto->GetArray());

  if(dphitNEUTPriorXSecFile->Get("Fake Data") && dphitGENIEXSecFile->Get("Fake Data")  && dphitNEUTBaskXSecFile->Get("Fake Data")
     && dphitNuWroXSecFile->Get("Fake Data") && dphitNEUTNo2p2hXSecFile->Get("Fake Data") && var=="dphit"){
    dphitGenieXSecHist = (TH1D*) dphitGENIEXSecFile->Get("Fake Data");
    dphitGenieXSecHist->SetNameTitle("Genie", "Genie");
    dphitNuWroXSecHist = (TH1D*) dphitNuWroXSecFile->Get("Fake Data");
    dphitNuWroXSecHist->SetNameTitle("NuWro", "NuWro");
    dphitNEUTPriorXSecHist = (TH1D*) dphitNEUTPriorXSecFile->Get("Fake Data");
    dphitNEUTPriorXSecHist->SetNameTitle("Neut6B", "Neut6B");
    dphitNEUTNo2p2hXSecHist = (TH1D*) dphitNEUTNo2p2hXSecFile->Get("Fake Data");
    dphitNEUTNo2p2hXSecHist->SetNameTitle("Neut6BNo2p2h", "Neut6BNo2p2h");   
    dphitNEUTBaskHist = (TH1D*) dphitNEUTBaskXSecFile->Get("Fake Data");
    dphitNEUTBaskHist->SetNameTitle("Neut6D", "Neut6D"); 
  }
  else{
    cout << "Pre generated MC histos for dphit are not ready, run calcXsecWithErrors2 again to fill them" << endl;
    if(!dphitNEUTPriorXSecFile->Get("Fake Data")) cout << "Issue is with dphitNEUTPriorXSecFile" << endl;
    if(!dphitGENIEXSecFile->Get("Fake Data")) cout << "Issue is with dphitGENIEXSecFile" << endl;
    if(!dphitNuWroXSecFile->Get("Fake Data")) cout << "Issue is with dphitNuWroXSecFile" << endl;
    if(!dphitNEUTNo2p2hXSecFile->Get("Fake Data")) cout << "Issue is with dphitNEUTNo2p2hXSecFile" << endl;
  }


  dphitGenieXSecHist->SetNameTitle("GENIE", "GENIE");
  dphitNuWroXSecHist->SetNameTitle("NuWro (prod)", "NuWro (prod)");
  dphitNEUTPriorXSecHist->SetNameTitle("NEUT Prior", "NEUT Prior");
  dphitNEUTNo2p2hXSecHist->SetNameTitle("NEUT No2p2h", "NEUT no2ph2h");
  dphitNEUTBaskHist->SetNameTitle("NEUT 6D", "NEUT 6D");


  TH1D* dphitGenieXSecHist_shapeOnly = (TH1D*)dphitGenieXSecHist->Clone();
  TH1D* dphitNuWroXSecHist_shapeOnly = (TH1D*)dphitNuWroXSecHist->Clone();
  TH1D* dphitNEUTPriorXSecHist_shapeOnly = (TH1D*)dphitNEUTPriorXSecHist->Clone();
  TH1D* dphitNEUTNo2p2hXSecHist_shapeOnly = (TH1D*)dphitNEUTNo2p2hXSecHist->Clone();

  //Shape only stuff:

  for(int i=1;i<=nbinsAuto;i++){
    cout << "Running shape only on inputs" << endl;
    dif_shapeOnly_xSecFit = makeShapeOnly(dif_xSecFit ,dif_shapeOnly_xSecFit, nbinsAuto);
    dif_shapeOnly_xSecFD = makeShapeOnlyNoError(dif_xSecFD ,dif_shapeOnly_xSecFD, nbinsAuto);
    dif_shapeOnly_xSecMC = makeShapeOnlyNoError(dif_xSecMC ,dif_shapeOnly_xSecMC, nbinsAuto);
    if(var=="dpt"){
      cout << "Running shape only on dpt" << endl;
      dif_shapeOnly_xSecNeut = makeShapeOnlyNoError(dif_xSecNeut ,dif_shapeOnly_xSecNeut, nbinsAuto);
      dif_shapeOnly_xSecGenie = makeShapeOnlyNoError(dif_xSecGenie ,dif_shapeOnly_xSecGenie, nbinsAuto);
      dif_shapeOnly_xSecNuwro = makeShapeOnlyNoError(dif_xSecNuwro ,dif_shapeOnly_xSecNuwro, nbinsAuto);
      dif_shapeOnly_xSecNeutNo2p2h = makeShapeOnlyNoError(dif_xSecNeutNo2p2h ,dif_shapeOnly_xSecNeutNo2p2h, nbinsAuto);
    }
    if(var=="dat"){
      cout << "Running shape only on dat" << endl;
      datGenieXSecHist_shapeOnly = makeShapeOnlyNoError(datGenieXSecHist, datGenieXSecHist_shapeOnly, nbinsAuto);
      datNuWroXSecHist_shapeOnly = makeShapeOnlyNoError(datNuWroXSecHist, datNuWroXSecHist_shapeOnly, nbinsAuto);
      datNEUTPriorXSecHist_shapeOnly = makeShapeOnlyNoError(datNEUTPriorXSecHist, datNEUTPriorXSecHist_shapeOnly, nbinsAuto);
      datNEUTNo2p2hXSecHist_shapeOnly = makeShapeOnlyNoError(datNEUTNo2p2hXSecHist, datNEUTNo2p2hXSecHist_shapeOnly, nbinsAuto);
    }
    if(var=="dphit"){
      cout << "Running shape only on dphit" << endl;
      dphitGenieXSecHist_shapeOnly = makeShapeOnlyNoError(dphitGenieXSecHist, dphitGenieXSecHist_shapeOnly, nbinsAuto);
      dphitNuWroXSecHist_shapeOnly = makeShapeOnlyNoError(dphitNuWroXSecHist, dphitNuWroXSecHist_shapeOnly, nbinsAuto);
      dphitNEUTPriorXSecHist_shapeOnly = makeShapeOnlyNoError(dphitNEUTPriorXSecHist, dphitNEUTPriorXSecHist_shapeOnly, nbinsAuto);
      dphitNEUTNo2p2hXSecHist_shapeOnly = makeShapeOnlyNoError(dphitNEUTNo2p2hXSecHist, dphitNEUTNo2p2hXSecHist_shapeOnly, nbinsAuto);
    }
  }
   

  //**********************************************
  //  Add the flux norm, ntgt and eff errors:
  //**********************************************

  TFile* modelEffCovFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/effCov/model/modelEffCovar250Toys.root");
  TFile* dptDetEffCovFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/effCov/det/dptDetCovForEffCorr.root");
  TFile* dphitDetEffCovFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/effCov/det/dphitDetCovForEffCorr.root");
  TFile* datDetEffCovFile = new TFile("/data/t2k/dolan/fitting/feb17_refit/effCov/det/datDetCovForEffCorr.root");

  TMatrixDSym *dptModelCovar = (TMatrixDSym*) modelEffCovFile->Get("covar_norm_dpt");
  TMatrixDSym *dphitModelCovar = (TMatrixDSym*) modelEffCovFile->Get("covar_norm_dphit");
  TMatrixDSym *datModelCovar = (TMatrixDSym*) modelEffCovFile->Get("covar_norm_dat");

  TMatrixDSym *dptDetCovar = (TMatrixDSym*) dptDetEffCovFile->Get("covMat_mean_norm");
  TMatrixDSym *dphitDetCovar = (TMatrixDSym*) dphitDetEffCovFile->Get("covMat_mean_norm");
  TMatrixDSym *datDetCovar = (TMatrixDSym*) datDetEffCovFile->Get("covMat_mean_norm");

  double ntgterror=0.0067;
  double fluxNormErr=0.085;

  double dptNucFSIErr[8]   = {0.001,0.010,0.060,0.014,0.040,0.032,0.012,0.072}; 
  double dphitNucFSIErr[8] = {0.005,0.008,0.007,0.001,0.018,0.058,0.060,0.025}; 
  double datNucFSIErr[8]   = {0.014,0.025,0.017,0.014,0.024,0.004,0.005,0.005}; 

  TMatrixDSym *dptCombExtraSyst = new TMatrixDSym(0,nbinsAuto);
  TMatrixDSym *dphitCombExtraSyst = new TMatrixDSym(0,nbinsAuto);
  TMatrixDSym *datCombExtraSyst = new TMatrixDSym(0,nbinsAuto);

  //dptModelCovar->Print();
  //dptDetCovar->Print();
  //dptCombExtraSyst->Print();

  for(int r=0;r<nbinsAuto;r++){
    //cout << "On row " << r << endl;
    for(int c=0;c<nbinsAuto;c++){
      if(var=="dpt")   (*dptCombExtraSyst)[r][c]   = (*dptModelCovar)[r][c] + (*dptDetCovar)[r][c] + fluxNormErr*fluxNormErr + ntgterror*ntgterror;
      if(var=="dphit") (*dphitCombExtraSyst)[r][c] = (*dphitModelCovar)[r][c] + (*dphitDetCovar)[r][c] + fluxNormErr*fluxNormErr + ntgterror*ntgterror;
      if(var=="dat")   (*datCombExtraSyst)[r][c]   = (*datModelCovar)[r][c] + (*datDetCovar)[r][c] + fluxNormErr*fluxNormErr + ntgterror*ntgterror;
      if(r==c){
        if(var=="dpt")    (*dptCombExtraSyst)[r][c]   += dptNucFSIErr[r]*dptNucFSIErr[r];
        if(var=="dphit")  (*dphitCombExtraSyst)[r][c] += dphitNucFSIErr[r]*dphitNucFSIErr[r];
        if(var=="dat")    (*datCombExtraSyst)[r][c]   += datNucFSIErr[r]*datNucFSIErr[r];
      }
    }
  }

  //covarNevts->Print("all");

  for(int r=0;r<nbinsAuto;r++){
    for(int c=0;c<nbinsAuto;c++){
        // WHY WOULD I DO THISSSS!!!???!?
        //if(var=="dpt")   (*covarNevts)[r][c]= ((*covarNevts)[r][c]) + ( ((*dptCombExtraSyst)[r][c]) * ( (fluxNormErr*fitSpect->GetBinContent(r+1))*(fluxNormErr*fitSpect->GetBinContent(r+1)) ) );
        //if(var=="dphit") (*covarNevts)[r][c]= ((*covarNevts)[r][c]) + ( ((*dphitCombExtraSyst)[r][c]) * ( (fluxNormErr*fitSpect->GetBinContent(r+1))*(fluxNormErr*fitSpect->GetBinContent(r+1)) ) );
        //if(var=="dat")   (*covarNevts)[r][c]= ((*covarNevts)[r][c]) + ( ((*datCombExtraSyst)[r][c]) * ( (fluxNormErr*fitSpect->GetBinContent(r+1))*(flxNormErr*fitSpect->GetBinContent(r+1)) ) );
        if(var=="dpt")   (*covarNevts)[r][c]= ((*covarNevts)[r][c]) + ( ((*dptCombExtraSyst)[r][c]) * ( (fitSpect->GetBinContent(r+1))*(fitSpect->GetBinContent(c+1)) ) );
        if(var=="dphit") (*covarNevts)[r][c]= ((*covarNevts)[r][c]) + ( ((*dphitCombExtraSyst)[r][c]) * ( (fitSpect->GetBinContent(r+1))*(fitSpect->GetBinContent(c+1)) ) );
        if(var=="dat")   (*covarNevts)[r][c]= ((*covarNevts)[r][c]) + ( ((*datCombExtraSyst)[r][c]) * ( (fitSpect->GetBinContent(r+1))*(fitSpect->GetBinContent(c+1)) ) );
    }
  }

  TH1D* dif_xSecFit_allError = new TH1D(*dif_xSecFit);
  for(int r=0;r<nbinsAuto;r++){ 
    dif_xSecFit_allError->SetBinError(r+1, dif_xSecFit_allError->GetBinContent(r+1)*sqrt((*covarNevts)[r][r])/fitSpect->GetBinContent(r+1) );
  }


  /*

    TH1D* effFluxErr = (TH1D*) propErrFile->Get("effFluxErr");
    TH1D* dif_xSecFit_allError = new TH1D(*dif_xSecFit);
    double fluxNormErr=0.085;
    for(int r=0;r<nbinsAuto;r++){
      for(int c=0;c<nbinsAuto;c++){
        if(r!=c) (*covarNevts)[r][c]= ((*covarNevts)[r][c]) + ( (fluxNormErr*fitSpect->GetBinContent(r+1))*(fluxNormErr*fitSpect->GetBinContent(c+1)) );
        else{
          (*covarNevts)[r][r] = ((*covarNevts)[r][r]) + pow(effFluxErr->GetBinContent(r+1)*fitSpect->GetBinContent(r+1),2);
          dif_xSecFit_allError->SetBinError(r+1, dif_xSecFit_allError->GetBinContent(r+1)*sqrt((*covarNevts)[r][r])/fitSpect->GetBinContent(r+1) );
        }
      }
    }
  */
  /*
    TH1D* dif_xSecFit_allError = new TH1D(*dif_xSecFit);
    double newErrXsec;
    double newErrNevts;
    for(int i=1;i<=nbinsAuto;i++){
      newErrXsec = sqrt( pow(dif_xSecFit_allError->GetBinError(i),2) + pow(effFluxErr->GetBinContent(i)*dif_xSecFit_allError->GetBinContent(i),2) );
      newErrNevts = sqrt( pow(fitSpect->GetBinError(i),2) + pow(effFluxErr->GetBinContent(i)*fitSpect->GetBinContent(i),2) );
      dif_xSecFit_allError->SetBinError(i, newErrXsec);
      (*covarNevts)(i-1,i-1) = newErrNevts*newErrNevts;
    }
  */

  //covarNevts->Print("all");

  //Calculate Corrolation Matrix
  TMatrixD cormatrix(nbinsAuto,nbinsAuto);
  for(int r=0;r<nbinsAuto;r++){
    for(int c=0;c<nbinsAuto;c++){
      cormatrix[r][c]= (*covarNevts)[r][c]/sqrt((((*covarNevts)[r][r])*((*covarNevts))[c][c]));
      //cout << "covarNevts(" << r << ", " << c << ") is: " <<  (*covarNevts)[r][c] << endl;
      //cout << "covarNevts(" << r << ", " << r << ") is: " <<  (*covarNevts)[r][r] << endl;
      //cout << "covarNevts(" << c << ", " << c << ") is: " <<  (*covarNevts)[c][c] << endl;
      //cout << "    cormatrix(" << r << ", " << c << ") is: " <<  cormatrix[r][c] << endl;
    }
  }

  //Calculate XSec Covar Matrix
  TMatrixD covarXsec(nbinsAuto,nbinsAuto);
  for(int r=0;r<nbinsAuto;r++){
    for(int c=0;c<nbinsAuto;c++){
      covarXsec[r][c]= (cormatrix)[r][c]*(dif_xSecFit_allError->GetBinError(r+1))*(dif_xSecFit_allError->GetBinError(c+1)) ;
    }
  }


  //Calculate XSec Shape Only Covar Matrix
  TMatrixD covarXsec_shapeOnly(nbinsAuto,nbinsAuto);
  for(int r=0;r<nbinsAuto;r++){
    for(int c=0;c<nbinsAuto;c++){
      covarXsec_shapeOnly[r][c]= (cormatrix)[r][c]*(dif_shapeOnly_xSecFit->GetBinError(r+1))*(dif_shapeOnly_xSecFit->GetBinError(c+1)) ;
    }
  }

  double selfChi2 = calcChi2(dif_xSecFit_allError, dif_xSecFit_allError, covarXsec);
  double mcChi2 = calcChi2(dif_xSecFit_allError, dif_xSecMC, covarXsec);
  double fdChi2 = calcChi2(dif_xSecFit_allError, dif_xSecFD, covarXsec);

  cout << "Chi2 tests for self, MC, FD are " << selfChi2 << ", " << mcChi2 << " and " << fdChi2 << endl; 

  TH1D* chi2Hist = new TH1D("chi2Hist", "chi2Hist", 3, 0, 2);
  chi2Hist->GetXaxis()->SetBinLabel(1, "Self");
  chi2Hist->GetXaxis()->SetBinLabel(2, "Input MC");
  chi2Hist->GetXaxis()->SetBinLabel(3, "Fake Data");

  chi2Hist->SetBinContent(1, selfChi2);
  chi2Hist->SetBinContent(2, mcChi2);
  chi2Hist->SetBinContent(3, fdChi2);

  TH1D* chi2Hist_realdata = new TH1D("chi2Hist_realdata", "chi2Hist_realdata", 5, 0, 5);
  TH1D* chi2Hist_realdata_dat = new TH1D("chi2Hist_realdata_dat", "chi2Hist_realdata_dat", 5, 0, 5);
  TH1D* chi2Hist_realdata_dphit = new TH1D("chi2Hist_realdata_dphit", "chi2Hist_realdata_dphit", 5, 0, 5);

  TH1D* chi2Hist_realdata_shapeOnly = new TH1D("chi2Hist_realdata_shapeOnly", "chi2Hist_realdata_shapeOnly", 4, 0, 4);
  TH1D* chi2Hist_realdata_dat_shapeOnly = new TH1D("chi2Hist_realdata_dat_shapeOnly", "chi2Hist_realdata_dat_shapeOnly", 4, 0, 4);
  TH1D* chi2Hist_realdata_dphit_shapeOnly = new TH1D("chi2Hist_realdata_dphit_shapeOnly", "chi2Hist_realdata_dphit_shapeOnly", 4, 0, 4);

  double neutChi2 = 0, neutChi2_dat = 0, neutChi2_dphit = 0;
  double neutRWChi2 = 0, neutRWChi2_dat = 0, neutRWChi2_dphit = 0;
  double genieChi2 = 0, genieChi2_dat = 0, genieChi2_dphit = 0;
  double nuWroChi2 = 0, nuWroChi2_dat = 0, nuWroChi2_dphit = 0;
  double nuWroFixChi2 = 0, nuWroFixChi2_dat = 0, nuWroFixChi2_dphit = 0;
  double neutNo2p2hChi2 = 0, neutNo2p2hChi2_dat = 0, neutNo2p2hChi2_dphit = 0;
  double neutBaskChi2 = 0, neutBaskChi2_dat = 0, neutBaskChi2_dphit = 0;

  double neutChi2ShapeOnly = 0, neutChi2ShapeOnly_dat = 0, neutChi2ShapeOnly_dphit = 0;
  double neutRWChi2ShapeOnly = 0, neutRWChi2ShapeOnly_dat = 0, neutRWChi2ShapeOnly_dphit = 0;
  double genieChi2ShapeOnly = 0, genieChi2ShapeOnly_dat = 0, genieChi2ShapeOnly_dphit = 0;
  double nuWroChi2ShapeOnly = 0, nuWroChi2ShapeOnly_dat = 0, nuWroChi2ShapeOnly_dphit = 0;
  double nuWroFixChi2ShapeOnly = 0, nuWroFixChi2ShapeOnly_dat = 0, nuWroFixChi2ShapeOnly_dphit = 0;
  double neutNo2p2hChi2ShapeOnly = 0, neutNo2p2hChi2ShapeOnly_dat = 0, neutNo2p2hChi2ShapeOnly_dphit = 0;

  if((dif_xSecFit_allError->GetNbinsX() == dif_xSecNeut->GetNbinsX()) && (var=="dpt") ){
    cout << "calc dpt chi2" << endl;
    neutChi2 = calcChi2(dif_xSecFit_allError, dif_xSecNeut, covarXsec);
    genieChi2 = calcChi2(dif_xSecFit_allError, dif_xSecGenie, covarXsec);
    nuWroChi2 = calcChi2(dif_xSecFit_allError, dif_xSecNuwro, covarXsec);
    neutNo2p2hChi2 = calcChi2(dif_xSecFit_allError, dif_xSecNeutNo2p2h, covarXsec);
    neutBaskChi2 = calcChi2(dif_xSecFit_allError, dif_xSecNeutBask, covarXsec);

    neutChi2ShapeOnly = calcChi2(dif_xSecFit_allError, dif_shapeOnly_xSecNeut, covarXsec_shapeOnly);
    genieChi2ShapeOnly = calcChi2(dif_xSecFit_allError, dif_shapeOnly_xSecGenie, covarXsec_shapeOnly);
    nuWroChi2ShapeOnly = calcChi2(dif_xSecFit_allError, dif_shapeOnly_xSecNuwro, covarXsec_shapeOnly);
    neutNo2p2hChi2ShapeOnly = calcChi2(dif_xSecFit_allError, dif_shapeOnly_xSecNeutNo2p2h, covarXsec_shapeOnly);

    chi2Hist_realdata->SetBinContent(1, neutChi2);
    chi2Hist_realdata->SetBinContent(2, genieChi2);
    chi2Hist_realdata->SetBinContent(3, nuWroChi2);
    chi2Hist_realdata->SetBinContent(4, neutNo2p2hChi2);
    chi2Hist_realdata->SetBinContent(5, neutBaskChi2);
    chi2Hist_realdata_shapeOnly->SetBinContent(1, neutChi2ShapeOnly);
    chi2Hist_realdata_shapeOnly->SetBinContent(2, genieChi2ShapeOnly);
    chi2Hist_realdata_shapeOnly->SetBinContent(3, nuWroChi2ShapeOnly);
    chi2Hist_realdata_shapeOnly->SetBinContent(4, neutNo2p2hChi2ShapeOnly);

    cout << endl << "Real data chi2 results: " << endl;
    cout << "NEUT Prior:  " << neutChi2 << endl;
    cout << "Neut Nominal:  " << neutRWChi2 << endl;
    cout << "GENIE:  " << genieChi2 << endl;
    cout << "NuWro:  " << nuWroChi2 << endl;
    cout << "NuWro Scale Fix:  " << nuWroFixChi2 << endl;
    cout << "Neut No 2p2h:  " << neutNo2p2hChi2 << endl;
    cout << "Neut 6D:  " << neutBaskChi2 << endl << endl;

    chi2Hist_realdata->GetXaxis()->SetBinLabel(1, "NEUT 6B");
    chi2Hist_realdata->GetXaxis()->SetBinLabel(2, "GENIE");
    chi2Hist_realdata->GetXaxis()->SetBinLabel(3, "NuWro (prod)");
    chi2Hist_realdata->GetXaxis()->SetBinLabel(4, "NEUT No 2p2h");
    chi2Hist_realdata->GetXaxis()->SetBinLabel(5, "NEUT 6D");
    chi2Hist_realdata_shapeOnly->GetXaxis()->SetBinLabel(1, "NEUT Prior");
    chi2Hist_realdata_shapeOnly->GetXaxis()->SetBinLabel(2, "GENIE");
    chi2Hist_realdata_shapeOnly->GetXaxis()->SetBinLabel(3, "NuWro (prod)");
    chi2Hist_realdata_shapeOnly->GetXaxis()->SetBinLabel(4, "NEUT No 2p2h");
  }

  if((dif_xSecFit_allError->GetNbinsX() == datGenieXSecHist->GetNbinsX()) && (var=="dat")){
    cout << "calc dat chi2" << endl;
    neutChi2_dat = calcChi2(dif_xSecFit_allError, datNEUTPriorXSecHist, covarXsec);
    genieChi2_dat = calcChi2(dif_xSecFit_allError, datGenieXSecHist, covarXsec);
    nuWroChi2_dat = calcChi2(dif_xSecFit_allError, datNuWroXSecHist, covarXsec);
    neutNo2p2hChi2_dat = calcChi2(dif_xSecFit_allError, datNEUTNo2p2hXSecHist, covarXsec);
    neutBaskChi2_dat = calcChi2(dif_xSecFit_allError, datNEUTBaskHist, covarXsec);

    neutChi2ShapeOnly_dat = calcChi2(dif_xSecFit_allError, datNEUTPriorXSecHist_shapeOnly, covarXsec_shapeOnly);
    genieChi2ShapeOnly_dat = calcChi2(dif_xSecFit_allError, datGenieXSecHist_shapeOnly, covarXsec_shapeOnly);
    nuWroChi2ShapeOnly_dat = calcChi2(dif_xSecFit_allError, datGenieXSecHist_shapeOnly, covarXsec_shapeOnly);
    neutNo2p2hChi2ShapeOnly_dat = calcChi2(dif_xSecFit_allError, datNEUTNo2p2hXSecHist_shapeOnly, covarXsec_shapeOnly);

    chi2Hist_realdata_dat->SetBinContent(1, neutChi2_dat);
    chi2Hist_realdata_dat->SetBinContent(2, genieChi2_dat);
    chi2Hist_realdata_dat->SetBinContent(3, nuWroChi2_dat);
    chi2Hist_realdata_dat->SetBinContent(4, neutNo2p2hChi2_dat);
    chi2Hist_realdata_dat->SetBinContent(5, neutBaskChi2_dat);
    chi2Hist_realdata_dat_shapeOnly->SetBinContent(1, neutChi2ShapeOnly_dat);
    chi2Hist_realdata_dat_shapeOnly->SetBinContent(2, genieChi2ShapeOnly_dat);
    chi2Hist_realdata_dat_shapeOnly->SetBinContent(3, nuWroChi2ShapeOnly_dat);
    chi2Hist_realdata_dat_shapeOnly->SetBinContent(4, neutNo2p2hChi2ShapeOnly_dat);

    chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(1, "NEUT 6B");
    chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(2, "GENIE");
    chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(3, "NuWro (prod)");
    chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(4, "NEUT No 2p2h");
    chi2Hist_realdata_dat->GetXaxis()->SetBinLabel(5, "NEUT 6D");
    chi2Hist_realdata_dat_shapeOnly->GetXaxis()->SetBinLabel(1, "NEUT Prior");
    chi2Hist_realdata_dat_shapeOnly->GetXaxis()->SetBinLabel(2, "GENIE");
    chi2Hist_realdata_dat_shapeOnly->GetXaxis()->SetBinLabel(3, "NuWro (prod)");
    chi2Hist_realdata_dat_shapeOnly->GetXaxis()->SetBinLabel(4, "NEUT No 2p2h");
  }

  if((dif_xSecFit_allError->GetNbinsX() == dphitGenieXSecHist->GetNbinsX()) && (var=="dphit")){
    cout << "calc dphit chi2" << endl;
    neutChi2_dphit = calcChi2(dif_xSecFit_allError, dphitNEUTPriorXSecHist, covarXsec);
    genieChi2_dphit = calcChi2(dif_xSecFit_allError, dphitGenieXSecHist, covarXsec);
    nuWroChi2_dphit = calcChi2(dif_xSecFit_allError, dphitNuWroXSecHist, covarXsec);
    neutNo2p2hChi2_dphit = calcChi2(dif_xSecFit_allError, dphitNEUTNo2p2hXSecHist, covarXsec);
    neutBaskChi2_dphit = calcChi2(dif_xSecFit_allError, dphitNEUTBaskHist, covarXsec);

    neutChi2ShapeOnly_dphit = calcChi2(dif_shapeOnly_xSecFit, dphitNEUTPriorXSecHist_shapeOnly, covarXsec_shapeOnly);
    genieChi2ShapeOnly_dphit = calcChi2(dif_shapeOnly_xSecFit, dphitGenieXSecHist_shapeOnly, covarXsec_shapeOnly);
    nuWroChi2ShapeOnly_dphit = calcChi2(dif_shapeOnly_xSecFit, dphitNuWroXSecHist_shapeOnly, covarXsec_shapeOnly);
    neutNo2p2hChi2ShapeOnly_dphit = calcChi2(dif_shapeOnly_xSecFit, dphitNEUTNo2p2hXSecHist_shapeOnly, covarXsec_shapeOnly);

    chi2Hist_realdata_dphit->SetBinContent(1, neutChi2_dphit);
    chi2Hist_realdata_dphit->SetBinContent(2, genieChi2_dphit);
    chi2Hist_realdata_dphit->SetBinContent(3, nuWroChi2_dphit);
    chi2Hist_realdata_dphit->SetBinContent(4, neutNo2p2hChi2_dphit);
    chi2Hist_realdata_dphit->SetBinContent(5, neutBaskChi2_dphit);
    chi2Hist_realdata_dphit_shapeOnly->SetBinContent(1, neutChi2ShapeOnly_dphit);
    chi2Hist_realdata_dphit_shapeOnly->SetBinContent(2, genieChi2ShapeOnly_dphit);
    chi2Hist_realdata_dphit_shapeOnly->SetBinContent(3, nuWroChi2ShapeOnly_dphit);
    chi2Hist_realdata_dphit_shapeOnly->SetBinContent(4, neutNo2p2hChi2ShapeOnly_dphit);

    chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(1, "NEUT 6B");
    chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(2, "GENIE");
    chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(3, "NuWro (prod)");
    chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(4, "NEUT No 2p2h");
    chi2Hist_realdata_dphit->GetXaxis()->SetBinLabel(5, "NEUT 6D");
    chi2Hist_realdata_dphit_shapeOnly->GetXaxis()->SetBinLabel(1, "NEUT Prior");
    chi2Hist_realdata_dphit_shapeOnly->GetXaxis()->SetBinLabel(2, "GENIE");
    chi2Hist_realdata_dphit_shapeOnly->GetXaxis()->SetBinLabel(3, "NuWro (prod)");
    chi2Hist_realdata_dphit_shapeOnly->GetXaxis()->SetBinLabel(4, "NEUT No 2p2h");
  }



  outputFile->cd();

  chi2Hist->SetYTitle("#chi^{2}");
  chi2Hist->Write();
  chi2Hist_realdata->SetYTitle("#chi^{2}");
  chi2Hist_realdata_dat->SetYTitle("#chi^{2}");
  chi2Hist_realdata_dphit->SetYTitle("#chi^{2}");
  chi2Hist_realdata->Write();
  chi2Hist_realdata_dat->Write();
  chi2Hist_realdata_dphit->Write();

  TCanvas* canv_chi2 = new TCanvas(); 

  if(var=="dpt"){
    chi2Hist_realdata->Draw();
    canv_chi2->SaveAs("chi2Comp.png");
  }
  if(var=="dat"){
    chi2Hist_realdata_dat->Draw();
    canv_chi2->SaveAs("chi2Comp.png");
  }
  if(var=="dphit"){
    chi2Hist_realdata_dphit->Draw();
    canv_chi2->SaveAs("chi2Comp.png");
  }

  chi2Hist_realdata_shapeOnly->SetYTitle("#chi^{2}");
  chi2Hist_realdata_dat_shapeOnly->SetYTitle("#chi^{2}");
  chi2Hist_realdata_dphit_shapeOnly->SetYTitle("#chi^{2}");
  chi2Hist_realdata_shapeOnly->Write();
  chi2Hist_realdata_dat_shapeOnly->Write();
  chi2Hist_realdata_dphit_shapeOnly->Write();

  effHisto->Write();
  effHisto_fd->Write();

  dif_xSecFit->SetLineColor(kRed);
  dif_xSecFit->SetLineWidth(2);

  dif_xSecFit_allError->SetLineColor(kRed);
  dif_xSecFit_allError->SetLineWidth(2);

  dif_xSecMC->SetLineColor(kBlue);
  dif_xSecMC->SetLineWidth(2);
  if(var=="dpt"){
    dif_xSecMC->SetXTitle("#deltap_{T}(GeV/c)");
    dif_xSecMC->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (Nucleon^{-1} cm^{2} GeV^{-1})");
    dif_xSecMC->GetYaxis()->SetRangeUser(0,12.5E-39);
  }
  else if(var=="dphit"){
    dif_xSecMC->SetXTitle("#delta#phi_{T}(rads)");
    dif_xSecMC->SetYTitle("#frac{d#sigma}{d#delta#phi_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");
    dif_xSecMC->GetYaxis()->SetRangeUser(0,6.5E-39);
  }
  else if(var=="dat"){
    dif_xSecMC->SetXTitle("#delta#alpha_{T}(rads)");
    dif_xSecMC->SetYTitle("#frac{d#sigma}{d#delta#alpha_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");
    dif_xSecMC->GetYaxis()->SetRangeUser(0,1.5E-39);
  }

  if(var=="dpt"){
    dif_xSecFit_allError->SetXTitle("#deltap_{T}(GeV/c)");
    dif_xSecFit_allError->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (Nucleon^{-1} cm^{2} GeV^{-1})");
  }
  else if(var=="dphit"){
    dif_xSecFit_allError->SetXTitle("#delta#phi_{T}(rads)");
    dif_xSecFit_allError->SetYTitle("#frac{d#sigma}{d#delta#phi_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");
  }
  else if(var=="dat"){
    dif_xSecFit_allError->SetXTitle("#delta#alpha_{T}(rads)");
    dif_xSecFit_allError->SetYTitle("#frac{d#sigma}{d#delta#alpha_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");
  }

  if(var=="dpt"){
    dif_xSecFD->SetXTitle("#deltap_{T}(GeV/c)");
    dif_xSecFD->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (Nucleon^{-1} cm^{2} GeV^{-1})");
  }
  else if(var=="dphit"){
    dif_xSecFD->SetXTitle("#delta#phi_{T}(rads)");
    dif_xSecFD->SetYTitle("#frac{d#sigma}{d#delta#phi_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");
  }
  else if(var=="dat"){
    dif_xSecFD->SetXTitle("#delta#alpha_{T}(rads)");
    dif_xSecFD->SetYTitle("#frac{d#sigma}{d#delta#alpha_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");
  }



  dif_xSecFD->SetLineColor(kGreen+2);
  dif_xSecFD->SetLineWidth(2);

  dif_xSecFit_allError->Write("dif_xSecFit_allError");
  dif_xSecFit->Write();
  dif_xSecMC->Write();
  dif_xSecFD->Write();

  if(true){
    dif_xSecNeut->SetLineColor(kRed);
    dif_xSecNeut->SetLineWidth(2);
    datNEUTPriorXSecHist->SetLineColor(kRed);
    datNEUTPriorXSecHist->SetLineWidth(2);
    dphitNEUTPriorXSecHist->SetLineColor(kRed);
    dphitNEUTPriorXSecHist->SetLineWidth(2);

    dif_xSecNiwgNeut->SetLineColor(kRed);
    dif_xSecNiwgNeut->SetLineWidth(2);
    dif_xSecNiwgNeut->SetLineStyle(2);


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

    dif_xSecNeutBask->SetLineColor(kRed+2);
    dif_xSecNeutBask->SetLineWidth(2);
    dif_xSecNeutBask->SetLineStyle(2);
    datNEUTBaskHist->SetLineColor(kRed+2);
    datNEUTBaskHist->SetLineWidth(2);
    datNEUTBaskHist->SetLineStyle(2);
    dphitNEUTBaskHist->SetLineColor(kRed+2);
    dphitNEUTBaskHist->SetLineWidth(2);
    dphitNEUTBaskHist->SetLineStyle(2);

    //SHAPE ONLY:

    dif_shapeOnly_xSecNeut->SetLineColor(kRed);
    dif_shapeOnly_xSecNeut->SetLineWidth(2);
    datNEUTPriorXSecHist_shapeOnly->SetLineColor(kRed);
    datNEUTPriorXSecHist_shapeOnly->SetLineWidth(2);
    dphitNEUTPriorXSecHist_shapeOnly->SetLineColor(kRed);
    dphitNEUTPriorXSecHist_shapeOnly->SetLineWidth(2);



    dif_shapeOnly_xSecNuwro->SetLineColor(kBlue);
    dif_shapeOnly_xSecNuwro->SetLineWidth(2);
    dif_shapeOnly_xSecNuwro->SetXTitle("#deltap_{T}(GeV/c)");
    dif_shapeOnly_xSecNuwro->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (Nucleon^{-1} cm^{2} GeV^{-1})");
    datNuWroXSecHist_shapeOnly->SetLineColor(kBlue);
    datNuWroXSecHist_shapeOnly->SetLineWidth(2);
    datNuWroXSecHist_shapeOnly->SetXTitle("#delta#alpha_{T}(rads)");
    datNuWroXSecHist_shapeOnly->SetYTitle("#frac{d#sigma}{d#delta#alpha_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");
    dphitNuWroXSecHist_shapeOnly->SetLineColor(kBlue);
    dphitNuWroXSecHist_shapeOnly->SetLineWidth(2);
    dphitNuWroXSecHist_shapeOnly->SetXTitle("#delta#phi_{T}(rads)");
    dphitNuWroXSecHist_shapeOnly->SetYTitle("#frac{d#sigma}{d#delta#phi_{T}} (Nucleon^{-1} cm^{2} rads^{-1})");


    dif_shapeOnly_xSecGenie->SetLineColor(kGreen-2);
    dif_shapeOnly_xSecGenie->SetLineWidth(2);
    datGenieXSecHist_shapeOnly->SetLineColor(kGreen-2);
    datGenieXSecHist_shapeOnly->SetLineWidth(2);
    dphitGenieXSecHist_shapeOnly->SetLineColor(kGreen-2);
    dphitGenieXSecHist_shapeOnly->SetLineWidth(2);

    dif_shapeOnly_xSecNeutNo2p2h->SetLineColor(kRed+2);
    dif_shapeOnly_xSecNeutNo2p2h->SetLineWidth(2);
    dif_shapeOnly_xSecNeutNo2p2h->SetLineStyle(3);
    datNEUTNo2p2hXSecHist_shapeOnly->SetLineColor(kRed+2);
    datNEUTNo2p2hXSecHist_shapeOnly->SetLineWidth(2);
    datNEUTNo2p2hXSecHist_shapeOnly->SetLineStyle(3);
    dphitNEUTNo2p2hXSecHist_shapeOnly->SetLineColor(kRed+2);
    dphitNEUTNo2p2hXSecHist_shapeOnly->SetLineWidth(2);
    dphitNEUTNo2p2hXSecHist_shapeOnly->SetLineStyle(3);

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

  TLegend* leg_xsecCanv = new TLegend(0.4,0.55,0.85,0.85);
  leg_xsecCanv->AddEntry(dif_xSecMC,Form("Input Simulation, #chi^{2}=%.2f", mcChi2),"l");
  leg_xsecCanv->AddEntry(dif_xSecFD,Form("Fake Data, #chi^{2}=%.2f", fdChi2),"l");
  leg_xsecCanv->AddEntry(dif_xSecFit_allError,"Result","lep");
  leg_xsecCanv->SetFillColor(kWhite);
  leg_xsecCanv->SetFillStyle(0);

  TLegend* leg_xsecCanv_asi = new TLegend(0.4,0.55,0.85,0.85);
  leg_xsecCanv_asi->AddEntry(dif_xSecFD,"Fake Data = Input Simulation","l");
  leg_xsecCanv_asi->AddEntry(dif_xSecFit_allError,"Result","lep");
  leg_xsecCanv_asi->SetFillColor(kWhite);
  leg_xsecCanv_asi->SetFillStyle(0);

  if(var=="dat"){
    leg_xsecCanv->SetX1(0.2);
    leg_xsecCanv->SetX2(0.65);
    leg_xsecCanv_asi->SetX1(0.2);
    leg_xsecCanv_asi->SetX2(0.65);
  }


  if(var=="dpt") dif_xSecMC->GetYaxis()->SetRangeUser(0,12.5E-39);
  if(var=="dphit") dif_xSecMC->GetYaxis()->SetRangeUser(0,6.5E-39);
  if(var=="dat") dif_xSecMC->GetYaxis()->SetRangeUser(0,1.5E-39);
  if(var=="dpt") dif_xSecFD->GetYaxis()->SetRangeUser(0,12.5E-39);
  if(var=="dphit") dif_xSecFD->GetYaxis()->SetRangeUser(0,6.5E-39);
  if(var=="dat") dif_xSecFD->GetYaxis()->SetRangeUser(0,1.5E-39);
  if(var=="dpt") dif_xSecFit_allError->GetYaxis()->SetRangeUser(0,12.5E-39);
  if(var=="dphit") dif_xSecFit_allError->GetYaxis()->SetRangeUser(0,6.5E-39);
  if(var=="dat") dif_xSecFit_allError->GetYaxis()->SetRangeUser(0,1.5E-39);

  TCanvas* canvxs = new TCanvas("xsecCanv","xsecCanv");
  dif_xSecMC->Draw();
  dif_xSecFD->Draw("same");
  dif_xSecFit->Draw("same");
  leg_xsecCanv->Draw("same");
  canvxs->Write();


  TCanvas* canvxs_allError = new TCanvas("xsecCanv_allError","xsecCanv_allError");
  dif_xSecMC->Draw();
  dif_xSecFD->Draw("same");
  dif_xSecFit_allError->Draw("same");
  leg_xsecCanv->Draw("same");
  canvxs_allError->Write();
  canvxs_allError->SaveAs("xsecComp.png");

  TCanvas* canvxs_allError_2p2hComp = new TCanvas("xsecCanv_allError_2p2hComp","xsecCanv_allError_2p2hComp");
  dif_xSecMC->Draw();
  dif_xSecFD->Draw("same");
  dif_xSecFit_allError->Draw("same");
  leg_xsecCanv->Draw("same");
  canvxs_allError->Write();
  canvxs_allError->SaveAs("xsecComp.png");

  TCanvas* canvxs_allError_asi = new TCanvas("xsecCanv_allError_asi","xsecCanv_allError_asi");
  dif_xSecMC->Draw();
  dif_xSecFD->Draw("same");
  dif_xSecFit_allError->Draw("same");
  leg_xsecCanv_asi->Draw("same");
  canvxs_allError_asi->Write();
  canvxs_allError_asi->SaveAs("xsecComp_asi.png");

  //*********************************
  //** 2p2h comp canv:
  //*********************************

  TLegend* leg_xsecCanv_2p2hComp = new TLegend(0.4,0.55,0.85,0.85);
  leg_xsecCanv_2p2hComp->AddEntry(dif_xSecMC,Form("Input Simulation, #chi^{2}=%.2f", mcChi2),"l");
  leg_xsecCanv_2p2hComp->AddEntry(dif_xSecNeutBask,Form("NEUT 6D, #chi^{2}=%.2f", neutBaskChi2),"l");
  leg_xsecCanv_2p2hComp->AddEntry(dif_xSecNeutNo2p2h,Form("NEUT 6D w/o 2p2h, #chi^{2}=%.2f", neutNo2p2hChi2),"l");
  leg_xsecCanv_2p2hComp->AddEntry(dif_xSecFit_allError,"Result","lep");
  leg_xsecCanv_2p2hComp->SetFillColor(kWhite);
  leg_xsecCanv_2p2hComp->SetFillStyle(0);

  TLegend* leg_xsecCanv_2p2hComp_dat = new TLegend(0.2,0.55,0.65,0.85);
  leg_xsecCanv_2p2hComp_dat->AddEntry(dif_xSecMC,Form("Input Simulation, #chi^{2}=%.2f", mcChi2),"l");
  leg_xsecCanv_2p2hComp_dat->AddEntry(datNEUTBaskHist,Form("NEUT 6D, #chi^{2}=%.2f", neutBaskChi2_dat),"l");
  leg_xsecCanv_2p2hComp_dat->AddEntry(datNEUTNo2p2hXSecHist,Form("NEUT 6D w/o 2p2h, #chi^{2}=%.2f", neutNo2p2hChi2_dat),"l");
  leg_xsecCanv_2p2hComp_dat->AddEntry(dif_xSecFit_allError,"Result","lep");
  leg_xsecCanv_2p2hComp_dat->SetFillColor(kWhite);
  leg_xsecCanv_2p2hComp_dat->SetFillStyle(0);

  TLegend* leg_xsecCanv_2p2hComp_dphit = new TLegend(0.4,0.55,0.85,0.85);
  leg_xsecCanv_2p2hComp_dphit->AddEntry(dif_xSecMC,Form("Input Simulation, #chi^{2}=%.2f", mcChi2),"l");
  leg_xsecCanv_2p2hComp_dphit->AddEntry(dphitNEUTBaskHist,Form("NEUT 6D, #chi^{2}=%.2f", neutBaskChi2_dphit),"l");
  leg_xsecCanv_2p2hComp_dphit->AddEntry(dphitNEUTNo2p2hXSecHist,Form("NEUT 6D w/o 2p2h, #chi^{2}=%.2f", neutNo2p2hChi2_dphit),"l");
  leg_xsecCanv_2p2hComp_dphit->AddEntry(dif_xSecFit_allError,"Result","lep");
  leg_xsecCanv_2p2hComp_dphit->SetFillColor(kWhite);
  leg_xsecCanv_2p2hComp_dphit->SetFillStyle(0);

  TCanvas* canvxs_2p2hComp = new TCanvas("xsecCanv_2p2hComp","xsecCanv_2p2hComp");
  dif_xSecMC->Draw();
  dif_xSecNeutBask->Draw("same");
  dif_xSecNeutNo2p2h->Draw("same");
  dif_xSecFit_allError->Draw("same");
  leg_xsecCanv_2p2hComp->Draw("same");
  canvxs_2p2hComp->Write();
  if(var=="dpt") canvxs_2p2hComp->SaveAs("xsec2p2hComp.png");

  TCanvas* canvxs_2p2hComp_dphit = new TCanvas("xsecCanv_2p2hComp_dphit","xsecCanv_2p2hComp_dphit");
  dif_xSecMC->Draw();
  dphitNEUTBaskHist->Draw("same");
  dphitNEUTNo2p2hXSecHist->Draw("same");
  dif_xSecFit_allError->Draw("same");
  leg_xsecCanv_2p2hComp_dphit->Draw("same");
  canvxs_2p2hComp_dphit->Write();
  if(var=="dphit") canvxs_2p2hComp_dphit->SaveAs("xsec2p2hComp.png");

  TCanvas* canvxs_2p2hComp_dat = new TCanvas("xsecCanv_2p2hComp_dat","xsecCanv_2p2hComp_dat");
  dif_xSecMC->Draw();
  datNEUTBaskHist->Draw("same");
  datNEUTNo2p2hXSecHist->Draw("same");
  dif_xSecFit_allError->Draw("same");
  leg_xsecCanv_2p2hComp_dat->Draw("same");
  canvxs_2p2hComp_dat->Write();
  if(var=="dat") canvxs_2p2hComp_dat->SaveAs("xsec2p2hComp.png");


  if(true){

    //*********************************
    //** Gen comp canv:
    //*********************************

    TLegend* leg_xsecGenCompCanv = new TLegend(0.3,0.55,0.8,0.85);
    leg_xsecGenCompCanv->AddEntry(dif_xSecNeut,Form("Input Simulation (NEUT 6B), #chi^{2}=%.2f", mcChi2),"l");
    leg_xsecGenCompCanv->AddEntry(dif_xSecNuwro,Form("NuWro, #chi^{2}=%.2f", nuWroChi2),"l");
    leg_xsecGenCompCanv->AddEntry(dif_xSecNeutBask,Form("NEUT 6D, #chi^{2}=%.2f", neutBaskChi2),"l");
    leg_xsecGenCompCanv->AddEntry(dif_xSecNeutNo2p2h,Form("NEUT No 2p2h, #chi^{2}=%.2f", neutNo2p2hChi2),"l");
    leg_xsecGenCompCanv->AddEntry(dif_xSecGenie,Form("GENIE, #chi^{2}=%.2f", genieChi2),"l");
    leg_xsecGenCompCanv->AddEntry(dif_xSecFit_allError,"Extracted Cross Section","lep");
    leg_xsecGenCompCanv->SetFillColor(kWhite);
    leg_xsecGenCompCanv->SetFillStyle(0);

    TLegend* leg_xsecGenCompCanv_dat = new TLegend(0.2,0.65,0.75,0.85);
    leg_xsecGenCompCanv_dat->AddEntry(datNEUTPriorXSecHist,Form("Input Simulation (NEUT 6B), #chi^{2}=%.2f", mcChi2),"l");
    leg_xsecGenCompCanv_dat->AddEntry(datNuWroXSecHist,Form("NuWro, #chi^{2}=%.2f", nuWroChi2_dat),"l");
    leg_xsecGenCompCanv_dat->AddEntry(datNEUTBaskHist,Form("NEUT 6D, #chi^{2}=%.2f", neutBaskChi2_dat),"l");
    leg_xsecGenCompCanv_dat->AddEntry(datNEUTNo2p2hXSecHist,Form("NEUT No 2p2h, #chi^{2}=%.2f", neutNo2p2hChi2_dat),"l");
    leg_xsecGenCompCanv_dat->AddEntry(datGenieXSecHist,Form("GENIE, #chi^{2}=%.2f", genieChi2_dat),"l");
    leg_xsecGenCompCanv_dat->AddEntry(dif_xSecFit_allError,"Extracted Cross Section","lep");
    leg_xsecGenCompCanv_dat->SetFillColor(kWhite);
    leg_xsecGenCompCanv_dat->SetFillStyle(0);

    TLegend* leg_xsecGenCompCanv_dphit = new TLegend(0.3,0.55,0.8,0.85);
    leg_xsecGenCompCanv_dphit->AddEntry(dphitNEUTPriorXSecHist,Form("Input Simulation (NEUT 6B), #chi^{2}=%.2f", mcChi2),"l");
    leg_xsecGenCompCanv_dphit->AddEntry(dphitNuWroXSecHist,Form("NuWro, #chi^{2}=%.2f", nuWroChi2_dphit),"l");
    leg_xsecGenCompCanv_dphit->AddEntry(dphitNEUTBaskHist,Form("NEUT 6D, #chi^{2}=%.2f", neutBaskChi2_dphit),"l");
    leg_xsecGenCompCanv_dphit->AddEntry(dphitNEUTNo2p2hXSecHist,Form("NEUT No 2p2h, #chi^{2}=%.2f", neutNo2p2hChi2_dphit),"l");
    leg_xsecGenCompCanv_dphit->AddEntry(dphitGenieXSecHist,Form("GENIE, #chi^{2}=%.2f", genieChi2_dphit),"l");
    leg_xsecGenCompCanv_dphit->AddEntry(dif_xSecFit_allError,"Extracted Cross Section","lep");
    leg_xsecGenCompCanv_dphit->SetFillColor(kWhite);
    leg_xsecGenCompCanv_dphit->SetFillStyle(0);

    TCanvas* canvxs_genComp = new TCanvas("xsecCanv_genComp","xsecCanv_genComp");
    dif_xSecNuwro->GetYaxis()->SetRangeUser(0, 12.5E-39);
    dif_xSecNuwro->Draw();
    dif_xSecNeut->Draw("same");
    dif_xSecNeutBask->Draw("same");
    dif_xSecNeutNo2p2h->Draw("same");
    dif_xSecGenie->Draw("same");
    dif_xSecFit_allError->Draw("same");
    leg_xsecGenCompCanv->Draw("same");
    canvxs_genComp->Write();
    if(var=="dpt") canvxs_genComp->SaveAs("xsecGenComp.png");


    TCanvas* canvxs_genComp_dat = new TCanvas("xsecCanv_genComp_dat","xsecCanv_genComp_dat");
    datNuWroXSecHist->GetYaxis()->SetRangeUser(0, 1.5E-39);
    datNuWroXSecHist->Draw("HIST");
    datNEUTPriorXSecHist->Draw("sameHIST");
    datNEUTBaskHist->Draw("sameHIST");
    datNEUTNo2p2hXSecHist->Draw("sameHIST");
    datGenieXSecHist->Draw("sameHIST");
    dif_xSecFit_allError->Draw("same");
    leg_xsecGenCompCanv_dat->Draw("same");
    canvxs_genComp_dat->Write();
    if(var=="dat") canvxs_genComp_dat->SaveAs("xsecGenComp.png");

    TCanvas* canvxs_genComp_dphit = new TCanvas("xsecCanv_genComp_dphit","xsecCanv_genComp_dphit");
    dphitNuWroXSecHist->GetYaxis()->SetRangeUser(0, 6.5E-39);
    dphitNuWroXSecHist->Draw("HIST");
    dphitNEUTPriorXSecHist->Draw("sameHIST");
    dphitNEUTBaskHist->Draw("sameHIST");
    dphitNEUTNo2p2hXSecHist->Draw("sameHIST");
    dphitGenieXSecHist->Draw("sameHIST");
    dif_xSecFit_allError->Draw("same");
    leg_xsecGenCompCanv_dphit->Draw("same");
    canvxs_genComp_dphit->Write();
    if(var=="dphit") canvxs_genComp_dphit->SaveAs("xsecGenComp.png");

    //*********************************
    //** No result canv:
    //*********************************

    TLegend* leg_xsecGenOnlyCanv = new TLegend(0.4,0.55,0.8,0.85);
    leg_xsecGenOnlyCanv->AddEntry(dif_xSecNeut,"Input Simulation (NEUT 6B)","l");
    leg_xsecGenOnlyCanv->AddEntry(dif_xSecNuwro,"NuWro","l");
    leg_xsecGenOnlyCanv->AddEntry(dif_xSecNeutBask,"NEUT 6D","l");
    leg_xsecGenOnlyCanv->AddEntry(dif_xSecNeutNo2p2h,"NEUT No 2p2h","l");
    leg_xsecGenOnlyCanv->AddEntry(dif_xSecGenie,"GENIE","l");
    leg_xsecGenOnlyCanv->SetFillColor(kWhite);
    leg_xsecGenOnlyCanv->SetFillStyle(0);

    TLegend* leg_xsecGenOnlyCanv_dat = new TLegend(0.2,0.65,0.9,0.85);
    leg_xsecGenOnlyCanv_dat->AddEntry(datNEUTPriorXSecHist,"Input Simulation (NEUT 6B)","l");
    leg_xsecGenOnlyCanv_dat->AddEntry(dif_xSecNuwro,"NuWro","l");
    leg_xsecGenOnlyCanv_dat->AddEntry(dif_xSecNeutBask,"NEUT 6D","l");
    leg_xsecGenOnlyCanv_dat->AddEntry(dif_xSecNeutNo2p2h,"NEUT No 2p2h","l");
    leg_xsecGenOnlyCanv_dat->AddEntry(dif_xSecGenie,"GENIE","l");
    leg_xsecGenOnlyCanv_dat->SetFillColor(kWhite);
    leg_xsecGenOnlyCanv_dat->SetFillStyle(0);

    TLegend* leg_xsecGenOnlyCanv_dphit = new TLegend(0.4,0.55,0.8,0.85);
    leg_xsecGenOnlyCanv_dphit->AddEntry(dphitNEUTPriorXSecHist,"Input Simulation (NEUT 6B)","l");
    leg_xsecGenOnlyCanv_dphit->AddEntry(dif_xSecNuwro,"NuWro","l");
    leg_xsecGenOnlyCanv_dphit->AddEntry(dif_xSecNeutBask,"NEUT 6D","l");
    leg_xsecGenOnlyCanv_dphit->AddEntry(dif_xSecNeutNo2p2h,"NEUT No 2p2h","l");
    leg_xsecGenOnlyCanv_dphit->AddEntry(dif_xSecGenie,"GENIE","l");
    leg_xsecGenOnlyCanv_dphit->SetFillColor(kWhite);
    leg_xsecGenOnlyCanv_dphit->SetFillStyle(0);

    TCanvas* canvxs_GenOnly = new TCanvas("xsecCanv_GenOnly","xsecCanv_GenOnly");
    dif_xSecNuwro->Draw();
    dif_xSecNeut->Draw("same");
    dif_xSecNeutBask->Draw("same");
    dif_xSecNeutNo2p2h->Draw("same");
    dif_xSecGenie->Draw("same");
    leg_xsecGenOnlyCanv->Draw("same");
    canvxs_GenOnly->Write();
    if(var=="dpt") canvxs_GenOnly->SaveAs("xsecGenOnly.png");

    TCanvas* canvxs_GenOnly_dat = new TCanvas("xsecCanv_GenOnly_dat","xsecCanv_GenOnly_dat");
    datNuWroXSecHist->GetYaxis()->SetRangeUser(0, 1.5E-39);
    datNuWroXSecHist->Draw("HIST");
    datNEUTPriorXSecHist->Draw("sameHIST");
    datNEUTBaskHist->Draw("sameHIST");
    datNEUTNo2p2hXSecHist->Draw("sameHIST");
    datGenieXSecHist->Draw("sameHIST");
    leg_xsecGenOnlyCanv_dat->Draw("same");
    canvxs_GenOnly_dat->Write();
    if(var=="dat") canvxs_GenOnly_dat->SaveAs("xsecGenOnly.png");

    TCanvas* canvxs_GenOnly_dphit = new TCanvas("xsecCanv_GenOnly_dphit","xsecCanv_GenOnly_dphit");
    dphitNuWroXSecHist->Draw("HIST");
    dphitNEUTPriorXSecHist->Draw("sameHIST");
    dphitNEUTBaskHist->Draw("sameHIST");
    dphitNEUTNo2p2hXSecHist->Draw("sameHIST");
    dphitGenieXSecHist->Draw("sameHIST");
    leg_xsecGenOnlyCanv_dphit->Draw("same");
    canvxs_GenOnly_dphit->Write();
    if(var=="dphit") canvxs_GenOnly_dphit->SaveAs("xsecGenOnly.png");


    //*********************************
    //** NEUT GENIE only canv:
    //*********************************

    TLegend* leg_ngComp = new TLegend(0.5,0.6,0.8,0.9);
    leg_ngComp->AddEntry(dif_xSecNeut,"NEUT 6B","l");
    leg_ngComp->AddEntry(dif_xSecGenie,"GENIE","l");
    leg_ngComp->AddEntry(dif_xSecFit_allError,"Result","lep");
    leg_ngComp->SetFillColor(kWhite);
    leg_ngComp->SetFillStyle(0);

    TLegend* leg_ngCompW6D = new TLegend(0.5,0.6,0.8,0.9);
    leg_ngCompW6D->AddEntry(dif_xSecNeut,"NEUT 6B","l");
    leg_ngCompW6D->AddEntry(dif_xSecNeutBask,"NEUT 6D","l");
    leg_ngCompW6D->AddEntry(dif_xSecGenie,"GENIE","l");
    leg_ngCompW6D->AddEntry(dif_xSecFit_allError,"Result","lep");
    leg_ngCompW6D->SetFillColor(kWhite);
    leg_ngCompW6D->SetFillStyle(0);

    if(var=="dat"){
      leg_ngComp->SetX1(0.2);
      leg_ngComp->SetX2(0.5);
      leg_ngCompW6D->SetX1(0.2);
      leg_ngCompW6D->SetX2(0.5);
    }

    cout << "Making NEUT GENIE only comparissons:" << endl;

    TCanvas* canvxs_genComp_ngOnly = new TCanvas("xsecCanv_genComp_ngOnly","xsecCanv_genComp_ngOnly");
    dif_xSecNeut->GetYaxis()->SetRangeUser(0,12.5E-39);
    dif_xSecNeut->Draw("HIST");
    dif_xSecGenie->Draw("same");
    dif_xSecFit_allError->Draw("same");
    leg_ngComp->Draw("same");
    canvxs_genComp_ngOnly->Write();

    TCanvas* canvxs_genComp_ngOnly_dat = new TCanvas("xsecCanv_genComp_ngOnly_dat","xsecCanv_genComp_ngOnly_dat");
    datNEUTPriorXSecHist->GetYaxis()->SetRangeUser(0,6.5E-39);
    datNEUTPriorXSecHist->Draw("HIST");
    datGenieXSecHist->Draw("sameHIST");
    dif_xSecFit_allError->Draw("same");
    leg_ngComp->Draw("same");
    canvxs_genComp_ngOnly_dat->Write();

    TCanvas* canvxs_genComp_ngOnly_dphit = new TCanvas("xsecCanv_genComp_ngOnly_dphit","xsecCanv_genComp_ngOnly_dphit");
    dphitNEUTPriorXSecHist->GetYaxis()->SetRangeUser(0,1.5E-39);
    dphitNEUTPriorXSecHist->Draw("HIST");
    dphitGenieXSecHist->Draw("sameHIST");
    dif_xSecFit_allError->Draw("same");
    leg_ngComp->Draw("same");
    canvxs_genComp_ngOnly_dphit->Write();

    cout << "Making NEUT GENIE and 6D NEUT only comparissons:" << endl;


    TCanvas* canvxs_genComp_ngW6DOnly = new TCanvas("xsecCanv_genComp_ng6DOnly","xsecCanv_genComp_ng6DOnly");
    dif_xSecNeutBask->GetYaxis()->SetRangeUser(0,12.5E-39);
    dif_xSecNeutBask->Draw("HIST");
    dif_xSecNeut->Draw("same");
    dif_xSecGenie->Draw("same");
    dif_xSecFit_allError->Draw("same");
    leg_ngCompW6D->Draw("same");
    canvxs_genComp_ngW6DOnly->Write();

    TCanvas* canvxs_genComp_ngW6DOnly_dat = new TCanvas("xsecCanv_genComp_ng6DOnly_dat","xsecCanv_genComp_ng6DOnly_dat");
    datNEUTBaskHist->GetYaxis()->SetRangeUser(0,6.5E-39);
    datNEUTBaskHist->Draw("HIST");
    datNEUTPriorXSecHist->Draw("sameHIST");
    datGenieXSecHist->Draw("sameHIST");
    dif_xSecFit_allError->Draw("same");
    leg_ngCompW6D->Draw("same");
    canvxs_genComp_ngW6DOnly_dat->Write();

    TCanvas* canvxs_genComp_ngW6DOnly_dphit = new TCanvas("xsecCanv_genComp_ng6DOnly_dphit","xsecCanv_genComp_ng6DOnly_dphit");
    dphitNEUTBaskHist->GetYaxis()->SetRangeUser(0,1.5E-39);
    dphitNEUTBaskHist->Draw("HIST");
    dphitNEUTPriorXSecHist->Draw("sameHIST");
    dphitGenieXSecHist->Draw("sameHIST");
    dif_xSecFit_allError->Draw("same");
    leg_ngCompW6D->Draw("same");
    canvxs_genComp_ngW6DOnly_dphit->Write();


    //SHAPE ONLY

    cout << "Making shape only comparissons:" << endl;

    TCanvas* canvxs_genComp_shapeOnly = new TCanvas("xsecCanv_genComp_shapeOnly","xsecCanv_genComp_shapeOnly");
    dif_shapeOnly_xSecNuwro->Draw();
    dif_shapeOnly_xSecNeut->Draw("same");
    dif_shapeOnly_xSecNeutNo2p2h->Draw("same");
    dif_shapeOnly_xSecGenie->Draw("same");
    dif_shapeOnly_xSecFit->Draw("same");
    canvxs_genComp_shapeOnly->Write();

    TCanvas* canvxs_genComp_dat_shapeOnly = new TCanvas("xsecCanv_genComp_dat_shapeOnly","xsecCanv_genComp_dat_shapeOnly");
    datNuWroXSecHist_shapeOnly->Draw("HIST");
    datNEUTPriorXSecHist_shapeOnly->Draw("sameHIST");
    datNEUTNo2p2hXSecHist_shapeOnly->Draw("sameHIST");
    datGenieXSecHist_shapeOnly->Draw("sameHIST");
    dif_shapeOnly_xSecFit->Draw("same");
    canvxs_genComp_dat_shapeOnly->Write();

    TCanvas* canvxs_genComp_dphit_shapeOnly = new TCanvas("xsecCanv_genComp_dphit_shapeOnly","xsecCanv_genComp_dphit_shapeOnly");
    dphitNuWroXSecHist_shapeOnly->Draw("HIST");
    dphitNEUTPriorXSecHist_shapeOnly->Draw("sameHIST");
    dphitNEUTNo2p2hXSecHist_shapeOnly->Draw("sameHIST");
    dphitGenieXSecHist_shapeOnly->Draw("sameHIST");
    dif_shapeOnly_xSecFit->Draw("same");
    canvxs_genComp_dphit_shapeOnly->Write();
  }

  cout << "Writting all MC histos:" << endl;

  dif_xSecNuwro->Write("NuWro_dpt");
  dif_xSecNuwroFix->Write("NuWroScale_dpt");
  dif_xSecNeut->Write("NEUT6B_dpt");
  dif_xSecNeutBask->Write("NEUT6D_dpt");
  dif_xSecNeutNo2p2h->Write("NEUTNo2p2h_dpt");
  dif_xSecGenie->Write("GENIE_dpt");
  datNuWroXSecHist->Write("NuWro_dat");
  datNEUTPriorXSecHist->Write("NEUT6B_dat");
  datNEUTBaskHist->Write("NEUT6D_dat");
  datNEUTNo2p2hXSecHist->Write("NEUTNo2p2h_dat");
  datGenieXSecHist->Write("GENIE_dat");
  dphitNuWroXSecHist->Write("NuWro_dphit");
  dphitNEUTPriorXSecHist->Write("NEUT6B_dphit");
  dphitNEUTBaskHist->Write("NEUT6D_dphit");
  dphitNEUTNo2p2hXSecHist->Write("NEUTNo2p2h_dpt");
  dphitGenieXSecHist->Write("GENIE_dphit");

  dif_shapeOnly_xSecFit->Write();
  dif_shapeOnly_xSecFD->Write();
  dif_shapeOnly_xSecMC->Write();
  dif_shapeOnly_xSecNeut->Write();
  dif_shapeOnly_xSecGenie->Write();
  dif_shapeOnly_xSecNuwro->Write();
  dif_shapeOnly_xSecNeutNo2p2h->Write();
  datGenieXSecHist_shapeOnly->Write();
  datNuWroXSecHist_shapeOnly->Write();
  datNEUTPriorXSecHist_shapeOnly->Write();
  datNEUTNo2p2hXSecHist_shapeOnly->Write();
  dphitGenieXSecHist_shapeOnly->Write();
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

  (dptCombExtraSyst->GetSub(0,7,*dptCombExtraSyst)).Write("dptCovarCombExtraSyst");
  (dphitCombExtraSyst->GetSub(0,7,*dphitCombExtraSyst)).Write("dphitCovarCombExtraSyst");
  (datCombExtraSyst->GetSub(0,7,*datCombExtraSyst)).Write("datCovarCombExtraSyst");
  
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

  dptNEUTPriorXSecFile->Close();
  dptGENIEXSecFile->Close();
  dptNuWroXSecFile->Close();
  dptNEUTNo2p2hXSecFile->Close();
  datGENIEXSecFile->Close();
  datNuWroXSecFile->Close();
  datNEUTPriorXSecFile->Close();
  datNEUTPriorXSecFile->Close();
  dphitGENIEXSecFile->Close();
  dphitNuWroXSecFile->Close();
  dphitNEUTPriorXSecFile->Close();
  dphitNEUTPriorXSecFile->Close();

  propErrFile->Close();
  mcFile->Close();
  fitResultFile->Close();
  fakeDataFile->Close();


  outputFile->Close();

  cout << "Finished calcXSecWithErrors2" << endl;

}
