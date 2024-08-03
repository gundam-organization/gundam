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



using namespace std;

double calcChi2(TH1D* h1, TH1D* h2, TMatrixDSym covar_in, bool isShapeOnly=false){
  double chi2=0;
  TMatrixDSym covar(covar_in);
  TMatrixDSym covarForShapeOnly(covar_in);
  //To avoid working with tiny numbers (and maybe running into double precision issues)
  if(!isShapeOnly){
    for(int i=0; i<h1->GetNbinsX(); i++){
      for(int j=0; j<h1->GetNbinsX(); j++){
        covar[i][j]=covar_in[i][j];
        covarForShapeOnly[i][j]=covar_in[i][j];
      }
    }
  }

  //Matrix inversion section:
  bool inversionError = false;
  TMatrixDSym* covar_inv = new TMatrixDSym();
  int count = 0;
  while(true){
    inversionError = false;
    //For an accurate matrix inversion (and guaranteed symmetric inverse):  
    //TDecompBK covar_bk(covar);
    //TMatrixDSym covar_inv = covar_bk.Invert();
    //Alternative method (the one used in NUISANCE):
    TDecompSVD LU = TDecompSVD(covar);
    covar_inv = new TMatrixDSym(covar.GetNrows(), LU.Invert().GetMatrixArray(), "");
  
    //Check that the matrix inversion worked 
    TMatrixD test = covar*(*covar_inv);
    for(int i=0; i<h1->GetNbinsX(); i++){
      for(int j=0; j<h1->GetNbinsX(); j++){
        if(i==j && (test[i][j]>1.00001 || test[i][j]<0.99999)){
          if(!inversionError) cout << "****** WARNING: Issue with matrix inversion in chi2 calculation. See below!" << endl; 
          inversionError=true;
        }
        else if(i!=j && (test[i][j]>0.0000001)) {
          if(!inversionError) cout << "****** WARNING: Issue with matrix inversion in chi2 calculation. See below!" << endl; 
          inversionError=true;
        }
      }
    }
    // Section below still WIP
    //inversionError=false; 
    if(inversionError){ 
      cout << "DEBUG info after interation " << count << endl;
      cout << "  Is shape only: " << isShapeOnly << endl;
      //cout << "  Input matrix:" << endl;
      //covar.Print();
      //cout << "  Inverted matrix:" << endl;
      //covar_inv->Print();
      //cout << "  Test matrix:" << endl;
      //test.Print();
      cout << "  Input matrix: deritminent is: " << covar.Determinant() <<  endl;
      cout << "Will try adding 0.000001pc to the diagonal ..." << endl;
      for(int i=0; i<h1->GetNbinsX(); i++){
          covar[i][i]=covar[i][i]*1.00000001;
      }
      count++;
      //if(count>10)getchar();
    }
    else{
      cout << "chi2 calculated successfully after interaction " << count << endl;
      //if(count>0) test.Print();
      break;
    }
  }

  for(int i=0; i<h1->GetNbinsX(); i++){
    for(int j=0; j<h1->GetNbinsX(); j++){
      if(!isShapeOnly) chi2+= ((h1->GetBinContent(i+1)) - (h2->GetBinContent(i+1)))*((*covar_inv)[i][j])*((h1->GetBinContent(j+1)) - (h2->GetBinContent(j+1)));
      else chi2+= ((h1->GetBinContent(i+1)) - (h2->GetBinContent(i+1)))*((*covar_inv)[i][j])*((h1->GetBinContent(j+1)) - (h2->GetBinContent(j+1)));
    }
  }

  //Calc quick shape chi2: 
  if(!isShapeOnly){
    double shapeOnlyChi2=0;
    //double normRelError=9.58976791458074074e-02; // hard coded, sorry
    double normRelError=8.5e-02; // Just take the flux error for the moment

    TH1D* h2Norm = new TH1D(*h2);
    h2Norm->Scale(h1->Integral()/h2->Integral());

    if(abs(h2Norm->Integral()-h1->Integral())>1e-10) cout << "Problem with histo scaling in shape only chi2, integrals are: " << h2Norm->Integral() << " and " << h1->Integral() <<endl;

    for(int i=0; i<h1->GetNbinsX(); i++){
      for(int j=0; j<h1->GetNbinsX(); j++){
        covarForShapeOnly[i][j]=(covarForShapeOnly[i][j]-(normRelError*normRelError*h1->GetBinContent(i+1)*h1->GetBinContent(j+1)));
      }
    }
    TDecompSVD LU_2 = TDecompSVD(covarForShapeOnly);
    covar_inv = new TMatrixDSym(covarForShapeOnly.GetNrows(), LU_2.Invert().GetMatrixArray(), "");

    for(int i=0; i<h1->GetNbinsX(); i++){
      for(int j=0; j<h1->GetNbinsX(); j++){
        shapeOnlyChi2+= ((h1->GetBinContent(i+1)) - (h2Norm->GetBinContent(i+1)))*((*covar_inv)[i][j])*((h1->GetBinContent(j+1)) - (h2Norm->GetBinContent(j+1)));
      }
    }
    cout << "Quick shape only chi2 calcualted successfully: " << shapeOnlyChi2 << endl;  
  }

  if(!inversionError) cout << "Chi2 calcualted successfully: " << chi2 << endl;
  return chi2;
}

double getCombinedError(TMatrixDSym covar){
  double combError=0;
  for(int i=0; i < covar.GetNrows(); i++){
    for(int j=0; j < covar.GetNrows(); j++){
      combError+=covar[i][j];
    }
  }
  cout << "Found combined error of " << sqrt(combError) << endl;
  //cout << "Marix used: " << endl;
  //covar.Print();
  return sqrt(combError);
}

void makeComp_multiDif_0p(TString inNuisFileName, TString inResultFileName, TString outFileName, char* genName="NuWro11q SF"){
  TFile *inNuisFile = new TFile(inNuisFileName);
  TFile *inResultFile = new TFile(inResultFileName);

  TH1D* linearResult = (TH1D*)inResultFile->Get("LinResult");
  TH2D* covhist = (TH2D*)inResultFile->Get("CovMatrix");
  //covhist->Scale(0.01);
  TMatrixDSym* covMat = new TMatrixDSym(60);

  if(!linearResult || !covhist){
    cout << "Incorrect file format, exiting ..." << endl;
    return;
  }

  cout << "Building covariance matrix ... " << endl;
  for(int i=0;i<covhist->GetNbinsX();i++){
    for(int j=0;j<covhist->GetNbinsY();j++){
      //Convert to absolute covariance matrix: 
      if(i==j) cout << "For bin " << i+1 << ", relative covariance was " << covhist->GetBinContent(i+1,j+1);
      covhist->SetBinContent(i+1,j+1,(covhist->GetBinContent(i+1,j+1))*linearResult->GetBinContent(i+1)*linearResult->GetBinContent(j+1));
      (*covMat)[i][j] = covhist->GetBinContent(i+1,j+1);
      if(i==j) cout << ". Absolute covariance is now " << covhist->GetBinContent(i+1,j+1) << ", linear xsec is: " << linearResult->GetBinContent(i+1) << endl;
    }
    linearResult->SetBinError(i+1, sqrt(covhist->GetBinContent(i+1,i+1)));
  } 

  //covMat->Print();

  cout << "Assembling result TH2Polys ... " << endl;
  std::vector<TH1D*> result_slice;
  for (int i = 0; i < 10; i++){ 
    result_slice.push_back(new TH1D());
    result_slice[i] = (TH1D*) inNuisFile->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_data_Slice%i",i));
  }

  cout << "Assembling MC results ... " << endl;
  TH1D* linearResult_MC = (TH1D*)inNuisFile->Get("T2K_CC0pi_XSec_2DPcos_nu_joint_MC");
  if (strcasestr(genName,"gibuu")) linearResult_MC->Scale(1/10.0);


  std::vector<TH1D*> MC_slice;
  for (int i = 0; i < 10; i++){ 
    MC_slice.push_back(new TH1D());
    MC_slice[i] = (TH1D*)inNuisFile->Get(Form("T2K_CC0pi_XSec_2DPcos_nu_joint_MC_Slice%i",i));
    if (strcasestr(genName,"gibuu")) MC_slice[i]->Scale(1/10.0);
  }

  cout << "  Calculating chi2: " << endl;

  double chi2 = calcChi2(linearResult, linearResult_MC, *covMat);

  //Integrated over muon angle: 
  Double_t result_cthmu_bins[11] = {-1.0, -0.3, 0.3, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.98, 1.00};
  TH1D* result_cthmu = new TH1D("result_cthmu","Integrated over all p_{#mu}",10,result_cthmu_bins);
  result_cthmu->Sumw2();
  int binCount=1;
  for (int i = 0; i < 10; i++){ 
    //result_cthmu->SetBinContent(i+1, result_slice[i]->Integral("WIDTH"));
    if(i==0) result_cthmu->SetBinContent(i+1, linearResult->Integral(binCount,binCount+result_slice[i]->GetNbinsX()-1));
    else result_cthmu->SetBinContent(i+1, linearResult->Integral(binCount,binCount+result_slice[i]->GetNbinsX()-1));
    binCount+=result_slice[i]->GetNbinsX();
    //for(int b = 0; b < result_slice[i]->GetNbinsX(); b++){
    //  result_cthmu->SetBinContent(i+1, result_cthmu->GetBinContent(i+1)+result_cthmu->SetBinContent>GetBinContent(b+1));
    //}
    if(i==0) result_cthmu->SetBinError( i+1, getCombinedError(covMat->GetSub(0,result_slice[i]->GetNbinsX(),0,result_slice[i]->GetNbinsX())) );
    else result_cthmu->SetBinError( i+1, getCombinedError(covMat->GetSub(1+result_slice[i-1]->GetNbinsX(),1+result_slice[i-1]->GetNbinsX()+result_slice[i]->GetNbinsX(),1+result_slice[i-1]->GetNbinsX(),1+result_slice[i-1]->GetNbinsX()+result_slice[i]->GetNbinsX())) );
  }
  result_cthmu->Scale(1, "width");

  binCount=1;
  Double_t MC_cthmu_bins[11] = {-1.0, -0.3, 0.3, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.98, 1.00};
  TH1D* MC_cthmu = new TH1D("MC_cthmu","MC_cthmu",10,MC_cthmu_bins);
  for (int i = 0; i < 10; i++){ 
    //MC_cthmu->SetBinContent(i+1, MC_slice[i]->Integral("WIDTH"));
    if(i==0) MC_cthmu->SetBinContent(i+1, linearResult_MC->Integral(binCount,binCount+result_slice[i]->GetNbinsX()-1));
    else MC_cthmu->SetBinContent(i+1, linearResult_MC->Integral(binCount,binCount+result_slice[i]->GetNbinsX()-1));
    binCount+=result_slice[i]->GetNbinsX();
    //for(int b = 0; b < MC_slice[i]->GetNbinsX(); b++){
    //  MC_cthmu->SetBinContent(i+1, MC_cthmu->GetBinContent(i+1)+MC_slice[i]->GetBinContent(b+1));
    //}
  }
  MC_cthmu->Scale(1, "width");


  //***********************************************************************************

  cout << "Writting output ... " << endl;

  TFile *outfile = new TFile(outFileName,"recreate");
  outfile->cd();

  linearResult->Write("LinResult");
  linearResult_MC->Write("LinResult_MC");
  covhist->Write("CovMatrix");

  result_cthmu->SetXTitle("p_{#mu} (GeV)");
  result_cthmu->SetYTitle("#frac{d#sigma}{d p_{#mu}} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");
  MC_cthmu->SetLineColor(kBlue);
  MC_cthmu->SetMarkerStyle(0);

  for (int i = 0; i < 10; i++){ 
    result_slice[i]->SetXTitle("p_{#mu} (GeV)");
    result_slice[i]->SetYTitle("#frac{d#sigma}{d p_{#mu}} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");
    MC_slice[i]->SetLineColor(kBlue);
    MC_slice[i]->SetMarkerStyle(0);

    result_slice[i]->Write();
    MC_slice[i]->Write();
  }

  TPaveText *genText = new TPaveText(0.45, 1.0, 0.95, 1.4, Form("#bf{Generator:} %s", genName)); // left-up
  genText->AddText(Form("#bf{Generator:} %s", genName));

  TPaveText *chiText = new TPaveText(0.45, 0.7, 0.95, 0.9, Form("#bf{#chi^{2}:}%6.2f",chi2)); // left-up
  chiText->AddText(Form("#bf{#chi^{2}:}%6.2f",chi2));

  genText->SetFillColor(kWhite);
  genText->SetFillStyle(0);
  chiText->SetFillColor(kWhite);
  chiText->SetFillStyle(0);
  genText->SetLineColor(kWhite);
  chiText->SetLineColor(kWhite);

  cout << "Making pretty figures ... " << endl;

  gStyle->SetOptTitle(1);

  TCanvas* compCanv = new TCanvas("compCanv", "compCanv", 1920, 1080);
  compCanv->cd();

  TPad* cthmu = new TPad("cthmu","cthmu",0.0,0.75,0.33,1.00);
  cthmu->cd();
  result_cthmu->Draw();
  MC_cthmu->Draw("sameHIST");

  std::vector<TPad*> pad;

  for (int i = 0; i < 9; i++){ 
    switch(i){
      case 0 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.33,0.75,0.66,1.00));
                  result_slice[i+1]->GetYaxis()->SetRangeUser(0, 1.6);
                  result_slice[i+1]->SetNameTitle("-0.3 < cos(#theta_{#mu}) < 0.3", "-0.3 < cos(#theta_{#mu}) < 0.3");
                  break;
               }
      case 1 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.66,0.75,0.99,1.00));
                  result_slice[i+1]->GetYaxis()->SetRangeUser(0, 1.6);
                  result_slice[i+1]->SetNameTitle("0.3 < cos(#theta_{#mu}) < 0.6", "0.3 < cos(#theta_{#mu}) < 0.6");
                  break;
               }
      case 2 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.50,0.33,0.75));
                  result_slice[i+1]->GetYaxis()->SetRangeUser(0, 0.8);
                  result_slice[i+1]->SetNameTitle("0.6 < cos(#theta_{#mu}) < 0.7", "0.6 < cos(#theta_{#mu}) < 0.7");
                  break;
               }
      case 3 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.33,0.50,0.66,0.75));
                  result_slice[i+1]->GetYaxis()->SetRangeUser(0, 0.9);
                  result_slice[i+1]->SetNameTitle("0.7 < cos(#theta_{#mu}) < 0.8", "0.7 < cos(#theta_{#mu}) < 0.8");
                  break;
               }
      case 4 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.66,0.50,0.99,0.75));
                  result_slice[i+1]->GetYaxis()->SetRangeUser(0, 0.6);
                  result_slice[i+1]->SetNameTitle("0.8 < cos(#theta_{#mu}) < 0.85", "0.8 < cos(#theta_{#mu}) < 0.85");
                  break;
               }
      case 5 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.00,0.25,0.33,0.50));
                  result_slice[i+1]->GetYaxis()->SetRangeUser(0, 0.6);
                  result_slice[i+1]->SetNameTitle("0.85 < cos(#theta_{#mu}) < 0.9", "0.85 < cos(#theta_{#mu}) < 0.9");
                  break;
               }
      case 6 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.33,0.25,0.66,0.50));
                  result_slice[i+1]->GetYaxis()->SetRangeUser(0, 0.4);
                  result_slice[i+1]->SetNameTitle("0.9 < cos(#theta_{#mu}) < 0.94", "0.9 < cos(#theta_{#mu}) < 0.94");
                  break;
               }
      case 7 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.66,0.25,0.99,0.50));
                  result_slice[i+1]->GetYaxis()->SetRangeUser(0, 0.4);
                  result_slice[i+1]->SetNameTitle("0.94 < cos(#theta_{#mu}) < 0.98", "0.94 < cos(#theta_{#mu}) < 0.98");
                  break;
               }
      case 8 : {  pad.push_back(new TPad(Form("pad_%d",i),Form("pad_%d",i),0.33,0.00,0.66,0.25));
                  result_slice[i+1]->GetYaxis()->SetRangeUser(0, 0.15);
                  result_slice[i+1]->SetNameTitle("0.98 < cos(#theta_{#mu}) < 1.0", "0.98 < cos(#theta_{#mu}) < 1.0");
                  break;
               }
    }
    pad[i]->cd();
    result_slice[i+1]->Draw();
    MC_slice[i+1]->Draw("sameHIST");
    if(i==0){
     genText->Draw("same");
     chiText->Draw("same");
    }
  }

  compCanv->cd();
  cthmu->Draw();
  for (int i = 0; i < 9; i++){ 
    pad[i]->Draw("same");
  }

  result_cthmu->Write();
  MC_cthmu->Write();

  compCanv->Write();
  char* saveName = Form("%s_multDifComp_0p.pdf",genName);
  char* saveNamepng = Form("%s_multDifComp_0p.png",genName);
  compCanv->SaveAs(saveName);
  compCanv->SaveAs(saveNamepng);

  cout << "Finished 0p :-D" << endl;

  return;

}


//****************************************************************************


void makeComp_multiDif(TString inNuisFileName, TString inResultFileName, TString outFileName, char* genName="NuWro11q SF"){
  TFile *inNuisFile = new TFile(inNuisFileName);
  TFile *inResultFile = new TFile(inResultFileName);

  TH1D* linearResult = (TH1D*)inResultFile->Get("LinResult");
  TH2D* covhist = (TH2D*)inResultFile->Get("CovMatrix");
  //covhist->Scale(0.01);
  TMatrixDSym* covMat = new TMatrixDSym(32);

  if(!linearResult || !covhist){
    cout << "Incorrect file format, exiting ..." << endl;
    return;
  }

  cout << "Building covariance matrix ... " << endl;
  for(int i=0;i<covhist->GetNbinsX();i++){
    for(int j=0;j<covhist->GetNbinsY();j++){
      //Convert to absolute covariance matrix: 
      if(i==j) cout << "For bin " << i+1 << ", relative covariance was " << covhist->GetBinContent(i+1,j+1);
      covhist->SetBinContent(i+1,j+1,(covhist->GetBinContent(i+1,j+1))*linearResult->GetBinContent(i+1)*linearResult->GetBinContent(j+1));
      (*covMat)[i][j] = covhist->GetBinContent(i+1,j+1);
      if(i==j) cout << ". Absolute covariance is now " << covhist->GetBinContent(i+1,j+1) << ", linear xsec is: " << linearResult->GetBinContent(i+1) << endl;
    }
    linearResult->SetBinError(i+1, sqrt(covhist->GetBinContent(i+1,i+1)));
  } 

  //covMat->Print();

  cout << "Assembling result TH2Polys ... " << endl;
  std::vector<TH2Poly*> result_slice;
  for (int i = 0; i < 4; i++){ 
    result_slice.push_back(new TH2Poly());
    result_slice[i] = (TH2Poly*) inNuisFile->Get(Form("T2K_CC0pi1p_XSec_3DPcoscos_nu_nonuniform_data_Slice%i",i));
  }

  cout << "Assembling MC results ... " << endl;
  TH1D* linearResult_MC = (TH1D*)inNuisFile->Get("T2K_CC0pi1p_XSec_3DPcoscos_nu_nonuniform_MC");
  if (strcasestr(genName,"gibuu")) linearResult_MC->Scale(1/10.0);

  std::vector<TH2Poly*> MC_slice;
  for (int i = 0; i < 4; i++){ 
    MC_slice.push_back(new TH2Poly());
    MC_slice[i] = (TH2Poly*)inNuisFile->Get(Form("T2K_CC0pi1p_XSec_3DPcoscos_nu_nonuniform_MC_Slice%i",i));
    if (strcasestr(genName,"gibuu")) MC_slice[i]->Scale(1/10.0);
  }

  cout << "  Calculating chi2: " << endl;

  double chi2 = calcChi2(linearResult, linearResult_MC, *covMat);

  cout << "Building 1D slices from the data ... " << endl;

  //Get 1D slices as well:
  //*************************DATA*********************** 
  //Integrated over muon angle: 
  Double_t result_cthmu_bins[5] = {-1.0, -0.3, 0.3, 0.8, 1.00};
  TH1D* result_cthmu = new TH1D("result_cthmu","Integrated over all cos(#theta_{p})",4,result_cthmu_bins);
  result_cthmu->Sumw2();
  for (int i = 0; i < 4; i++){ 
    for(int b = 0; b < result_slice[i]->GetNumberOfBins(); b++){
      result_cthmu->SetBinContent(i+1, result_cthmu->GetBinContent(i+1)+result_slice[i]->GetBinContent(b+1));
    }
    if(i==0) result_cthmu->SetBinError( i+1, getCombinedError(covMat->GetSub(0,result_slice[i]->GetNumberOfBins(),0,result_slice[i]->GetNumberOfBins())) );
    else result_cthmu->SetBinError( i+1, getCombinedError(covMat->GetSub(1+result_slice[i-1]->GetNumberOfBins(),1+result_slice[i-1]->GetNumberOfBins()+result_slice[i]->GetNumberOfBins(),1+result_slice[i-1]->GetNumberOfBins(),1+result_slice[i-1]->GetNumberOfBins()+result_slice[i]->GetNumberOfBins())) );
  }
  result_cthmu->Scale(1, "width");

  //First muon anglular bin:
  int globalBinCount=0;
  Double_t result_cthmu1_bins[5] = {-1.00,  0.87,  0.94,  0.97, 1.00};
  TH1D* result_cthmu1 = new TH1D("result_cthmu1","-1.0 < cos(#theta_{#mu}) < -0.3",4,result_cthmu1_bins);
  result_cthmu1->Sumw2();
  cout << "Number of bins in slice 0 is " << result_slice[0]->GetNumberOfBins() << endl;
  for(int b = 0; b < result_slice[0]->GetNumberOfBins(); b++){
    result_cthmu1->SetBinContent(b+1, result_cthmu1->GetBinContent(b+1)+result_slice[0]->GetBinContent(b+1));
    result_cthmu1->SetBinError(b+1, sqrt((*covMat)[globalBinCount][globalBinCount]));
    globalBinCount++;
  }
  result_cthmu1->Scale(1, "width");

  //Second muon anglular bin:
  int pAngBin=1;
  int pMomBin=1;
  Double_t result_cthmu2_bins[5] = {-1.00,  0.75,  0.85,  0.94, 1.00};
  Double_t result_cthmu2_pmom_bins[5] = {0.50,  0.68,  0.78,  0.90, 1.50};
  TH1D* result_cthmu2 = new TH1D("result_cthmu2","-0.3 < cos(#theta_{#mu}) < 0.3",4,result_cthmu2_bins);
  TH1D* result_cthmu2_pmom = new TH1D("result_cthmu2_pmom","-0.3<cos(#theta_{#mu})<0.3 : 0.85<cos(#theta_{p})<0.94",4,result_cthmu2_pmom_bins);
  result_cthmu2->Sumw2();
  result_cthmu2_pmom->Sumw2();
  cout << "Number of bins in slice 1 is " << result_slice[1]->GetNumberOfBins() << endl;
  for(int b = 0; b < result_slice[1]->GetNumberOfBins(); b++){
    result_cthmu2->SetBinContent(pAngBin, result_cthmu2->GetBinContent(pAngBin)+result_slice[1]->GetBinContent(b+1));
    if(b==5) result_cthmu2->SetBinError( pAngBin, getCombinedError(covMat->GetSub(globalBinCount-4,globalBinCount,globalBinCount-4,globalBinCount)) );
    else result_cthmu2->SetBinError(pAngBin, sqrt((*covMat)[globalBinCount][globalBinCount]));
    if(b==0 || b==1 || b==5) pAngBin++;
    else{
      result_cthmu2_pmom->SetBinContent(pMomBin, result_cthmu2_pmom->GetBinContent(pMomBin)+result_slice[1]->GetBinContent(b+1));
      result_cthmu2_pmom->SetBinError(pMomBin, sqrt((*covMat)[globalBinCount][globalBinCount]));
      pMomBin++;
    }
    globalBinCount++;
  }
  result_cthmu2->Scale(1, "width");
  result_cthmu2_pmom->Scale(1, "width");
  result_cthmu2_pmom->SetBinContent(4, result_cthmu2_pmom->GetBinContent(4)*(result_cthmu2_pmom_bins[4]-result_cthmu2_pmom_bins[3])/(30.0-result_cthmu2_pmom_bins[4]));   // Scale for the true width of the last bin
  result_cthmu2_pmom->SetBinError(4, result_cthmu2_pmom->GetBinError(4)*(result_cthmu2_pmom_bins[4]-result_cthmu2_pmom_bins[3])/(30.0-result_cthmu2_pmom_bins[4]));   // Scale for the true width of the last bin

  //Third muon angular bin:
  pAngBin=1;
  pMomBin=1;
  Double_t result_cthmu3_bins[5] = {-1.00,  0.30,  0.50,  0.80, 1.00};
  Double_t result_cthmu3_pmom1_bins[6] = {0.50, 0.60, 0.70, 0.80, 0.90, 1.5};
  Double_t result_cthmu3_pmom2_bins[6] = {0.50, 0.60, 0.70, 0.80, 1.00, 1.5};
  TH1D* result_cthmu3 = new TH1D("result_cthmu3","0.3 < cos(#theta_{#mu}) < 0.8",4,result_cthmu3_bins);
  TH1D* result_cthmu3_pmom1 = new TH1D("result_cthmu3_pmom1","0.3<cos(#theta_{#mu})<0.8 : 0.5<cos(#theta_{p})<0.8",5,result_cthmu3_pmom1_bins);
  TH1D* result_cthmu3_pmom2 = new TH1D("result_cthmu3_pmom2","0.3<cos(#theta_{#mu})<0.8 : 0.8<cos(#theta_{p})<1.0",5,result_cthmu3_pmom2_bins);
  result_cthmu3->Sumw2();
  result_cthmu3_pmom1->Sumw2();
  result_cthmu3_pmom2->Sumw2();
  cout << "Number of bins in slice 2 is " << result_slice[2]->GetNumberOfBins() << endl;
  for(int b = 0; b < result_slice[2]->GetNumberOfBins(); b++){
    result_cthmu3->SetBinContent(pAngBin, result_cthmu3->GetBinContent(pAngBin)+result_slice[2]->GetBinContent(b+1));
    if(b==6) result_cthmu3->SetBinError( pAngBin, getCombinedError(covMat->GetSub(globalBinCount-5,globalBinCount,globalBinCount-5,globalBinCount)) );
    else if(b==11) result_cthmu3->SetBinError( pAngBin, getCombinedError(covMat->GetSub(globalBinCount-5,globalBinCount,globalBinCount-5,globalBinCount)) );
    else result_cthmu3->SetBinError(pAngBin, sqrt((*covMat)[globalBinCount][globalBinCount]));
    if(b==0 || b==1 || b==6) pAngBin++;
    if(b>=2 && b<=6){
      result_cthmu3_pmom1->SetBinContent(pMomBin, result_cthmu3_pmom1->GetBinContent(pMomBin)+result_slice[2]->GetBinContent(b+1));
      result_cthmu3_pmom1->SetBinError(pMomBin, sqrt((*covMat)[globalBinCount][globalBinCount]));
      pMomBin++;
    }
    if(b==6)pMomBin=1;
    else if(b>6){
      result_cthmu3_pmom2->SetBinContent(pMomBin, result_cthmu3_pmom2->GetBinContent(pMomBin)+result_slice[2]->GetBinContent(b+1));
      result_cthmu3_pmom2->SetBinError(pMomBin, sqrt((*covMat)[globalBinCount][globalBinCount]));
      pMomBin++;
    }
    globalBinCount++;
  }
  result_cthmu3->Scale(1, "width");
  result_cthmu3_pmom1->Scale(1, "width");
  result_cthmu3_pmom1->SetBinContent(5, result_cthmu3_pmom1->GetBinContent(5)*(result_cthmu3_pmom1_bins[5]-result_cthmu3_pmom1_bins[4])/(30.0-result_cthmu3_pmom1_bins[5]));   // Scale for the true width of the last bin
  result_cthmu3_pmom1->SetBinError(5, result_cthmu3_pmom1->GetBinError(5)*(result_cthmu3_pmom1_bins[5]-result_cthmu3_pmom1_bins[4])/(30.0-result_cthmu3_pmom1_bins[5]));   // Scale for the true width of the last bin
  result_cthmu3_pmom2->Scale(1, "width");
  result_cthmu3_pmom2->SetBinContent(5, result_cthmu3_pmom2->GetBinContent(5)*(result_cthmu3_pmom2_bins[5]-result_cthmu3_pmom2_bins[4])/(30.0-result_cthmu3_pmom2_bins[5]));   // Scale for the true width of the last bin
  result_cthmu3_pmom2->SetBinError(5, result_cthmu3_pmom2->GetBinError(5)*(result_cthmu3_pmom2_bins[5]-result_cthmu3_pmom2_bins[4])/(30.0-result_cthmu3_pmom2_bins[5]));   // Scale for the true width of the last bin


  //Fourth muon angular bin:
  pAngBin=1;
  pMomBin=1;
  Double_t result_cthmu4_bins[5] = {-1.00,  0.00,  0.30,  0.80, 1.00};
  Double_t result_cthmu4_pmom_bins[7] = {0.50, 0.60, 0.70, 0.80, 0.90, 1.10, 1.5};
  TH1D* result_cthmu4 = new TH1D("result_cthmu4","0.8 < cos(#theta_{#mu}) < 1.0",4,result_cthmu4_bins);
  TH1D* result_cthmu4_pmom = new TH1D("result_cthmu4_pmom","0.8<cos(#theta_{#mu})<1.0 : 0.3<cos(#theta_{p})<0.8",6,result_cthmu4_pmom_bins);
  result_cthmu4->Sumw2();
  result_cthmu4_pmom->Sumw2();
  cout << "Number of bins in slice 3 is " << result_slice[3]->GetNumberOfBins() << endl;
  for(int b = 0; b < result_slice[3]->GetNumberOfBins(); b++){
    result_cthmu4->SetBinContent(pAngBin, result_cthmu4->GetBinContent(pAngBin)+result_slice[3]->GetBinContent(b+1));
    if(b==7) result_cthmu4->SetBinError( pAngBin, getCombinedError(covMat->GetSub(globalBinCount-6,globalBinCount,globalBinCount-6,globalBinCount)) );
    else result_cthmu4->SetBinError(pAngBin, sqrt((*covMat)[globalBinCount][globalBinCount]));
    if(b==0 || b==1 || b==7) pAngBin++;
    else{
      result_cthmu4_pmom->SetBinContent(pMomBin, result_cthmu4_pmom->GetBinContent(pMomBin)+result_slice[3]->GetBinContent(b+1));
      result_cthmu4_pmom->SetBinError(pMomBin, sqrt((*covMat)[globalBinCount][globalBinCount]));
      pMomBin++;
    }
    globalBinCount++;
  }
  result_cthmu4->Scale(1, "width");
  result_cthmu4_pmom->Scale(1, "width");
  result_cthmu4_pmom->SetBinContent(6, result_cthmu4_pmom->GetBinContent(6)*(result_cthmu4_pmom_bins[6]-result_cthmu4_pmom_bins[5])/(30.0-result_cthmu4_pmom_bins[6]));   // Scale for the true width of the last bin
  result_cthmu4_pmom->SetBinError(6, result_cthmu4_pmom->GetBinError(6)*(result_cthmu4_pmom_bins[6]-result_cthmu4_pmom_bins[5])/(30.0-result_cthmu4_pmom_bins[6]));   // Scale for the true width of the last bin


  cout << "After filling all the data, globalBinCount is " << globalBinCount << endl;

  cout << "Building 1D slices from the MC ... " << endl;
  //*************************MC*********************** 
  //Integrated over muon angle: 
  Double_t MC_cthmu_bins[5] = {-1.0, -0.3, 0.3, 0.8, 1.00};
  TH1D* MC_cthmu = new TH1D("MC_cthmu","MC_cthmu",4,MC_cthmu_bins);
  globalBinCount=0;
  for (int i = 0; i < 4; i++){ 
    for(int b = 0; b < MC_slice[i]->GetNumberOfBins(); b++){
      MC_cthmu->SetBinContent(i+1, MC_cthmu->GetBinContent(i+1)+MC_slice[i]->GetBinContent(b+1));
    }
  }
  MC_cthmu->Scale(1, "width");

  //First muon anglular bin:
  Double_t MC_cthmu1_bins[5] = {-1.00,  0.87,  0.94,  0.97, 1.00};
  TH1D* MC_cthmu1 = new TH1D("MC_cthmu1","MC_cthmu1",4,MC_cthmu1_bins);
  cout << "Number of bins in slice 0 is " << MC_slice[0]->GetNumberOfBins() << endl;
  for(int b = 0; b < MC_slice[0]->GetNumberOfBins(); b++){
    MC_cthmu1->SetBinContent(b+1, MC_cthmu1->GetBinContent(b+1)+MC_slice[0]->GetBinContent(b+1));
  }
  MC_cthmu1->Scale(1, "width");

  //Second muon anglular bin:
  pAngBin=1;
  pMomBin=1;
  Double_t MC_cthmu2_bins[5] = {-1.00,  0.75,  0.85,  0.94, 1.00};
  Double_t MC_cthmu2_pmom_bins[5] = {0.50,  0.68,  0.78,  0.90, 1.50};
  TH1D* MC_cthmu2 = new TH1D("MC_cthmu2","MC_cthmu2",4,MC_cthmu2_bins);
  TH1D* MC_cthmu2_pmom = new TH1D("MC_cthmu2_pmom","MC_cthmu2_pmom",4,MC_cthmu2_pmom_bins);
  cout << "Number of bins in slice 1 is " << MC_slice[1]->GetNumberOfBins() << endl;
  for(int b = 0; b < MC_slice[1]->GetNumberOfBins(); b++){
    MC_cthmu2->SetBinContent(pAngBin, MC_cthmu2->GetBinContent(pAngBin)+MC_slice[1]->GetBinContent(b+1));
    if(b==0 || b==1 || b==5) pAngBin++;
    else{
      MC_cthmu2_pmom->SetBinContent(pMomBin, MC_cthmu2_pmom->GetBinContent(pMomBin)+MC_slice[1]->GetBinContent(b+1));
      pMomBin++;
    }
  }
  MC_cthmu2->Scale(1, "width");
  MC_cthmu2_pmom->Scale(1, "width");
  MC_cthmu2_pmom->SetBinContent(4, MC_cthmu2_pmom->GetBinContent(4)*(MC_cthmu2_pmom_bins[4]-MC_cthmu2_pmom_bins[3])/(30.0-MC_cthmu2_pmom_bins[4]));   // Scale for the true width of the last bin

  //Third muon angular bin:
  pAngBin=1;
  pMomBin=1;
  Double_t MC_cthmu3_bins[5] = {-1.00,  0.30,  0.50,  0.80, 1.00};
  Double_t MC_cthmu3_pmom1_bins[6] = {0.50, 0.60, 0.70, 0.80, 0.90, 1.5};
  Double_t MC_cthmu3_pmom2_bins[6] = {0.50, 0.60, 0.70, 0.80, 1.00, 1.5};
  TH1D* MC_cthmu3 = new TH1D("MC_cthmu3","MC_cthmu3",4,MC_cthmu3_bins);
  TH1D* MC_cthmu3_pmom1 = new TH1D("MC_cthmu3_pmom1","MC_cthmu3_pmom1",5,MC_cthmu3_pmom1_bins);
  TH1D* MC_cthmu3_pmom2 = new TH1D("MC_cthmu3_pmom2","MC_cthmu3_pmom2",5,MC_cthmu3_pmom2_bins);
  cout << "Number of bins in slice 2 is " << MC_slice[2]->GetNumberOfBins() << endl;
  for(int b = 0; b < MC_slice[2]->GetNumberOfBins(); b++){
    MC_cthmu3->SetBinContent(pAngBin, MC_cthmu3->GetBinContent(pAngBin)+MC_slice[2]->GetBinContent(b+1));
    if(b==0 || b==1 || b==6) pAngBin++;
    if(b>=2 && b<=6){
      MC_cthmu3_pmom1->SetBinContent(pMomBin, MC_cthmu3_pmom1->GetBinContent(pMomBin)+MC_slice[2]->GetBinContent(b+1));
      pMomBin++;
    }
    if(b==6)pMomBin=1;
    else if(b>6){
      MC_cthmu3_pmom2->SetBinContent(pMomBin, MC_cthmu3_pmom2->GetBinContent(pMomBin)+MC_slice[2]->GetBinContent(b+1));
      pMomBin++;
    }
  }
  MC_cthmu3->Scale(1, "width");
  MC_cthmu3_pmom1->Scale(1, "width");
  MC_cthmu3_pmom1->SetBinContent(5, MC_cthmu3_pmom1->GetBinContent(5)*(MC_cthmu3_pmom1_bins[5]-MC_cthmu3_pmom1_bins[4])/(30.0-MC_cthmu3_pmom1_bins[5]));   // Scale for the true width of the last bin
  MC_cthmu3_pmom2->Scale(1, "width");
  MC_cthmu3_pmom2->SetBinContent(5, MC_cthmu3_pmom2->GetBinContent(5)*(MC_cthmu3_pmom2_bins[5]-MC_cthmu3_pmom2_bins[4])/(30.0-MC_cthmu3_pmom2_bins[5]));   // Scale for the true width of the last bin

  //Fourth muon angular bin:
  pAngBin=1;
  pMomBin=1;
  Double_t MC_cthmu4_bins[5] = {-1.00,  0.00,  0.30,  0.80, 1.00};
  Double_t MC_cthmu4_pmom_bins[7] = {0.50, 0.60, 0.70, 0.80, 0.90, 1.10, 1.5};
  TH1D* MC_cthmu4 = new TH1D("MC_cthmu4","MC_cthmu4",4,MC_cthmu4_bins);
  TH1D* MC_cthmu4_pmom = new TH1D("MC_cthmu4_pmom","MC_cthmu4_pmom",6,MC_cthmu4_pmom_bins);
  cout << "Number of bins in slice 3 is " << MC_slice[3]->GetNumberOfBins() << endl;
  for(int b = 0; b < MC_slice[3]->GetNumberOfBins(); b++){
    MC_cthmu4->SetBinContent(pAngBin, MC_cthmu4->GetBinContent(pAngBin)+MC_slice[3]->GetBinContent(b+1));
    if(b==0 || b==1 || b==7) pAngBin++;
    else{
      MC_cthmu4_pmom->SetBinContent(pMomBin, MC_cthmu4_pmom->GetBinContent(pMomBin)+MC_slice[3]->GetBinContent(b+1));
      pMomBin++;
    }
  }
  MC_cthmu4->Scale(1, "width");
  MC_cthmu4_pmom->Scale(1, "width");
  MC_cthmu4_pmom->SetBinContent(6, MC_cthmu4_pmom->GetBinContent(6)*(MC_cthmu4_pmom_bins[6]-MC_cthmu4_pmom_bins[5])/(30.0-MC_cthmu4_pmom_bins[6]));   // Scale for the true width of the last bin


  //***********************************************************************************

  cout << "Writting output ... " << endl;

  TFile *outfile = new TFile(outFileName,"recreate");
  outfile->cd();

  linearResult->Write("LinResult");
  linearResult_MC->Write("LinResult_MC");
  covhist->Write("CovMatrix");

  for (int i = 0; i < 4; i++){ 
    result_slice[i]->Write();
    MC_slice[i]->Write();
  }

  result_cthmu->SetXTitle("cos(#theta_{#mu})");
  result_cthmu1->SetXTitle("cos(#theta_{p})");
  result_cthmu2->SetXTitle("cos(#theta_{p})");
  result_cthmu3->SetXTitle("cos(#theta_{p})");
  result_cthmu4->SetXTitle("cos(#theta_{p})");
  result_cthmu2_pmom->SetXTitle("p_{p} (GeV)");
  result_cthmu3_pmom1->SetXTitle("p_{p} (GeV)");
  result_cthmu3_pmom2->SetXTitle("p_{p} (GeV)");
  result_cthmu4_pmom->SetXTitle("p_{p} (GeV)");

  result_cthmu->SetYTitle("#frac{d#sigma}{d cos(#theta_{p})} (10^{-39} cm^{2} Nucleon^{-1})");
  result_cthmu1->SetYTitle("#frac{d#sigma}{d cos(#theta_{p})} (10^{-39} cm^{2} Nucleon^{-1})");
  result_cthmu2->SetYTitle("#frac{d#sigma}{d cos(#theta_{p})} (10^{-39} cm^{2} Nucleon^{-1})");
  result_cthmu3->SetYTitle("#frac{d#sigma}{d cos(#theta_{p})} (10^{-39} cm^{2} Nucleon^{-1})");
  result_cthmu4->SetYTitle("#frac{d#sigma}{d cos(#theta_{p})} (10^{-39} cm^{2} Nucleon^{-1})");
  result_cthmu2_pmom->SetYTitle("#frac{d#sigma}{d p_{p}} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");
  result_cthmu3_pmom1->SetYTitle("#frac{d#sigma}{d p_{p}} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");
  result_cthmu3_pmom2->SetYTitle("#frac{d#sigma}{d p_{p}} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");
  result_cthmu4_pmom->SetYTitle("#frac{d#sigma}{d p_{p}} (10^{-39} cm^{2} Nucleon^{-1} GeV^{-1})");

  MC_cthmu->SetLineColor(kBlue);
  MC_cthmu1->SetLineColor(kBlue);
  MC_cthmu2->SetLineColor(kBlue);
  MC_cthmu3->SetLineColor(kBlue);
  MC_cthmu4->SetLineColor(kBlue);
  MC_cthmu2_pmom->SetLineColor(kBlue);
  MC_cthmu3_pmom1->SetLineColor(kBlue);
  MC_cthmu3_pmom2->SetLineColor(kBlue);
  MC_cthmu4_pmom->SetLineColor(kBlue);

  TLatex latex;
  latex.SetTextSize(0.075);

  TPaveText *genText = new TPaveText(-0.8, 3.5, 0.8, 4.0, Form("#bf{Generator:} %s", genName)); // left-up
  genText->AddText(Form("#bf{Generator:} %s", genName));

  TPaveText *chiText = new TPaveText(-0.8, 2.9, 0.8, 3.4, Form("#bf{#chi^{2}:}%6.2f",chi2)); // left-up
  chiText->AddText(Form("#bf{#chi^{2}:}%6.2f",chi2));

  genText->SetFillColor(kWhite);
  genText->SetFillStyle(0);
  chiText->SetFillColor(kWhite);
  chiText->SetFillStyle(0);
  genText->SetLineColor(kWhite);
  chiText->SetLineColor(kWhite);


  result_cthmu->Write();
  result_cthmu1->Write();
  result_cthmu2->Write();
  result_cthmu3->Write();
  result_cthmu4->Write();
  result_cthmu2_pmom->Write();
  result_cthmu3_pmom1->Write();
  result_cthmu3_pmom2->Write();
  result_cthmu4_pmom->Write();

  MC_cthmu->Write();
  MC_cthmu1->Write();
  MC_cthmu2->Write();
  MC_cthmu3->Write();
  MC_cthmu4->Write();
  MC_cthmu2_pmom->Write();
  MC_cthmu3_pmom1->Write();
  MC_cthmu3_pmom2->Write();
  MC_cthmu4_pmom->Write();

  cout << "Making pretty figures ... " << endl;

  gStyle->SetOptTitle(1);

  TCanvas* compCanv = new TCanvas("compCanv", "compCanv", 1920, 1080);
  compCanv->cd();

  TPad* cthmu = new TPad("cthmu","cthmu",0.0,0.66,0.33,0.99);
  cthmu->cd();
  result_cthmu->Draw();
  MC_cthmu->Draw("same");

  TPad* cthmu1 = new TPad("cthmu1","cthmu1",0.33,0.66,0.66,0.99);
  cthmu1->cd();
  result_cthmu1->Draw();
  MC_cthmu1->Draw("same");
  genText->Draw("same");
  chiText->Draw("same");
  //latex.DrawLatex(gPad->GetUxmax()*0.0,gPad->GetUymax()*0.9,"#bf{Generator:} NuWro11q SF");
  //latex.DrawLatex(gPad->GetUxmax()*0.0,gPad->GetUymax()*0.75,Form("#bf{#chi^{2}:}%f",chi2));


  TPad* cthmu2 = new TPad("cthmu2","cthmu2",0.66,0.66,0.99,0.99);
  cthmu2->cd();
  result_cthmu2->Draw();
  MC_cthmu2->Draw("same");

  TPad* cthmu2_pmom = new TPad("cthmu2_pmom","cthmu2_pmom",0.0,0.33,0.33,0.66);
  cthmu2_pmom->cd();
  result_cthmu2_pmom->GetYaxis()->SetRangeUser(0,0.6);
  result_cthmu2_pmom->Draw();
  MC_cthmu2_pmom->Draw("same");

  TPad* cthmu3 = new TPad("cthmu3","cthmu3",0.33,0.33,0.66,0.66);
  cthmu3->cd();
  result_cthmu3->GetYaxis()->SetRangeUser(0,1.6);
  result_cthmu3->Draw();
  MC_cthmu3->Draw("same");

  TPad* cthmu3_pmom1 = new TPad("cthmu3_pmom1","cthmu3_pmom1",0.66,0.33,0.99,0.66);
  cthmu3_pmom1->cd();
  result_cthmu3_pmom1->Draw();
  MC_cthmu3_pmom1->Draw("same");

  TPad* cthmu3_pmom2 = new TPad("cthmu3_pmom2","cthmu3_pmom2",0.0,0.00,0.33,0.33);
  cthmu3_pmom2->cd();
  result_cthmu3_pmom2->Draw();
  MC_cthmu3_pmom2->Draw("same");

  TPad* cthmu4 = new TPad("cthmu4","cthmu4",0.33,0.00,0.66,0.33);
  cthmu4->cd();
  result_cthmu4->Draw();
  MC_cthmu4->Draw("same");

  TPad* cthmu4_pmom = new TPad("cthmu4_pmom","cthmu4_pmom",0.66,0.00,0.99,0.33);
  cthmu4_pmom->cd();
  result_cthmu4_pmom->Draw();
  MC_cthmu4_pmom->Draw("same");

  compCanv->cd();
  cthmu->Draw();
  cthmu1->Draw("same");
  cthmu2->Draw("same");
  cthmu2_pmom->Draw("same");
  cthmu3->Draw("same");
  cthmu3_pmom1->Draw("same");
  cthmu3_pmom2->Draw("same");
  cthmu4->Draw("same");
  cthmu4_pmom->Draw("same");

  compCanv->Write();
  char* saveName = Form("%s_multDifComp.pdf",genName);
  char* saveNamepng = Form("%s_multDifComp.png",genName);
  compCanv->SaveAs(saveName);
  compCanv->SaveAs(saveNamepng);


  cout << "Finished :-D" << endl;

  return;
}


void makeComp_allComb(TString in0pFileName, TString in1pFileName, TString inNpFileName,TString inResultCovarFileName, TString outFileName, char* genName="NuWro11q SF"){

  TFile *in0pFile = new TFile(in0pFileName);
  TFile *in1pFile = new TFile(in1pFileName);
  TFile *inNpFile = new TFile(inNpFileName);

  TFile *inResultCovarFile = new TFile(inResultCovarFileName);

  TH1D* linearResult_0p = (TH1D*)in0pFile->Get("LinResult");
  TH1D* linearResult_1p = (TH1D*)in1pFile->Get("LinResult");
  TH1D* linearResult_Np = (TH1D*)inNpFile->Get("T2K_CC0pi_XSec_1bin_nu_data");

  TH2D* covHist_0p = (TH2D*)in0pFile->Get("CovMatrix");
  TH2D* covHist_1p = (TH2D*)in1pFile->Get("CovMatrix");
  TMatrixDSym* covMat_0p = new TMatrixDSym(linearResult_0p->GetNbinsX());
  TMatrixDSym* covMat_1p = new TMatrixDSym(linearResult_1p->GetNbinsX());

  TH1D* linearMC_0p = (TH1D*)in0pFile->Get("LinResult_MC");
  TH1D* linearMC_1p = (TH1D*)in1pFile->Get("LinResult_MC");
  TH1D* linearMC_Np = (TH1D*)inNpFile->Get("T2K_CC0pi_XSec_1bin_nu_MC");

  TH1D* cthResult_0p = (TH1D*)in0pFile->Get("result_cthmu");
  TH1D* cthResult_1p = (TH1D*)in1pFile->Get("result_cthmu");

  TH1D* cthMC_0p = (TH1D*)in0pFile->Get("MC_cthmu");
  TH1D* cthMC_1p = (TH1D*)in1pFile->Get("MC_cthmu");

  TMatrixDSym* covhist_syst = (TMatrixDSym*)inResultCovarFile->Get("RelCov_syst");
  TMatrixDSym* covhist_stat = (TMatrixDSym*)inResultCovarFile->Get("RelCov_stat");
  //covhist->Scale(0.01);

  TMatrixDSym* covMat = new TMatrixDSym(93);
  TH1D* linearResult_comb = new TH1D("linearResult_comb","linearResult_comb",93,0,93);
  TH1D* linearMC_comb = new TH1D("linearResult_comb","linearResult_comb",93,0,93);

  if(!linearResult_0p || !covhist_syst){
    cout << "Incorrect file format, exiting ..." << endl;
    return;
  }

  for(int i=0;i<linearResult_0p->GetNbinsX();i++){
    for(int j=0;j<linearResult_0p->GetNbinsX();j++){
      (*covMat_0p)[i][j] = covHist_0p->GetBinContent(i+1,j+1);
    }
  }

  for(int i=0;i<linearResult_1p->GetNbinsX();i++){
    for(int j=0;j<linearResult_1p->GetNbinsX();j++){
      (*covMat_1p)[i][j] = covHist_1p->GetBinContent(i+1,j+1);
    }
  }

  int globalBinCount=0;

  for(int i=0;i<linearResult_0p->GetNbinsX();i++){
    linearMC_comb->SetBinContent(globalBinCount+1, linearMC_0p->GetBinContent(i+1));
    linearResult_comb->SetBinContent(globalBinCount+1, linearResult_0p->GetBinContent(i+1));
    linearResult_comb->SetBinError(globalBinCount+1, linearResult_0p->GetBinError(i+1));
    globalBinCount++;
  }

  for(int i=0;i<linearResult_1p->GetNbinsX();i++){
    linearMC_comb->SetBinContent(globalBinCount+1, linearMC_1p->GetBinContent(i+1));
    linearResult_comb->SetBinContent(globalBinCount+1, linearResult_1p->GetBinContent(i+1));
    linearResult_comb->SetBinError(globalBinCount+1, linearResult_1p->GetBinError(i+1));
    globalBinCount++;
  }

  linearMC_comb->SetBinContent(globalBinCount+1, linearMC_Np->GetBinContent(1)/1E-39);
  linearResult_comb->SetBinContent(globalBinCount+1, linearResult_Np->GetBinContent(1)/1E-39);
  linearResult_comb->SetBinError(globalBinCount+1, linearResult_Np->GetBinError(1)/1E-39);



  cout << "Building covariance matrix ... " << endl;
  for(int i=0;i<linearResult_comb->GetNbinsX();i++){
    for(int j=0;j<linearResult_comb->GetNbinsX();j++){
      (*covMat)[i][j] = (*covhist_syst)[i][j] + (*covhist_stat)[i][j]; 
      //Convert to absolute covariance matrix: 
      (*covMat)[i][j] = (*covMat)[i][j]*linearResult_comb->GetBinContent(i+1)*linearResult_comb->GetBinContent(j+1);
    }
  } 

  cout << "  Calculating chi2: " << endl;

  double chi2 = 0;
  chi2 = calcChi2(linearResult_comb, linearMC_comb, *covMat);

  // Get integrated result

  double intResult_0p = cthResult_0p->Integral("WIDTH");
  double intResult_1p = cthResult_1p->Integral("WIDTH");

  double intMC_0p = cthMC_0p->Integral("WIDTH");
  double intMC_1p = cthMC_1p->Integral("WIDTH");

  double intError_0p = getCombinedError(*covMat_0p);
  double intError_1p = getCombinedError(*covMat_1p);

  //Make N protons histo

  TH1D* nprotonsResultHist = new TH1D ("# protons with p_{p} > 500 MeV", "# protons with p_{p} > 500 MeV", 3, -0.5, 2.5);
  TH1D* nprotonsMCHist = new TH1D ("# protons with p_{p} > 500 MeV", "# protons with p_{p} > 500 MeV", 3, -0.5, 2.5);

  nprotonsMCHist->SetBinContent(1, intMC_0p);
  nprotonsMCHist->SetBinContent(2, intMC_1p);
  nprotonsMCHist->SetBinContent(3, linearMC_Np->GetBinContent(1)/1E-39);

  nprotonsResultHist->SetBinContent(1, intResult_0p);
  nprotonsResultHist->SetBinContent(2, intResult_1p);
  nprotonsResultHist->SetBinContent(3, linearResult_Np->GetBinContent(1)/1E-39);

  nprotonsResultHist->SetBinError(1, intError_0p);
  nprotonsResultHist->SetBinError(2, intError_1p);
  nprotonsResultHist->SetBinError(3, linearResult_Np->GetBinError(1)/1E-39);

//***********************************************************************************

  cout << "Writting output ... " << endl;

  TFile *outfile = new TFile(outFileName,"recreate");
  outfile->cd();

  TH1D* chi2hist = new TH1D("chi2hist", "chi2hist", 1, 0, 1);
  chi2hist->SetBinContent(1,chi2);
  chi2hist->Write();

  linearResult_comb->SetXTitle("Analysis Bin");
  linearResult_comb->SetYTitle("Cross Section (Nucl.^{-1} 10^{-39} cm^{2})");

  nprotonsResultHist->SetXTitle("Number of protons");
  nprotonsResultHist->SetYTitle("Cross Section (Nucl.^{-1} 10^{-39} cm^{2})");

  linearResult_comb->SetNameTitle("Result in linear binning", "Result in linear binning");
  nprotonsResultHist->SetNameTitle("Number of protons w/ p_{p}>500 MeV", "Number of protons w/ p_{p}>500 MeV");

  nprotonsResultHist->GetYaxis()->SetRangeUser(0,3.0);
  linearResult_comb->GetYaxis()->SetRangeUser(0,0.5);

  nprotonsMCHist->SetLineColor(kBlue);
  linearMC_comb->SetLineColor(kBlue);

  linearResult_comb->Write("LinResult");
  linearMC_comb->Write("LinResult_MC");

  nprotonsResultHist->Write("nprotonsResult");
  nprotonsMCHist->Write("nprotonsResult_MC");

  covMat->Write("CovMatrix");

  TPaveText *genText = new TPaveText(1.6, 2.5, 2.4, 2.9, Form("#bf{Generator:} %s", genName)); // left-up
  genText->AddText(Form("#bf{Generator:} %s", genName));
  TPaveText *chiText = new TPaveText(1.6, 2.0, 2.4, 2.4, Form("#bf{#chi^{2}:}%6.2f",chi2)); // left-up
  chiText->AddText(Form("#bf{#chi^{2}:}%6.2f",chi2));

  TPaveText *genText_linResult = new TPaveText(30, 0.36, 70, 0.4, Form("#bf{Generator:} %s", genName)); // left-up
  genText_linResult->AddText(Form("#bf{Generator:} %s", genName));
  TPaveText *chiText_linResult = new TPaveText(30, 0.3, 70, 0.34, Form("#bf{#chi^{2}:}%6.2f",chi2)); // left-up
  chiText_linResult->AddText(Form("#bf{#chi^{2}:}%6.2f",chi2));

  genText->SetFillColor(kWhite);
  genText->SetFillStyle(0);
  chiText->SetFillColor(kWhite);
  chiText->SetFillStyle(0);
  genText->SetLineColor(kWhite);
  chiText->SetLineColor(kWhite);

  genText_linResult->SetFillColor(kWhite);
  genText_linResult->SetFillStyle(0);
  chiText_linResult->SetFillColor(kWhite);
  chiText_linResult->SetFillStyle(0);
  genText_linResult->SetLineColor(kWhite);
  chiText_linResult->SetLineColor(kWhite);


  cout << "Making pretty figures ... " << endl;

  gStyle->SetOptTitle(1);

  TCanvas* nprotonsCanv = new TCanvas("nprotonsCanv", "nprotonsCanv");
  nprotonsCanv->cd();
  nprotonsResultHist->Draw();
  nprotonsMCHist->Draw("same");
  nprotonsCanv->Write("nprotonsCanv");

  TCanvas* nprotonsCanv_wchi2 = new TCanvas("nprotonsCanv_wchi2", "nprotonsCanv_wchi2");
  nprotonsCanv_wchi2->cd();
  nprotonsResultHist->Draw();
  nprotonsMCHist->Draw("same");
  genText->Draw("same");
  chiText->Draw("same");
  nprotonsCanv_wchi2->Write("nprotonsCanv_wchi2");

  TCanvas* LinResultCanv = new TCanvas("LinResultCanv", "LinResultCanv");
  LinResultCanv->cd();
  linearResult_comb->Draw();
  linearMC_comb->Draw("same");
  LinResultCanv->Write("LinResultCanv");

  TCanvas* LinResultCanv_wchi2 = new TCanvas("LinResultCanv_wchi2", "LinResultCanv_wchi2");
  LinResultCanv_wchi2->cd();
  linearResult_comb->Draw();
  linearMC_comb->Draw("same");
  genText_linResult->Draw("same");
  chiText_linResult->Draw("same");
  LinResultCanv_wchi2->Write("LinResultCanv_wchi2");

  char* saveName = Form("%s_nprotons.pdf",genName);
  char* saveNamepng = Form("%s_nprotons.png",genName);
  nprotonsCanv->SaveAs(saveName);
  nprotonsCanv->SaveAs(saveNamepng);

  char* saveName_lin = Form("%s_linResult.pdf",genName);
  char* saveNamepng_lin = Form("%s_linResult.png",genName);
  LinResultCanv_wchi2->SaveAs(saveName_lin);
  LinResultCanv_wchi2->SaveAs(saveNamepng_lin);


  cout << "Finished :-D" << endl;

  return;
}
