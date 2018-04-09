/******************************************************

Code to convert Jiae's CC0pi+1p bins into the input
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


void convertJiae(TString inFileName, TString outFileName){
  TFile *infile = new TFile(inFileName);
  TH1D* result_p  = (TH1D*)infile->Get("hresult_momentum");
  TH1D* result_a  = (TH1D*)infile->Get("hresult_angle");
  TH1D* result_tp = (TH1D*)infile->Get("hresult_threemomentum");

  TMatrixD* cov_p  = (TMatrixD*)infile->Get("cvm_momentum");
  TMatrixD* cov_a  = (TMatrixD*)infile->Get("cvm_angle");
  TMatrixD* cov_tp = (TMatrixD*)infile->Get("cvm_threemomentum");

  const int nbins_p  = 7;
  const int nbins_a  = 5;
  const int nbins_tp = 7;

  //Real bin sizes
  //double bins_p[nbins_p+1]   = {-5,-0.3,0,0.1,0.2,0.3,0.5,5};
  //double bins_a[nbins_a+1]   = {-360,-5. , 5., 10., 20., 360};
  //double bins_tp[nbins_tp+1] = {0, 0.3, 0.4, 0.5, 0.6, 0.7,0.9, 5.0};

  //Modified bin sizes for pretty plotting
  double bins_p[nbins_p+1]   = {-1.0,-0.3,0,0.1,0.2,0.3,0.5,1.5};
  double bins_a[nbins_a+1]   = {-50,-5. , 5., 10., 20., 80};
  double bins_tp[nbins_tp+1] = {0, 0.3, 0.4, 0.5, 0.6, 0.7,0.9, 1.5};

  //Calculate correlation matricies

  TMatrixD* cor_p  = new TMatrixD(result_p->GetNbinsX(),result_p->GetNbinsX());
  TMatrixD* cor_a  = new TMatrixD(result_a->GetNbinsX(),result_a->GetNbinsX());
  TMatrixD* cor_tp = new TMatrixD(result_tp->GetNbinsX(),result_tp->GetNbinsX());

  for(int i=0;i<result_p->GetNbinsX();i++){
    if(result_p->GetBinContent(i)==0){
      result_p->SetBinContent(i, 0.0);
      result_p->SetBinError(i, 0.0001);
    }
    for(int j=0;j<result_p->GetNbinsX();j++){
      if((*cov_p)[i][i]!=0 && (*cov_p)[j][j]!=0) (*cor_p)[i][j] = (*cov_p)[i][j]/sqrt(((*cov_p)[i][i]*(*cov_p)[j][j]));
      else if (i!=j)((*cor_p)[i][j] = 0.0);
      else if (i==j)((*cor_p)[i][j] = 1.0);
    }
  }

  for(int i=0;i<result_a->GetNbinsX();i++){
    if(result_a->GetBinContent(i)==0){
      result_a->SetBinContent(i, 0.0);
      result_a->SetBinError(i, 0.000001);
    }
    for(int j=0;j<result_a->GetNbinsX();j++){
      if((*cov_a)[i][i]!=0 && (*cov_a)[j][j]!=0) (*cor_a)[i][j] = (*cov_a)[i][j]/sqrt(((*cov_a)[i][i]*(*cov_a)[j][j]));
      else if (i!=j)((*cor_p)[i][j] = 0.0);
      else if (i==j)((*cor_p)[i][j] = 1.0);    }
  }

  for(int i=0;i<result_tp->GetNbinsX();i++){
    if(result_tp->GetBinContent(i)==0){
      result_tp->SetBinContent(i, 0.0);
      result_tp->SetBinError(i, 0.0001);
    }
    for(int j=0;j<result_tp->GetNbinsX();j++){
      if((*cov_tp)[i][i]!=0 && (*cov_tp)[j][j]!=0) (*cor_tp)[i][j] = (*cov_tp)[i][j]/sqrt(((*cov_tp)[i][i]*(*cov_tp)[j][j]));
      else if (i!=j)((*cor_p)[i][j] = 0.0);
      else if (i==j)((*cor_p)[i][j] = 1.0);
    }
  }


  TH1D* resultBin0_p  = new TH1D("-1.0 < cos(#theta_{#mu}) < -0.6","-1.0 < cos(#theta_{#mu}) < -0.6",nbins_p,bins_p);
  TH1D* resultBin1_p  = new TH1D("-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} < 250MeV/c","-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} < 250MeV/c",nbins_p,bins_p);
  TH1D* resultBin2_p  = new TH1D("-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} > 250MeV/c","-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} > 250MeV/c",nbins_p,bins_p);
  TH1D* resultBin3_p  = new TH1D("0.0 < cos(#theta_{#mu}) < 1.0, p_{#mu} < 250MeV/c","0.0 < cos(#theta_{#mu}) < 0.8, p_{#mu} < 250MeV/c",nbins_p,bins_p);
  TH1D* resultBin4_p  = new TH1D("0.0 < cos(#theta_{#mu}) < 0.8, p_{#mu} > 250MeV/c","0.0 < cos(#theta_{#mu}) < 0.8, p_{#mu} > 250MeV/c",nbins_p,bins_p);
  TH1D* resultBin5_p  = new TH1D("0.8 < cos(#theta_{#mu}) < 1.0, 250MeV/c < p_{#mu} < 750MeV/c","0.8 < cos(#theta_{#mu}) < 1.0, 250MeV/c < p_{#mu} < 750MeV/c",nbins_p,bins_p);
  TH1D* resultBin6_p  = new TH1D("0.8 < cos(#theta_{#mu}) < 1.0, 750MeV/c < p_{#mu}","0.8 < cos(#theta_{#mu}) < 1.0, 750MeV/c < p_{#mu}",nbins_p,bins_p);

  TH1D* resultBin0_a  = new TH1D("-1.0 < cos(#theta_{#mu}) < -0.6","-1.0 < cos(#theta_{#mu}) < -0.6",nbins_a,bins_a);
  TH1D* resultBin1_a  = new TH1D("-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} < 250MeV/c","-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} < 250MeV/c",nbins_a,bins_a);
  TH1D* resultBin2_a  = new TH1D("-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} > 250MeV/c","-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} > 250MeV/c",nbins_a,bins_a);
  TH1D* resultBin3_a  = new TH1D("0.0 < cos(#theta_{#mu}) < 1.0, p_{#mu} < 250MeV/c","0.0 < cos(#theta_{#mu})<0.8, p_{#mu} < 250MeV/c",nbins_a,bins_a);
  TH1D* resultBin4_a  = new TH1D("0.0 < cos(#theta_{#mu}) < 0.8, p_{#mu} > 250MeV/c","0.0 < cos(#theta_{#mu})<0.8, p_{#mu} > 250MeV/c",nbins_a,bins_a);
  TH1D* resultBin5_a  = new TH1D("0.8 < cos(#theta_{#mu}) < 1.0, 250MeV/c < p_{#mu} < 750MeV/c","0.8 < cos(#theta_{#mu}) < 1.0, 250MeV/c < p_{#mu} < 750MeV/c",nbins_a,bins_a);
  TH1D* resultBin6_a  = new TH1D("0.8 < cos(#theta_{#mu}) < 1.0, 750MeV/c < p_{#mu}","0.8 < cos(#theta_{#mu}) < 1.0, 750MeV/c < p_{#mu}",nbins_a,bins_a);

  TH1D* resultBin0_tp = new TH1D("-1.0 < cos(#theta_{#mu}) < -0.6","-1.0 < cos(#theta_{#mu}) < -0.6",nbins_tp,bins_tp);
  TH1D* resultBin1_tp = new TH1D("-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} < 250MeV/c","-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} < 250MeV/c",nbins_tp,bins_tp);
  TH1D* resultBin2_tp = new TH1D("-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} > 250MeV/c","-0.6 < cos(#theta_{#mu}) <  0.0, p_{#mu} > 250MeV/c",nbins_tp,bins_tp);
  TH1D* resultBin3_tp = new TH1D("0.0 < cos(#theta_{#mu}) < 1.0, p_{#mu}<250MeV/c","0.0 < cos(#theta_{#mu}) < 0.8, p_{#mu} < 250MeV/c",nbins_tp,bins_tp);
  TH1D* resultBin4_tp = new TH1D("0.0 < cos(#theta_{#mu}) < 0.8, p_{#mu}>250MeV/c","0.0 < cos(#theta_{#mu}) < 0.8, p_{#mu} > 250MeV/c",nbins_tp,bins_tp);
  TH1D* resultBin5_tp = new TH1D("0.8 < cos(#theta_{#mu}) < 1.0, 250MeV/c < p_{#mu} < 750MeV/c","0.8 < cos(#theta_{#mu}) < 1.0, 250MeV/c < p_{#mu} < 750MeV/c",nbins_tp,bins_tp);
  TH1D* resultBin6_tp = new TH1D("0.8 < cos(#theta_{#mu}) < 1.0, 750MeV/c < p_{#mu}","0.8 < cos(#theta_{#mu}) < 1.0, 750MeV/c < p_{#mu}",nbins_tp,bins_tp);

  for(int i=2; i<7*nbins_p+2;i++){ // i starts at 2 to deal with the under/over-flow bin being bin 1 (and not -1)
    if(i<nbins_p+2) resultBin0_p->SetBinContent(i-1, result_p->GetBinContent(i)); 
    else if(i<(2*nbins_p)+2) resultBin1_p->SetBinContent(i-(1*nbins_p)-1, result_p->GetBinContent(i)); 
    else if(i<(3*nbins_p)+2) resultBin2_p->SetBinContent(i-(2*nbins_p)-1, result_p->GetBinContent(i)); 
    else if(i<(4*nbins_p)+2) resultBin3_p->SetBinContent(i-(3*nbins_p)-1, result_p->GetBinContent(i)); 
    else if(i<(5*nbins_p)+2) resultBin4_p->SetBinContent(i-(4*nbins_p)-1, result_p->GetBinContent(i)); 
    else if(i<(6*nbins_p)+2) resultBin5_p->SetBinContent(i-(5*nbins_p)-1, result_p->GetBinContent(i)); 
    else if(i<(7*nbins_p)+2) resultBin6_p->SetBinContent(i-(6*nbins_p)-1, result_p->GetBinContent(i));
    else std::cout << "WARNING: seem to have overran the bins ...; iterator is: " << i << std::endl; 
  }

  for(int i=2; i<7*nbins_a+2;i++){ // i starts at 2 to deal with the under/over-flow bin being bin 1 (and not -1)
    if(i<nbins_a+2) resultBin0_a->SetBinContent(i-1, result_a->GetBinContent(i)); 
    else if(i<(2*nbins_a)+2) resultBin1_a->SetBinContent(i-(1*nbins_a)-1, result_a->GetBinContent(i)); 
    else if(i<(3*nbins_a)+2) resultBin2_a->SetBinContent(i-(2*nbins_a)-1, result_a->GetBinContent(i)); 
    else if(i<(4*nbins_a)+2) resultBin3_a->SetBinContent(i-(3*nbins_a)-1, result_a->GetBinContent(i)); 
    else if(i<(5*nbins_a)+2) resultBin4_a->SetBinContent(i-(4*nbins_a)-1, result_a->GetBinContent(i)); 
    else if(i<(6*nbins_a)+2) resultBin5_a->SetBinContent(i-(5*nbins_a)-1, result_a->GetBinContent(i)); 
    else if(i<(7*nbins_a)+2) resultBin6_a->SetBinContent(i-(6*nbins_a)-1, result_a->GetBinContent(i));
    else std::cout << "WARNING: seem to have overran the bins ...; iterator is: " << i << std::endl; 
  }

  for(int i=2; i<7*nbins_tp+2;i++){ // i starts at 2 to deal with the under/over-flow bin being bin 1 (and not -1){
    if(i<nbins_tp+2) resultBin0_tp->SetBinContent(i-1, result_tp->GetBinContent(i)); 
    else if(i<(2*nbins_tp)+2) resultBin1_tp->SetBinContent(i-(1*nbins_tp)-1, result_tp->GetBinContent(i)); 
    else if(i<(3*nbins_tp)+2) resultBin2_tp->SetBinContent(i-(2*nbins_tp)-1, result_tp->GetBinContent(i)); 
    else if(i<(4*nbins_tp)+2) resultBin3_tp->SetBinContent(i-(3*nbins_tp)-1, result_tp->GetBinContent(i)); 
    else if(i<(5*nbins_tp)+2) resultBin4_tp->SetBinContent(i-(4*nbins_tp)-1, result_tp->GetBinContent(i)); 
    else if(i<(6*nbins_tp)+2) resultBin5_tp->SetBinContent(i-(5*nbins_tp)-1, result_tp->GetBinContent(i)); 
    else if(i<(7*nbins_tp)+2) resultBin6_tp->SetBinContent(i-(6*nbins_tp)-1, result_tp->GetBinContent(i));
    else std::cout << "WARNING: seem to have overran the bins ...; iterator is: " << i << std::endl; 
  }

  for(int i=2; i<7*nbins_p+2;i++){ // i starts at 2 to deal with the under/over-flow bin being bin 1 (and not -1)
    if(i<nbins_p+2) resultBin0_p->SetBinError(i-1, result_p->GetBinError(i)); 
    else if(i<(2*nbins_p)+2) resultBin1_p->SetBinError(i-(1*nbins_p)-1, result_p->GetBinError(i)); 
    else if(i<(3*nbins_p)+2) resultBin2_p->SetBinError(i-(2*nbins_p)-1, result_p->GetBinError(i)); 
    else if(i<(4*nbins_p)+2) resultBin3_p->SetBinError(i-(3*nbins_p)-1, result_p->GetBinError(i)); 
    else if(i<(5*nbins_p)+2) resultBin4_p->SetBinError(i-(4*nbins_p)-1, result_p->GetBinError(i)); 
    else if(i<(6*nbins_p)+2) resultBin5_p->SetBinError(i-(5*nbins_p)-1, result_p->GetBinError(i)); 
    else if(i<(7*nbins_p)+2) resultBin6_p->SetBinError(i-(6*nbins_p)-1, result_p->GetBinError(i));
    else std::cout << "WARNING: seem to have overran the bins ...; iterator is: " << i << std::endl; 
  }

  for(int i=2; i<7*nbins_a+2;i++){ // i starts at 2 to deal with the under/over-flow bin being bin 1 (and not -1)
    if(i<nbins_a+2) resultBin0_a->SetBinError(i-1, result_a->GetBinError(i)); 
    else if(i<(2*nbins_a)+2) resultBin1_a->SetBinError(i-(1*nbins_a)-1, result_a->GetBinError(i)); 
    else if(i<(3*nbins_a)+2) resultBin2_a->SetBinError(i-(2*nbins_a)-1, result_a->GetBinError(i)); 
    else if(i<(4*nbins_a)+2) resultBin3_a->SetBinError(i-(3*nbins_a)-1, result_a->GetBinError(i)); 
    else if(i<(5*nbins_a)+2) resultBin4_a->SetBinError(i-(4*nbins_a)-1, result_a->GetBinError(i)); 
    else if(i<(6*nbins_a)+2) resultBin5_a->SetBinError(i-(5*nbins_a)-1, result_a->GetBinError(i)); 
    else if(i<(7*nbins_a)+2) resultBin6_a->SetBinError(i-(6*nbins_a)-1, result_a->GetBinError(i));
    else std::cout << "WARNING: seem to have overran the bins ...; iterator is: " << i << std::endl; 
  }

  for(int i=2; i<7*nbins_tp+2;i++){ // i starts at 2 to deal with the under/over-flow bin being bin 1 (and not -1){
    if(i<nbins_tp+2) resultBin0_tp->SetBinError(i-1, result_tp->GetBinError(i)); 
    else if(i<(2*nbins_tp)+2) resultBin1_tp->SetBinError(i-(1*nbins_tp)-1, result_tp->GetBinError(i)); 
    else if(i<(3*nbins_tp)+2) resultBin2_tp->SetBinError(i-(2*nbins_tp)-1, result_tp->GetBinError(i)); 
    else if(i<(4*nbins_tp)+2) resultBin3_tp->SetBinError(i-(3*nbins_tp)-1, result_tp->GetBinError(i)); 
    else if(i<(5*nbins_tp)+2) resultBin4_tp->SetBinError(i-(4*nbins_tp)-1, result_tp->GetBinError(i)); 
    else if(i<(6*nbins_tp)+2) resultBin5_tp->SetBinError(i-(5*nbins_tp)-1, result_tp->GetBinError(i)); 
    else if(i<(7*nbins_tp)+2) resultBin6_tp->SetBinError(i-(6*nbins_tp)-1, result_tp->GetBinError(i));
    else std::cout << "WARNING: seem to have overran the bins ...; iterator is: " << i << std::endl; 
  }


  resultBin0_p->Sumw2();
  resultBin1_p->Sumw2();
  resultBin2_p->Sumw2();
  resultBin3_p->Sumw2();
  resultBin4_p->Sumw2();
  resultBin5_p->Sumw2();
  resultBin6_p->Sumw2();
  resultBin0_a->Sumw2();
  resultBin1_a->Sumw2();
  resultBin2_a->Sumw2();
  resultBin3_a->Sumw2();
  resultBin4_a->Sumw2();
  resultBin5_a->Sumw2();
  resultBin6_a->Sumw2();
  resultBin0_tp->Sumw2();
  resultBin1_tp->Sumw2();
  resultBin2_tp->Sumw2();
  resultBin3_tp->Sumw2();
  resultBin4_tp->Sumw2();
  resultBin5_tp->Sumw2();
  resultBin6_tp->Sumw2();

  //resultBin0_p->Scale(1,"WIDTH");
  //resultBin1_p->Scale(1,"WIDTH");
  //resultBin2_p->Scale(1,"WIDTH");
  //resultBin3_p->Scale(1,"WIDTH");
  //resultBin4_p->Scale(1,"WIDTH");
  //resultBin5_p->Scale(1,"WIDTH");
  //resultBin6_p->Scale(1,"WIDTH");
  //resultBin0_a->Scale(1,"WIDTH");
  //resultBin1_a->Scale(1,"WIDTH");
  //resultBin2_a->Scale(1,"WIDTH");
  //resultBin3_a->Scale(1,"WIDTH");
  //resultBin4_a->Scale(1,"WIDTH");
  //resultBin5_a->Scale(1,"WIDTH");
  //resultBin6_a->Scale(1,"WIDTH");
  //resultBin0_tp->Scale(1,"WIDTH");
  //resultBin1_tp->Scale(1,"WIDTH");
  //resultBin2_tp->Scale(1,"WIDTH");
  //resultBin3_tp->Scale(1,"WIDTH");
  //resultBin4_tp->Scale(1,"WIDTH");
  //resultBin5_tp->Scale(1,"WIDTH");
  //resultBin6_tp->Scale(1,"WIDTH");


  TFile *outfile = new TFile(outFileName,"recreate");
  outfile->cd();

  result_p->Write("result_p");
  result_a->Write("result_a");
  result_tp->Write("result_tp");

  resultBin0_p->Write("resultBin0_p");
  resultBin1_p->Write("resultBin1_p");
  resultBin2_p->Write("resultBin2_p");
  resultBin3_p->Write("resultBin3_p");
  resultBin4_p->Write("resultBin4_p");
  resultBin5_p->Write("resultBin5_p");
  resultBin6_p->Write("resultBin6_p");
  resultBin0_a->Write("resultBin0_a");
  resultBin1_a->Write("resultBin1_a");
  resultBin2_a->Write("resultBin2_a");
  resultBin3_a->Write("resultBin3_a");
  resultBin4_a->Write("resultBin4_a");
  resultBin5_a->Write("resultBin5_a");
  resultBin6_a->Write("resultBin6_a");
  resultBin0_tp->Write("resultBin0_tp");
  resultBin1_tp->Write("resultBin1_tp");
  resultBin2_tp->Write("resultBin2_tp");
  resultBin3_tp->Write("resultBin3_tp");
  resultBin4_tp->Write("resultBin4_tp");
  resultBin5_tp->Write("resultBin5_tp");
  resultBin6_tp->Write("resultBin6_tp");

  cov_p->Write("cov_p");
  cov_a->Write("cov_a");
  cov_tp->Write("cov_tp");

  cor_p->Write("cor_p");
  cor_a->Write("cor_a");
  cor_tp->Write("cor_tp");

  return;
}


