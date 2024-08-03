#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <functional>
#include <numeric>
#include <TClonesArray.h>
#include <TRefArray.h>
#include <TMath.h>

#include <TSystem.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TF1.h>
#include <TGraph.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TFile.h>
#include <TChain.h>
#include <TClonesArray.h>
#include <TTree.h>
#include "TDirectory.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TString.h"
#include <TLegend.h>
#include "TLine.h"
#include "TMatrixDSym.h"
#include "TSystemDirectory.h"
#include "TList.h"

using namespace std;


void propPullStudy(const char * outFileName="propPullsStudyOut.root", const Int_t nbins=8, const char *dirname="./", const char *ext=".root")
{
  gStyle->SetOptStat(1);

  Int_t nbinsnc = nbins;
  TSystemDirectory dir(dirname, dirname);
  TList *files = dir.GetListOfFiles();
  int fileCount=0;
  TH1D* allPullHisto = new TH1D("allPullHisto","allPullHisto",nbins,0,nbins);
  TH1D* allBiasHisto = new TH1D("allBiasHisto","allBiasHisto",nbins,0,nbins);
  TH1D* pulls[100];
  TH1D* bias[100];
  TH1D* pullsInt = new TH1D("xsecIntPull","xsecIntPull",100,-10,10);
  TH1D* pullsPint = new TH1D("xsecPintPull","xsecPintPull",100,-10,10);
  TH1D* bfCoverSum = new TH1D("bfCoverSum","bfCoverSum",2,0,2);
  TH1D* coverSum = new TH1D("coverSum","coverSum",nbins,0,nbins);
  TH1D* cover = new TH1D("cover","cover",nbins,0,nbins);
  TH1D* pullsRoll[100];
  TH1D* chi2Histo = new TH1D("chi2Histo","chi2Histo",20,0,20);
  TH1D* chi2Histo_xsecBins = new TH1D("chi2Histo_xsecBins","chi2Histo_xsecBins",20,0,20);
  TH1D* chi2NDOFFit = new TH1D("chi2NDOFFit","chi2NDOFFit",6,4,10); // from NDOF is 4 to 9
  TH1D* chi2Histo_param = new TH1D("chi2Histo_param","chi2Histo_param",1000,0,1000);
  TH1D* chi2Histo_minuit = new TH1D("chi2Histo_minuit","chi2Histo_minuit",1000,0,1000);
  TH1D* chi2Histo_minuitParamComp = new TH1D("chi2Histo_minuitParamComp","chi2Histo_minuitParamComp",1000,-250,250);
  TH1D* chi2Histo_minuitParamComp_norm = new TH1D("chi2Histo_minuitParamComp_norm","chi2Histo_minuitParamComp_norm",1000,-10,10);

  for(int i=0; i<nbins; i++){
    pulls[i] = new TH1D(Form("xsecPullBin%d",i), Form("xsecPullBin%d",i), 100, -10, 10);
    bias[i] = new TH1D(Form("xsecBiasBin%d",i), Form("xsecBiasBin%d",i), 100, -10, 10);
  }
  for(int i=0; i<nbins-3; i++) pullsRoll[i] = new TH1D(Form("xsecRollPullBin%d",i), Form("xsecRollPullBin%d",i), 100, -10, 10);

  if(files){
    TSystemFile *file;
    TString fname;
    TIter next(files);
    while ((file=(TSystemFile*)next())) {
      fname = file->GetName();
      if(!file->IsDirectory() && fname.EndsWith(ext) && fname!=outFileName) {
        cout << fname.Data() << endl;
        TFile* inFile = new TFile(fname);
        if(!inFile) continue;

        TGraph* pullGr = (TGraph*)inFile->Get("pullGr");
        if(pullGr) cout << "Pulls Found" << endl;
        else continue;
        TGraph* biasGr = (TGraph*)inFile->Get("biasGr");
        if(pullGr) cout << "Bias graphs Found" << endl;
        else continue;
        TGraph* pullRollGr = (TGraph*)inFile->Get("pullRollGr");
        if(pullRollGr) cout << "Rolling Pulls Found" << endl;
        else continue;
        TGraph* pullIntGr = (TGraph*)inFile->Get("pullIntGr");
        if(pullIntGr) cout << "Integrated Pulls Found" << endl;
        else continue;
        TGraph* pullPintGr = (TGraph*)inFile->Get("pullPintGr");
        if(pullPintGr) cout << "Partially Integrated Pulls Found" << endl;
        else continue;
        TGraph* coverGr = (TGraph*)inFile->Get("coverGr");
        if(coverGr) cout << "Bin by bin coverage found" << endl;
        //else continue;
        TGraph* bfCoverGr = (TGraph*)inFile->Get("bfCoverGr");
        if(bfCoverGr) cout << "Bonferroni correction found" << endl;
        //else continue;

        //Chi2 section
        Double_t chi2 = ((TH1D*)inFile->Get("chi2"))->GetBinContent(1);
        cout << "chi2 found to be " << chi2 << endl;
        chi2Histo->Fill(chi2);

        Double_t chi2_xecBins = ((TH1D*)inFile->Get("chi2_xsecBins"))->GetBinContent(1);
        cout << "chi2_xsecBins found to be " << chi2_xecBins << endl;
        chi2Histo_xsecBins->Fill(chi2_xecBins);

        Double_t chi2_param = ((TH1D*)inFile->Get("chi2_param"))->GetBinContent(1);
        cout << "chi2_param found to be " << chi2_param << endl;
        chi2Histo_param->Fill(chi2_param);

        Double_t chi2_minuit = ((TH1D*)inFile->Get("chi2Histo_minuit"))->GetBinContent(1);
        cout << "chi2_minuit found to be " << chi2_minuit << endl;
        chi2Histo_minuit->Fill(chi2_minuit);

        Double_t chi2_minuitParamComp = ((TH1D*)inFile->Get("chi2Histo_minuitParamComp"))->GetBinContent(1);
        cout << "chi2_minuitParamComp found to be " << chi2_minuitParamComp << endl;
        chi2Histo_minuitParamComp->Fill(chi2_minuitParamComp);

        Double_t chi2_minuitParamComp_norm = chi2_minuitParamComp/chi2_minuit;
        cout << "chi2_minuitParamComp_norm found to be " << chi2_minuitParamComp_norm << endl;
        chi2Histo_minuitParamComp_norm->Fill(chi2_minuitParamComp_norm);


        if(!pullGr || !biasGr || !pullIntGr || !pullRollGr || !pullPintGr || !coverGr || !bfCoverGr){
          cout << "Cannot find all pulls in current file, skipping" << endl;
          continue;
        }
        fileCount++;
        pullsInt->Fill((pullIntGr->GetY())[0]);
        pullsPint->Fill((pullPintGr->GetY())[0]);
        bfCoverSum->Fill((bfCoverGr->GetY())[0]);
        for(int i=0; i<nbins; i++){
          pulls[i]->Fill((pullGr->GetY())[i]);
          bias[i]->Fill((biasGr->GetY())[i]);
          if(i<nbins-3) pullsRoll[i]->Fill((pullRollGr->GetY())[i]);
          double coverSumBinC = coverSum->GetBinContent(i+1);
          coverSum->SetBinContent(i+1, coverSumBinC+(coverGr->GetY())[i]);
          cover->SetBinContent(i+1, coverSumBinC+(coverGr->GetY())[i]);
        }
        inFile->Close();
        cout << fileCount << endl;
        if(fileCount>999){
          cout << "Warning, too many files, will need to increase array size." << endl;
          return;
        }
      }
    }
  }
  for(int i=0; i<nbins; i++){
    allPullHisto->SetBinContent(i+1, pulls[i]->GetMean());
    allPullHisto->SetBinError(i+1, pulls[i]->GetMeanError());
    allBiasHisto->SetBinContent(i+1, bias[i]->GetMean());
    allBiasHisto->SetBinError(i+1, bias[i]->GetMeanError());
  }
  cover->Scale(1.0/(double)fileCount);

  TFile* outFile = new TFile(outFileName, "RECREATE");
  outFile->cd();

  //Chi2 write section
  Double_t intNEvts = chi2Histo->Integral(0,20);
  TF1* chi2_4 = new TF1("chi2_4", "[0]*(exp(-x*0.5)*x^(1.0))/4",0,20);
  TF1* chi2_5 = new TF1("chi2_5", "[0]*(exp(-x*0.5)*x^(1.5))/7.52",0,20);
  TF1* chi2_6 = new TF1("chi2_6", "[0]*((exp(-(x*0.5))*(x^2))/16)",0,20);
  TF1* chi2_7 = new TF1("chi2_7", "[0]*(exp(-x*0.5)*x^(2.5))/37.6",0,20);
  TF1* chi2_8 = new TF1("chi2_8", "[0]*(exp(-x*0.5)*x^(3))/96",0,20);
  TF1* chi2_9 = new TF1("chi2_9", "[0]*(exp(-x*0.5)*x^(3.5))/263.196",0,20);
  chi2_4->SetParameter(0,intNEvts);
  chi2_5->SetParameter(0,intNEvts);
  chi2_6->SetParameter(0,intNEvts);
  chi2_7->SetParameter(0,intNEvts);
  chi2_8->SetParameter(0,intNEvts);
  chi2_9->SetParameter(0,intNEvts);
  chi2Histo->Fit(chi2_4,"R");
  chi2Histo->Fit(chi2_5,"R");
  chi2Histo->Fit(chi2_6,"R");
  chi2Histo->Fit(chi2_7,"R");
  chi2Histo->Fit(chi2_8,"R");
  chi2Histo->Fit(chi2_9,"R");
  Double_t fit_4 = chi2_4->GetChisquare();
  Double_t fit_5 = chi2_5->GetChisquare();
  Double_t fit_6 = chi2_6->GetChisquare();
  Double_t fit_7 = chi2_7->GetChisquare();
  Double_t fit_8 = chi2_8->GetChisquare();
  Double_t fit_9 = chi2_9->GetChisquare();
  chi2NDOFFit->SetBinContent(1,fit_4);
  chi2NDOFFit->SetBinContent(2,fit_5);
  chi2NDOFFit->SetBinContent(3,fit_6);
  chi2NDOFFit->SetBinContent(4,fit_7);
  chi2NDOFFit->SetBinContent(5,fit_8);
  chi2NDOFFit->SetBinContent(6,fit_9);

  TCanvas* canv = new TCanvas("chi2 NDOF comp","chi2 NDOF comp");
  chi2Histo->Draw();
  chi2_4->Draw("same");
  chi2_5->Draw("same");
  chi2_6->Draw("same");
  chi2_7->Draw("same");
  chi2_8->Draw("same");
  chi2_9->Draw("same");
  canv->Write();

  chi2Histo->Write("chi2Histo");
  chi2NDOFFit->Write("chi2NDOFFit");

  chi2Histo_xsecBins->Write("chi2Histo_xsecBins");

  chi2Histo_param->Write("chi2Histo_param");
  chi2Histo_minuit->Write("chi2Histo_minuit");
  chi2Histo_minuitParamComp->Write("chi2Histo_minuitParamComp");
  chi2Histo_minuitParamComp_norm->Write("chi2Histo_minuitParamComp_norm");

  allPullHisto->Write("xsecPullsAllBins");
  allBiasHisto->Write("xsecBiasAllBins");
  pullsInt->Write("xsecIntPull");
  pullsPint->Write("xsecPintPull");
  bfCoverSum->Write("bfCoverSum");
  coverSum->Write("coverSum");
  cover->Write("cover");
  for(int i=0; i<nbins; i++){
    pulls[i]->Write(Form("xsecPullBin%d",i)); 
    bias[i]->Write(Form("xsecBiasBin%d",i)); 
  }
  for(int i=0; i<(nbins-3); i++) pullsRoll[i]->Write(Form("xsecRollPullBin%d",i)); 
  //outFile->Close();
}

