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

void allPullsStudy(const char * outFileName="allPullsStudyOut.root", const Int_t nbins=9, const char *dirname="./", const char *ext=".root")
{
  Int_t nbinsnc = nbins;
  Int_t nccqepar=0, nfluxpar=0, ndetpar=0, nxsecpar=0, npionfsipar=0;
  TSystemDirectory dir(dirname, dirname);
  TList *files = dir.GetListOfFiles();
  Float_t params[1000][nbins];
  Float_t paramsErrs[1000][nbins];
  Float_t paramsTruth[1000][nbins];
  Float_t params_flux[1000][1000];
  Float_t paramsErrs_flux[1000][1000];
  Float_t paramsTruth_flux[1000][1000];
  Float_t params_det[1000][1000];
  Float_t paramsErrs_det[1000][1000];
  Float_t paramsTruth_det[1000][1000];
  Float_t params_xsec[1000][1000];
  Float_t paramsErrs_xsec[1000][1000];
  Float_t paramsTruth_xsec[1000][1000];
  Float_t params_PionFSI[1000][1000];
  Float_t paramsErrs_PionFSI[1000][1000];
  Float_t paramsTruth_PionFSI[1000][1000];
  int fileCount=0;
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

        TH1D* paramVals = (TH1D*)inFile->Get("paramhist_parpar_ccqe_result");
        TH1D* paramErrs = (TH1D*)inFile->Get("paramerrhist_parpar_ccqe_result");
        TH1D* paramTruth = (TH1D*)inFile->Get("paramhist_parpar_ccqe_iter0");
        if(!paramVals || !paramErrs || !paramTruth) continue;
        nccqepar = paramVals->GetEntries();
        cout << nccqepar << " CCQE params found." << endl;

        TH1D* paramVals_flux = (TH1D*)inFile->Get("paramhist_parpar_flux_result");
        TH1D* paramErrs_flux = (TH1D*)inFile->Get("paramerrhist_parpar_flux_result");
        TH1D* paramTruth_flux = (TH1D*)inFile->Get("paramhist_parpar_flux_iter0");
        if(!paramVals_flux || !paramErrs_flux || !paramTruth_flux) continue;
        nfluxpar = paramVals_flux->GetEntries();
        cout << nfluxpar << " Flux params found." << endl;

        TH1D* paramVals_det = (TH1D*)inFile->Get("paramhist_parpar_detAve_result");
        TH1D* paramErrs_det = (TH1D*)inFile->Get("paramerrhist_parpar_detAve_result");
        TH1D* paramTruth_det = (TH1D*)inFile->Get("paramhist_parpar_detAve_iter0");
        if(!paramVals_det || !paramErrs_det || !paramTruth_det) continue;
        ndetpar = paramVals_det->GetEntries();
        cout << ndetpar << " Det params found." << endl;

        TH1D* paramVals_xsec = (TH1D*)inFile->Get("paramhist_parpar_xsec_result");
        TH1D* paramErrs_xsec = (TH1D*)inFile->Get("paramerrhist_parpar_xsec_result");
        TH1D* paramTruth_xsec = (TH1D*)inFile->Get("paramhist_parpar_xsec_iter0");
        if(!paramVals_xsec || !paramErrs_xsec || !paramTruth_xsec) continue;
        nxsecpar = paramVals_xsec->GetEntries();
        cout << nxsecpar << " XSec params found." << endl;

        TH1D* paramVals_PionFSI = (TH1D*)inFile->Get("paramhist_parpar_PionFSI_result");
        TH1D* paramErrs_PionFSI = (TH1D*)inFile->Get("paramerrhist_parpar_PionFSI_result");
        TH1D* paramTruth_PionFSI = (TH1D*)inFile->Get("paramhist_parpar_PionFSI_iter0");
        if(!paramVals_PionFSI || !paramErrs_PionFSI || !paramTruth_PionFSI) continue;
        npionfsipar = paramVals_PionFSI->GetEntries();
        cout << npionfsipar << " Pion FSI params found." << endl;

        for(int i=0;i<nccqepar;i++){
          params[fileCount][i]=paramVals->GetBinContent(i+1);
          paramsErrs[fileCount][i]=paramErrs->GetBinContent(i+1);
          paramsTruth[fileCount][i]=paramTruth->GetBinContent(i+1);
        }
        for(int i=0;i<nfluxpar;i++){
          params_flux[fileCount][i]=paramVals_flux->GetBinContent(i+1);
          paramsErrs_flux[fileCount][i]=paramErrs_flux->GetBinContent(i+1);
          paramsTruth_flux[fileCount][i]=paramTruth_flux->GetBinContent(i+1);
        }
        for(int i=0;i<ndetpar;i++){
          params_det[fileCount][i]=paramVals_det->GetBinContent(i+1);
          paramsErrs_det[fileCount][i]=paramErrs_det->GetBinContent(i+1);
          paramsTruth_det[fileCount][i]=paramTruth_det->GetBinContent(i+1);
        }
        for(int i=0;i<nxsecpar;i++){
          params_xsec[fileCount][i]=paramVals_xsec->GetBinContent(i+1);
          paramsErrs_xsec[fileCount][i]=paramErrs_xsec->GetBinContent(i+1);
          paramsTruth_xsec[fileCount][i]=paramTruth_xsec->GetBinContent(i+1);
        }
        for(int i=0;i<npionfsipar;i++){
          params_PionFSI[fileCount][i]=paramVals_PionFSI->GetBinContent(i+1);
          paramsErrs_PionFSI[fileCount][i]=paramErrs_PionFSI->GetBinContent(i+1);
          paramsTruth_PionFSI[fileCount][i]=paramTruth_PionFSI->GetBinContent(i+1);
        }
        inFile->Close();
        fileCount++;
        cout << fileCount << endl;
        if(fileCount>999){
          cout << "Warning, too many files, will need to increase array size." << endl;
          return;
        }
      }
    }
  }
  cout << "Total of " << fileCount << " throws" << endl;
  cout << "Value:" << endl;
  cout << "file   bin   result" << endl;
  TFile* outFile = new TFile(outFileName, "RECREATE");
  outFile->cd();
  TTree *tree = new TTree("pullStudyOut", "pullStudyOut");
  Float_t mean[100] = {};
  Float_t fitParams[100] = {};
  Float_t fitParamsErrs[100] = {};
  Float_t TruthParams[100] = {};
  Float_t fitParams_flux[100] = {};
  Float_t fitParamsErrs_flux[100] = {};
  Float_t TruthParams_flux[100] = {};
  Float_t fitParams_det[100] = {};
  Float_t fitParamsErrs_det[100] = {};
  Float_t TruthParams_det[100] = {};
  Float_t fitParams_xsec[100] = {};
  Float_t fitParamsErrs_xsec[100] = {};
  Float_t TruthParams_xsec[100] = {};
  Float_t fitParams_pionfsi[100] = {};
  Float_t fitParamsErrs_pionfsi[100] = {};
  Float_t TruthParams_pionfsi[100] = {};
  cout << "CCQE Params ..." << endl;
  Float_t Pull_ccqe[100] = {};
  cout << "Flux Params ..." << endl;
  Float_t Pull_flux[100] = {};
  cout << "Det Params ..." << endl;
  Float_t Pull_det[100] = {};
  cout << "XSec Params ..." << endl;
  Float_t Pull_xsec[100] = {};
  cout << "Pion FSI Params ..." << endl;
  Float_t Pull_pionfsi[100] = {};
  cout << "All loaded" << endl;
  Float_t mean0 = 0.0;
  Float_t mean1 = 0.0;
  Float_t cumean[100] = {};
  int incCount[100] = {};
  tree->Branch("nbins", &nbinsnc, "nbins/I");
  tree->Branch("mean0", &mean0, "mean0/F");
  tree->Branch("mean1", &mean1, "mean1/F");
  tree->Branch("mean", mean, "mean[nbins]/F");
  tree->Branch("fitParams", fitParams, "fitParams[nbins]/F");
  tree->Branch("fitParamsErrs", fitParamsErrs, "fitParamsErrs[nbins]/F");
  tree->Branch("TruthParams", TruthParams, "TruthParams[nbins]/F");
  tree->Branch("fitParams_flux", fitParams_flux, "fitParams_flux[100]/F");
  tree->Branch("fitParamsErrs_flux", fitParamsErrs_flux, "fitParamsErr_flux[100]/F");
  tree->Branch("TruthParams_flux", TruthParams_flux, "TruthParams_flux[100]/F");
  tree->Branch("fitParams_det", fitParams_det, "fitParams_det[100]/F");
  tree->Branch("fitParamsErrs_det", fitParamsErrs_det, "fitParamsErrs_det[100]/F");
  tree->Branch("TruthParams_det", TruthParams_det, "TruthParams_det[100]/F");
  tree->Branch("fitParams_xsec", fitParams_xsec, "fitParams_xsec[100]/F");
  tree->Branch("fitParamsErrs_xsec", fitParamsErrs_xsec, "fitParamsErrs_xsec[100]/F");
  tree->Branch("TruthParams_xsec", TruthParams_xsec, "TruthParams_xsec[100]/F");
  tree->Branch("fitParams_pionfsi", fitParams_pionfsi, "fitParams_pionfsi[100]/F");
  tree->Branch("fitParamsErrs_pionfsi", fitParamsErrs_pionfsi, "fitParamsErrs_pionfsi[100]/F");
  tree->Branch("TruthParams_pionfsi", TruthParams_pionfsi, "TruthParams_pionfsi[100]/F");
  tree->Branch("Pull_ccqe", Pull_ccqe, "Pull_ccqe[nbins]/F");
  tree->Branch("Pull_flux", Pull_flux, "Pull_flux[100]/F");
  tree->Branch("Pull_det", Pull_det, "Pull_det[100]/F");
  tree->Branch("Pull_xsec", Pull_xsec, "Pull_xsec[100]/F");
  tree->Branch("Pull_pionfsi", Pull_pionfsi, "Pull_pionfsi[100]/F");
  for(int i=0;i<fileCount;i++){
    cout << "On file: " << i << endl;
    mean0=params[i][0];
    mean1=params[i][1];
    for(int j=0;j<nccqepar;j++){
      cout << i << "  CCQEParam: " << j << "   " << params[i][j] << endl;
      if(params[i][j]<10 && params[i][j]>0){
        fitParams[j]=params[i][j];
        fitParamsErrs[j]=paramsErrs[i][j];
        TruthParams[j]=paramsTruth[i][j];
        Pull_ccqe[j]=(fitParams[j]-TruthParams[j])/fitParamsErrs[j];

        mean[j]=params[i][j];
        cumean[j]+=mean[j];
        incCount[j]++;
      }
    }
    for(int j=0;j<nfluxpar;j++){
      cout << i << "  FluxParam: " << j << "   " << params[i][j] << endl;
      fitParams_flux[j]=params_flux[i][j];
      fitParamsErrs_flux[j]=paramsErrs_flux[i][j];
      TruthParams_flux[j]=paramsTruth_flux[i][j];
      Pull_flux[j]=(fitParams_flux[j]-TruthParams_flux[j])/fitParamsErrs_flux[j];;
    }
    for(int j=0;j<ndetpar;j++){
      cout << i << "  detParam: " << j << "   " << params[i][j] << endl;
      fitParams_det[j]=params_det[i][j];
      fitParamsErrs_det[j]=paramsErrs_det[i][j];
      TruthParams_det[j]=paramsTruth_det[i][j];
      Pull_det[j]=(fitParams_det[j]-TruthParams_det[j])/fitParamsErrs_det[j];;
    }
    for(int j=0;j<nxsecpar;j++){
      cout << i << "  xsecParam: " << j << "   " << params[i][j] << endl;
      fitParams_xsec[j]=params_xsec[i][j];
      fitParamsErrs_xsec[j]=paramsErrs_xsec[i][j];
      TruthParams_xsec[j]=paramsTruth_xsec[i][j];
      Pull_xsec[j]=(fitParams_xsec[j]-TruthParams_xsec[j])/fitParamsErrs_xsec[j];;
    }
    for(int j=0;j<npionfsipar;j++){
      cout << i << "  pionFSIParam: " << j << "   " << params[i][j] << endl;
      fitParams_pionfsi[j]=params_PionFSI[i][j];
      fitParamsErrs_pionfsi[j]=paramsErrs_PionFSI[i][j];
      TruthParams_pionfsi[j]=paramsTruth_PionFSI[i][j];
      Pull_pionfsi[j]=(fitParams_pionfsi[j]-TruthParams_pionfsi[j])/fitParamsErrs_pionfsi[j];;
    }    
    tree->Fill();
  }
  for(int j=0;j<nbins;j++){mean[j]=(cumean[j]/incCount[j]);}

  Float_t stdDev[nbins];
  for(int j=0;j<nbins;j++){stdDev[nbins]=0;}

  cout << "Deviation:" << endl;
  cout << "file   bin   result" << endl;
  for(int i=0;i<fileCount;i++){
    for(int j=0;j<nbins;j++){
      cout << i << "  " << j << "   " << (params[i][j]-mean[j]) << endl;;
      if(params[i][j]<10 && params[i][j]>0){
        stdDev[j]+=(params[i][j]-mean[j])*(params[i][j]-mean[j]);
      }
    }
  }
  for(int j=0;j<nbins;j++){stdDev[j]=sqrt(stdDev[j]/incCount[j]);}

  cout << "File count was: " << fileCount << endl;
  for(int i=0;i<nbins;i++){
    cout << "For bin " << i+1 << " included count is " << incCount[i] << endl;
  }

  TH1D* paramsHist = new TH1D("paramsHist","paramsHist",nbins,0,nbins);
  for(int i=0;i<nbins;i++){
    cout << "For bin " << i+1 << " param value is " << mean[i] << " with an error of " << stdDev[i] << endl;
    paramsHist->SetBinContent(i+1, mean[i]);
    paramsHist->SetBinError(i+1, stdDev[i]);
  }
  tree->Write();
  paramsHist->Write();
  //outFile->Close();
}

