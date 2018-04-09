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

void allPullsStudy_statFluctOnly(const char * nomParamFileName, const char * outFileName="allPullsStudyOut.root", const Int_t nbins=10, const char *dirname="./", const char *ext=".root")
{
  Int_t nbinsnc = nbins;
  Int_t nccqepar=0, nfluxpar=0, ndetpar=0, nxsecpar=0, npionfsipar=0;
  TSystemDirectory dir(dirname, dirname);
  TList *files = dir.GetListOfFiles();
  Float_t params[1000][1000];
  Float_t paramsErrs[1000][1000];
  Float_t paramsTruth[1000][1000];
  Float_t params_flux[1000][1000];
  Float_t paramsErrs_flux[1000][1000];
  Float_t paramsTruth_flux[1000][1000];
  Float_t params_det[1000][1000];
  Float_t paramsErrs_det[1000][1000];
  Float_t paramsTruth_det[1000][1000];
  Float_t params_xsec[1000][1000];
  Float_t paramsErrs_xsec[1000][1000];
  Float_t paramsTruth_xsec[1000][1000];
  Float_t params_pionfsi[1000][1000];
  Float_t paramsErrs_pionfsi[1000][1000];
  Float_t paramsTruth_pionfsi[1000][1000];
  int fileCount=0;
  TFile* nomParamFile = new TFile(nomParamFileName);
  if(!nomParamFile){ 
    cout << "Can't open nominal parameters reference file, closing ..." << endl;
    return;
  }
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
        TH1D* paramTruth = (TH1D*)nomParamFile->Get("paramhist_parpar_ccqe_result");
        if(!paramVals || !paramErrs || !paramTruth) continue;
        nccqepar = paramVals->GetEntries();
        cout << nccqepar << " CCQE params found." << endl;

        TH1D* paramVals_flux = (TH1D*)inFile->Get("paramhist_parpar_flux_result");
        TH1D* paramErrs_flux = (TH1D*)inFile->Get("paramerrhist_parpar_flux_result");
        TH1D* paramTruth_flux = (TH1D*)nomParamFile->Get("paramhist_parpar_flux_result");
        if(!paramVals_flux || !paramErrs_flux || !paramTruth_flux) continue;
        nfluxpar = paramVals_flux->GetEntries();
        cout << nfluxpar << " Flux params found." << endl;

        TH1D* paramVals_det = (TH1D*)inFile->Get("paramhist_parpar_detAve_result");
        TH1D* paramErrs_det = (TH1D*)inFile->Get("paramerrhist_parpar_detAve_result");
        TH1D* paramTruth_det = (TH1D*)nomParamFile->Get("paramhist_parpar_detAve_result");
        if(!paramVals_det || !paramErrs_det || !paramTruth_det) continue;
        ndetpar = paramVals_det->GetEntries();
        cout << ndetpar << " Det params found." << endl;

        TH1D* paramVals_xsec = (TH1D*)inFile->Get("paramhist_parpar_xsec_result");
        TH1D* paramErrs_xsec = (TH1D*)inFile->Get("paramerrhist_parpar_xsec_result");
        TH1D* paramTruth_xsec = (TH1D*)nomParamFile->Get("paramhist_parpar_xsec_result");
        if(!paramVals_xsec || !paramErrs_xsec || !paramTruth_xsec) continue;
        nxsecpar = paramVals_xsec->GetEntries();
        cout << nxsecpar << " XSec params found." << endl;

        TH1D* paramVals_pionfsi = (TH1D*)inFile->Get("paramhist_parpar_PionFSI_result");
        TH1D* paramErrs_pionfsi = (TH1D*)inFile->Get("paramerrhist_parpar_PionFSI_result");
        TH1D* paramTruth_pionfsi = (TH1D*)nomParamFile->Get("paramhist_parpar_PionFSI_result");
        if(!paramVals_pionfsi || !paramErrs_pionfsi || !paramTruth_pionfsi) continue;
        npionfsipar = paramVals_pionfsi->GetEntries();
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
          params_pionfsi[fileCount][i]=paramVals_pionfsi->GetBinContent(i+1);
          paramsErrs_pionfsi[fileCount][i]=paramErrs_pionfsi->GetBinContent(i+1);
          paramsTruth_pionfsi[fileCount][i]=paramTruth_pionfsi->GetBinContent(i+1);
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
  Float_t mean[100] = {};
  Float_t cumean[100] = {};
  int incCount[100] = {};
  Float_t stdDev[100] = {};
  Float_t mean_flux[100] = {};
  Float_t cumean_flux[100] = {};
  int incCount_flux[100] = {};
  Float_t stdDev_flux[100] = {};
  Float_t mean_det[100] = {};
  Float_t cumean_det[100] = {};
  int incCount_det[100] = {};
  Float_t stdDev_det[100] = {};
  Float_t mean_xsec[100] = {};
  Float_t cumean_xsec[100] = {};
  int incCount_xsec[100] = {};
  Float_t stdDev_xsec[100] = {};
  Float_t mean_pionfsi[100] = {};
  Float_t cumean_pionfsi[100] = {};
  int incCount_pionfsi[100] = {};
  Float_t stdDev_pionfsi[100] = {};
  cout << "All Arrays Initialised" << endl;


  tree->Branch("nbins", &nbinsnc, "nbins/I");
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
  cout << "Tree branches set" << endl;

  cout << "Starting loop over files, total number is " << fileCount << endl;

  for(int i=0;i<fileCount;i++){
    cout << "On file: " << i << endl;
    for(int j=0;j<nccqepar;j++){
      cout << i << "  CCQEParam: " << j << "   " << params[i][j] << endl;
      if(params[i][j]<100 && params[i][j]>0.00000001){
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
      cout << i << "  FluxParam: " << j << "   " << params_flux[i][j] << endl;
      fitParams_flux[j]=params_flux[i][j];
      fitParamsErrs_flux[j]=paramsErrs_flux[i][j];
      TruthParams_flux[j]=paramsTruth_flux[i][j];
      Pull_flux[j]=(fitParams_flux[j]-TruthParams_flux[j])/fitParamsErrs_flux[j];

      mean_flux[j]=params_flux[i][j];
      cumean_flux[j]+=mean_flux[j];
      incCount_flux[j]++;
    }
    for(int j=0;j<ndetpar;j++){
      cout << i << "  detParam: " << j << "   " << params_det[i][j] << endl;
      fitParams_det[j]=params_det[i][j];
      fitParamsErrs_det[j]=paramsErrs_det[i][j];
      TruthParams_det[j]=paramsTruth_det[i][j];
      Pull_det[j]=(fitParams_det[j]-TruthParams_det[j])/fitParamsErrs_det[j];

      mean_det[j]=params_det[i][j];
      cumean_det[j]+=mean_det[j];
      incCount_det[j]++;
    }
    for(int j=0;j<nxsecpar;j++){
      cout << i << "  xsecParam: " << j << "   " << params_xsec[i][j] << endl;
      fitParams_xsec[j]=params_xsec[i][j];
      fitParamsErrs_xsec[j]=paramsErrs_xsec[i][j];
      TruthParams_xsec[j]=paramsTruth_xsec[i][j];
      Pull_xsec[j]=(fitParams_xsec[j]-TruthParams_xsec[j])/fitParamsErrs_xsec[j];

      mean_xsec[j]=params_xsec[i][j];
      cumean_xsec[j]+=mean_xsec[j];
      incCount_xsec[j]++;
    }
    for(int j=0;j<npionfsipar;j++){
      cout << i << "  pionFSIParam: " << j << "   " << params_pionfsi[i][j] << endl;
      fitParams_pionfsi[j]=params_pionfsi[i][j];
      fitParamsErrs_pionfsi[j]=paramsErrs_pionfsi[i][j];
      TruthParams_pionfsi[j]=paramsTruth_pionfsi[i][j];
      Pull_pionfsi[j]=(fitParams_pionfsi[j]-TruthParams_pionfsi[j])/fitParamsErrs_pionfsi[j];

      mean_pionfsi[j]=params_pionfsi[i][j];
      cumean_pionfsi[j]+=mean_pionfsi[j];
      incCount_pionfsi[j]++;
    }    
    tree->Fill();
  }
  for(int j=0;j<nccqepar;j++){mean[j]=(cumean[j]/incCount[j]);}
  for(int j=0;j<nfluxpar;j++){mean_flux[j]=(cumean_flux[j]/incCount_flux[j]);}
  for(int j=0;j<ndetpar;j++){mean_det[j]=(cumean_det[j]/incCount_det[j]);}
  for(int j=0;j<nxsecpar;j++){mean_xsec[j]=(cumean_xsec[j]/incCount_xsec[j]);}
  for(int j=0;j<npionfsipar;j++){mean_pionfsi[j]=(cumean_pionfsi[j]/incCount_pionfsi[j]);}

  //for(int j=0;j<nccqepar;j++){stdDev[j]=0;}

  // cout << "Deviation:" << endl;
  // cout << "file   bin   result" << endl;
  for(int i=0;i<fileCount;i++){
    for(int j=0;j<nccqepar;j++){
      //cout << i << "  " << j << "   " << (params[i][j]-mean[j]) << endl;;
      if(params[i][j]<100 && params[i][j]>0.00000001){
        stdDev[j]+=(params[i][j]-mean[j])*(params[i][j]-mean[j]);
      }
    }
    for(int j=0;j<nfluxpar;j++){stdDev_flux[j]+=(params_flux[i][j]-mean_flux[j])*(params_flux[i][j]-mean_flux[j]);}
    for(int j=0;j<ndetpar;j++){stdDev_det[j]+=(params_det[i][j]-mean_det[j])*(params_det[i][j]-mean_det[j]);}
    for(int j=0;j<nxsecpar;j++){stdDev_xsec[j]+=(params_xsec[i][j]-mean_xsec[j])*(params_xsec[i][j]-mean_xsec[j]);}
    for(int j=0;j<npionfsipar;j++){stdDev_pionfsi[j]+=(params_pionfsi[i][j]-mean_pionfsi[j])*(params_pionfsi[i][j]-mean_pionfsi[j]);}    
  }

  for(int j=0;j<nccqepar;j++){stdDev[j]=sqrt(stdDev[j]/incCount[j]);}
  for(int j=0;j<nfluxpar;j++){stdDev_flux[j]=sqrt(stdDev_flux[j]/incCount_flux[j]);}
  for(int j=0;j<ndetpar;j++){stdDev_det[j]=sqrt(stdDev_det[j]/incCount_det[j]);}
  for(int j=0;j<nxsecpar;j++){stdDev_xsec[j]=sqrt(stdDev_xsec[j]/incCount_xsec[j]);}
  for(int j=0;j<npionfsipar;j++){stdDev_pionfsi[j]=sqrt(stdDev_pionfsi[j]/incCount_pionfsi[j]);}

  // cout << "File count was: " << fileCount << endl;
  // for(int i=0;i<nccqepar;i++){
  //   cout << "For bin " << i+1 << " included count is " << incCount[i] << endl;
  // }

  TH1D* paramsHist = new TH1D("paramsHist","paramsHist",nccqepar,0,nccqepar);
  TH1D* paramsHist_flux = new TH1D("paramsHist_flux","paramsHist_flux",nfluxpar,0,nfluxpar);
  TH1D* paramsHist_det = new TH1D("paramsHist_det","paramsHist_det",ndetpar,0,ndetpar);
  TH1D* paramsHist_xsec = new TH1D("paramsHist_xsec","paramsHist_xsec",nxsecpar,0,nxsecpar);
  TH1D* paramsHist_pionfsi = new TH1D("paramsHist_pionfsi","paramsHist_pionfsi",npionfsipar,0,npionfsipar);

  TH1D* paramsHist_truth = new TH1D("paramsHist_truth","paramsHist_truth",nccqepar,0,nccqepar);
  TH1D* paramsHist_flux_truth = new TH1D("paramsHist_flux_truth","paramsHist_flux_truth",nfluxpar,0,nfluxpar);
  TH1D* paramsHist_det_truth = new TH1D("paramsHist_det_truth","paramsHist_det_truth",ndetpar,0,ndetpar);
  TH1D* paramsHist_xsec_truth = new TH1D("paramsHist_xsec_truth","paramsHist_xsec_truth",nxsecpar,0,nxsecpar);
  TH1D* paramsHist_pionfsi_truth = new TH1D("paramsHist_pionfsi_truth","paramsHist_pionfsi_truth",npionfsipar,0,npionfsipar);

  TH1D* pullsHist = new TH1D("pullsHist_ccqe","pullsHist_ccqe",nccqepar,0,nccqepar);
  TH1D* pullsHist_flux = new TH1D("pullsHist_flux","pullsHist_flux",nfluxpar,0,nfluxpar);
  TH1D* pullsHist_det = new TH1D("pullsHist_det","pullsHist_det",ndetpar,0,ndetpar);
  TH1D* pullsHist_xsec = new TH1D("pullsHist_xsec","pullsHist_xsec",nxsecpar,0,nxsecpar);
  TH1D* pullsHist_pionfsi = new TH1D("pullsHist_pionfsi","pullsHist_pionfsi",npionfsipar,0,npionfsipar);
  for(int i=0;i<nccqepar;i++){
    cout << "For bin " << i+1 << " param value is " << mean[i] << " with an error of " << stdDev[i] << endl;
    if(stdDev[i]<0.0000001) stdDev[i]=0.0000001;
    paramsHist->SetBinContent(i+1, mean[i]);
    paramsHist->SetBinError(i+1, stdDev[i]);
    paramsHist_truth->SetBinContent(i+1, paramsTruth[0][i]);
    pullsHist->SetBinContent(i+1, (mean[i]-paramsTruth[0][i])/stdDev[i]);
    if(sqrt((mean[i]-paramsTruth[0][i])*(mean[i]-paramsTruth[0][i]))<0.0000001) pullsHist->SetBinContent(i+1, 0.0);
  }
  for(int i=0;i<nfluxpar;i++){
    if(stdDev_flux[i]<0.0000001) stdDev_flux[i]=0.0000001;
    paramsHist_flux->SetBinContent(i+1, mean_flux[i]);
    paramsHist_flux->SetBinError(i+1, stdDev_flux[i]);
    paramsHist_flux_truth->SetBinContent(i+1, paramsTruth_flux[0][i]);
    pullsHist_flux->SetBinContent(i+1, (mean_flux[i]-paramsTruth_flux[0][i])/stdDev_flux[i]);
    if(sqrt((mean_flux[i]-paramsTruth_flux[0][i])*(mean_flux[i]-paramsTruth_flux[0][i]))<0.0000001) pullsHist_flux->SetBinContent(i+1, 0.0);
  }
  for(int i=0;i<ndetpar;i++){
    if(stdDev_det[i]<0.0000001) stdDev_det[i]=0.0000001;
    paramsHist_det->SetBinContent(i+1, mean_det[i]);
    paramsHist_det->SetBinError(i+1, stdDev_det[i]);
    paramsHist_det_truth->SetBinContent(i+1, paramsTruth_det[0][i]);
    pullsHist_det->SetBinContent(i+1, (mean_det[i]-paramsTruth_det[0][i])/stdDev_det[i]);
    if(sqrt((mean_det[i]-paramsTruth_det[0][i])*(mean_det[i]-paramsTruth_det[0][i]))<0.0000001) pullsHist_det->SetBinContent(i+1, 0.0);
  }
  for(int i=0;i<nxsecpar;i++){
    if(stdDev_xsec[i]<0.0000001) stdDev_xsec[i]=0.0000001;
    paramsHist_xsec->SetBinContent(i+1, mean_xsec[i]);
    paramsHist_xsec->SetBinError(i+1, stdDev_xsec[i]);
    paramsHist_xsec_truth->SetBinContent(i+1, paramsTruth_xsec[0][i]);
    pullsHist_xsec->SetBinContent(i+1, (mean_xsec[i]-paramsTruth_xsec[0][i])/stdDev_xsec[i]);
    if(sqrt((mean_xsec[i]-paramsTruth_xsec[0][i])*(mean_xsec[i]-paramsTruth_xsec[0][i]))<0.0000001) pullsHist_xsec->SetBinContent(i+1, 0.0);
  }
  for(int i=0;i<npionfsipar;i++){
    if(stdDev_pionfsi[i]<0.0000001) stdDev_pionfsi[i]=0.0000001;
    paramsHist_pionfsi->SetBinContent(i+1, mean_pionfsi[i]);
    paramsHist_pionfsi->SetBinError(i+1, stdDev_pionfsi[i]);
    paramsHist_pionfsi_truth->SetBinContent(i+1, paramsTruth_pionfsi[0][i]);
    pullsHist_pionfsi->SetBinContent(i+1, (mean_pionfsi[i]-paramsTruth_pionfsi[0][i])/stdDev_pionfsi[i]);
    if(sqrt((mean_pionfsi[i]-paramsTruth_pionfsi[0][i])*(mean_pionfsi[i]-paramsTruth_pionfsi[0][i]))<0.0000001) pullsHist_pionfsi->SetBinContent(i+1, 0.0);
  }

  tree->Write();
  paramsHist_truth->Write();
  paramsHist_flux_truth->Write();
  paramsHist_det_truth->Write();
  paramsHist_xsec_truth->Write();
  paramsHist_pionfsi_truth->Write();

  paramsHist->Write();
  paramsHist_flux->Write();
  paramsHist_det->Write();
  paramsHist_xsec->Write();
  paramsHist_pionfsi->Write();

  pullsHist->Write();
  pullsHist_flux->Write();
  pullsHist_det->Write();
  pullsHist_xsec->Write();
  pullsHist_pionfsi->Write();
  outFile->Close();
  return;
}

