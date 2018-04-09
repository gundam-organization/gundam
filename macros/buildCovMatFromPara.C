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

void buildCovMatFromPara(const char * outFileName="covMatFromParaOut.root",  const char *dirname="./", const char *ext=".root")
{
  TSystemDirectory dir(dirname, dirname);
  TList *files = dir.GetListOfFiles();
  int fileCount=0;
  int finalIter=0;

  const int ntoys = 500;
  const int nusedbranches = 6;
  const int nbins=9;

  const int nbranchbins = nusedbranches*nbins;

  TMatrixDSym covar(nbranchbins);
  TMatrixDSym covar_mean(nbranchbins);
  TMatrixDSym covar_norm(nbranchbins);
  TMatrixDSym covar_mean_norm(nbranchbins);


  TH1D* binSpreadHist[nbranchbins];
  for(int b=0;b<nbranchbins;b++){
    binSpreadHist[b] = new TH1D(Form("binSpreadHist%d_distribution", b), Form("binSpreadHist%d_distribution", b), 11000, -1000, 10000); 
    for(int c=0;c<nbranchbins;c++){
      covar[b][c]=0;
      covar_mean[b][c]=0;
      covar_norm[b][c]=0;
      covar_mean_norm[b][c]=0;
    }
  }
  

  TH1D* nomhists[nusedbranches];
  TH1D* toyhists[nusedbranches][ntoys];
  TH1D* toyHistNow;
  TH1D* nomHistNow;


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
        for(int b=0;b<nbranchbins;b++){
          binSpreadHist[b]->Add((TH1D*)inFile->Get(Form("binSpreadHist%d_distribution", b)));
        }  
        for(int b=0;b<nusedbranches;b++){
          toyHistNow = (TH1D*)inFile->Get(Form("toyhist%d", b));
          toyhists[b][fileCount] = new TH1D(*toyHistNow);
          toyhists[b][fileCount]->SetDirectory(0);
          //toyhists[b][fileCount]->Print();
          if(fileCount==0){
            nomHistNow = (TH1D*)inFile->Get(Form("nomhist%d", b));
            nomhists[b] = new TH1D(*nomHistNow);
            nomhists[b]->SetDirectory(0);
          }
        }
        inFile->Close();
        fileCount++;
        cout << fileCount << endl;
        if(fileCount>ntoys){
          cout << "Warning, too many files, will need to increase hardcoded ntoys." << endl;
          return;
        }
      }
    }
  }
  TFile* outFile = new TFile(outFileName, "RECREATE");
  outFile->cd();

  cout << "Files successfully read, making covar matricies." << endl;

  for(Int_t t=0; t<ntoys; t++){
      for(Int_t ibr=0; ibr<nusedbranches; ibr++){
          for(Int_t jbr=0; jbr<nusedbranches; jbr++){
              for(Int_t i=0; i<nbins; i++){
                  for(Int_t j=0; j<nbins; j++){
                      Int_t iindex = (nbins*ibr)+i; 
                      Int_t jindex = (nbins*jbr)+j; 
                      //cout << " On toy, i, j: " << t << ", " << iindex << ", " << jindex << endl;
                      //toyhists[t][ibr]->Print("all");
                      //nomhists[ibr]->Print("all");
                      //binSpreadHist[iindex]->Print();
                      covar[iindex][jindex]+=(toyhists[ibr][t]->GetBinContent(i+1)-nomhists[ibr]->GetBinContent(i+1))*(toyhists[jbr][t]->GetBinContent(j+1)-nomhists[jbr]->GetBinContent(j+1)); 
                      covar_mean[iindex][jindex]+=(toyhists[ibr][t]->GetBinContent(i+1)-binSpreadHist[iindex]->GetMean())*(toyhists[jbr][t]->GetBinContent(j+1)-binSpreadHist[jindex]->GetMean()); 
                      //if(iindex==jindex && (iindex==44 || iindex==35)) cout << "toyhists[ibr][t]->GetBinContent(i+1):" << toyhists[ibr][t]->GetBinContent(i+1) << endl << "nomhists[ibr]->GetBinContent(i+1)" << nomhists[ibr]->GetBinContent(i+1) << endl;
                  }
              }
          }
      }
  }


  for(Int_t ibr=0; ibr<nusedbranches; ibr++){
      for(Int_t jbr=0; jbr<nusedbranches; jbr++){
          for(Int_t i=0; i<nbins; i++){
              for(Int_t j=0; j<nbins; j++){
                  Int_t iindex = (nbins*ibr)+i; 
                  Int_t jindex = (nbins*jbr)+j; 
                  covar_norm[iindex][jindex]+=covar[iindex][jindex]/(nomhists[ibr]->GetBinContent(i+1)*nomhists[jbr]->GetBinContent(j+1)); 
                  covar_mean_norm[iindex][jindex]+=covar_mean[iindex][jindex]/(binSpreadHist[iindex]->GetMean()*binSpreadHist[jindex]->GetMean()); 
              }
          }
      }
  }

  covar*=1.0/(Float_t)ntoys;
  covar_mean*=1.0/(Float_t)ntoys;

  covar_norm*=1.0/(Float_t)ntoys;
  covar_mean_norm*=1.0/(Float_t)ntoys;

  covar.Print();
  covar_norm.Print();
  covar.Write("covMat");
  covar_norm.Write("covMat_norm");

  covar_mean.Write("covMat_mean");
  covar_mean_norm.Write("covMat_mean_norm");

  for(Int_t ibr=0; ibr<nusedbranches; ibr++){ nomhists[ibr]->Write(); }

  for(int b=0;b<(nbins*(nusedbranches));b++){
    binSpreadHist[b]->Write(); 
  }

  //outFile->Close();
}

