/******************************************************

Code to convert a rootracker tree into the format output
by the generators.

Author: Stephen Dolan
Date Created: November 2015

******************************************************/

#if !defined(__CINT__) || defined(__MAKECINT__)
#endif

#include <iostream> 
#include <iomanip>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <assert.h>

#include <TClonesArray.h>
#include <TRefArray.h>
#include <TMath.h>

#include <TSystem.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TF1.h>
#include <TH1.h>
#include <TH1F.h>
#include <TH2F.h>
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


//#include "DataClasses.hxx"
//#include "InputConverter.hxx"
#include "/data/t2k/dolan/particleGun/nd280AnalysisTools/v1r9p3/amd64_linux26/AnalysisTools/libReadoaAnalysis/libReadoaAnalysisProjectHeaders.h"

using namespace std;


int treeConvert_rooTracker(TString inFileName, TString outFileName, Long64_t ncustomentries=0)
{

  gROOT->SetBatch(1);
  gROOT->ProcessLine(".x  /data/t2k/dolan/particleGun/nd280AnalysisTools/v1r9p3/AnalysisTools/oaAnalysisReadInitFile-amd64_linux26.C");  

  //TFile *infile = new TFile(inFileName);
  //TTree *intree = (TTree*)infile->Get("NRooTrackerVtx");

  TFile *outfile = new TFile(outFileName,"recreate");
  TTree *outtree = new TTree("nRooTracker", "nRooTracker");

  TChain *intree = new TChain("TruthDir/NRooTrackerVtx");

  // Get list of files to run over. 
  std::ifstream inputFile(inFileName.Data(), ios::in);

  // Check if the file exists.
  if (!inputFile.is_open())
  {
    std::cout << "ERROR: File list not found!" << std::endl;
    std::cout << " - File should contain list of files to be processed." << std::endl;
    return(1);
  }
  else
  {
    std::string curFileName;
    // Add the input files to the TChains.
    while(getline(inputFile,curFileName))
    {
        intree->Add(curFileName.c_str());
    }
  }

  //intree->Print();

  TObjString* NeutReacCode = 0;
  Int_t NStdHepN;
  Int_t NStdHepPdg[100];
  Int_t NStdHepStatus[100];
  Double_t EvtXSec;
  Double_t EvtWght;
  Double_t NStdHepP4[100][4];

  //intree->SetBranchAddress("Vtx.EvtCode", &NeutReacCode);
  //intree->SetBranchAddress("Vtx.StdHepN", &NStdHepN);
  //intree->SetBranchAddress("Vtx.StdHepPdg", NStdHepPdg);
  //intree->SetBranchAddress("Vtx.StdHepP4[100][4]", NStdHepP4);
  //intree->SetBranchAddress("Vtx.StdHepStatus", NStdHepStatus);

  Int_t NNVtx;
  TClonesArray* NVtx = new TClonesArray("ND::NRooTrackerVtx",100);

  intree->SetBranchAddress("NVtx",&NNVtx);
  intree->SetBranchAddress("Vtx",&NVtx);

  outtree->Branch("EvtCode", "TObjString", &NeutReacCode);
  outtree->Branch("StdHepN", &NStdHepN, "NStdHepN/I");

  outtree->Branch("StdHepPdg", NStdHepPdg, "NStdHepPdg[100]/I");
  outtree->Branch("StdHepP4", NStdHepP4, "NStdHepP4[100][4]/D");
  outtree->Branch("StdHepStatus", NStdHepStatus, "NStdHepStatus[100]/I");

  outtree->Branch("EvtXSec", &EvtXSec, "EvtXSec/D");
  outtree->Branch("EvtWght", &EvtWght, "EvtWght/D");

  ND::NRooTrackerVtx *lvtx = NULL;
  Long64_t nentries = intree->GetEntriesFast();
  if(ncustomentries!=0) nentries=ncustomentries;
  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    nb = intree->GetEntry(jentry); nbytes += nb;
    for (int roov = 0; roov < NNVtx; roov++) {
      lvtx = (ND::NRooTrackerVtx*) (NVtx->At(roov));
      NStdHepN = lvtx->StdHepN;
      //NStdHepPdg =  lvtx->StdHepPdg;
      //NStdHepStatus =  lvtx->StdHepStatus;
      copy(&(lvtx->StdHepPdg[0]),&(lvtx->StdHepPdg[0])+100, &(NStdHepPdg[0]));
      copy(&(lvtx->StdHepStatus[0]),&(lvtx->StdHepStatus[0])+100, &(NStdHepStatus[0]));
      copy(&(lvtx->StdHepP4[0][0]),&(lvtx->StdHepP4[0][0])+4*100, &(NStdHepP4[0][0]));
      NeutReacCode = lvtx->EvtCode;
      EvtXSec = lvtx->EvtXSec;
      EvtWght = lvtx->EvtWght;
      //cout << lvtx <<endl;
      //cout << lvtx->StdHepN <<endl;
      //cout << lvtx->StdHepPdg[1] <<endl;
      //cout << lvtx->StdHepP4[0][1] <<endl;
      outtree->Fill();
    }
    if(jentry%10000==0){
      cout << "Processing entry " << jentry << endl;
      cout << "Info for final event (" << NNVtx << ") in entry: " << endl;
      cout << "EvtCode is " << NeutReacCode->String().Data() << endl;
      cout << "NStdHepN is " << NStdHepN << endl;
      cout << "EvtXSec is " << EvtXSec << endl;
      cout << "EvtWght is " << EvtWght << endl;
      cout << "StdHepPdg[0] is " << NStdHepPdg[0] << endl;
      cout << "StdHepPdg[5] is " << NStdHepPdg[5] << endl;
      cout << "NStdHepStatus[0] is " << NStdHepStatus[0] << endl;
      cout << "NStdHepStatus[5] is " << NStdHepStatus[5] << endl;
      cout << "NStdHepP4[0][3] is " << NStdHepP4[0][3] << endl;
      cout << "NStdHepP4[5][3] is " << NStdHepP4[5][3] << endl << endl;
    }
  }

  
  printf("***Output Rec Tree: ***\n");
  outtree->Print();
  outfile->Write();

  //delete infile;
  delete outfile;
  return 0;
}
