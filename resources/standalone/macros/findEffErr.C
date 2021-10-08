/******************************************************

Code to calculate error on eff correction...

Author: Stephen Dolan
Date Created: May 2016

******************************************************/


#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <functional>
#include <numeric>
#include <assert.h>

#include <TCanvas.h>
#include <TH1F.h>
#include <TTree.h>
#include <TString.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TMath.h>
#include <TArrayF.h>
#include "TMatrixDSym.h"


using namespace std;

// Float_t dialLow = 0.93;
// Float_t dialHigh = 0.97;
//Float_t dialLow = -10;
//Float_t dialHigh = 10;
bool extraCuts=false;
bool psdim=true;
int  truncWeight = 10;
//can find weights and HL2 here:
//mt="/data/t2k/dolan/xsToolBasedMEC/CC0PiAnl/microtrees/mar16/job_NeutAirAllSystV5_out/allMerged.root"
//wd="/data/t2k/dolan/xsToolBasedMEC/CC0PiAnl/weights/outdir/cc0piv27Mar16NeutAirV2/"

int findEffErr(TString hl2InFileName, TString weightsInFileName, Int_t ntoys, TString outFileName, bool protonFSIOnly=false, 
               Float_t dialLow=-100, Float_t dialHigh=100, bool protonFSISystPlot=false)
{
  // You need to provide the number of branches in your HL2 tree
  // And the accum_level you want to cut each one at to get your selected events
  // i.e choosing n in accum_level[0][branch_i]>n
  const int nbranches = 8; //10
  //const int accumToCut[nbranches] =   {7,8,9,8,7,5,4,7,8,7};
  const int accumToCut[nbranches] =   {7,8,9,8,7,5,4,7};

  TFile *hl2Infile = new TFile(hl2InFileName);
  TTree *hl2Intree = (TTree*)hl2Infile->Get("default");
  TTree *hl2Intree_T = (TTree*)hl2Infile->Get("truth");

  TFile *weightsInfile = new TFile(weightsInFileName);
  TTree *weightsXsRecTree = (TTree*)weightsInfile->Get("xs_rec");
  TTree *weightsXsTrueTree = (TTree*)weightsInfile->Get("xs_tru");
  TTree *weightsFsiRecTree = (TTree*)weightsInfile->Get("fsi_rec");
  TTree *weightsFsiTrueTree = (TTree*)weightsInfile->Get("fsi_tru");
  TTree *weightsFluxRecTree = (TTree*)weightsInfile->Get("flux_rec");
  TTree *weightsFluxTrueTree = (TTree*)weightsInfile->Get("flux_tru");
  TTree *weightsOtherRecTree = (TTree*)weightsInfile->Get("other_rec");
  TTree *weightsOtherTrueTree = (TTree*)weightsInfile->Get("other_tru");

  TFile *outfile = new TFile(outFileName,"recreate");

  // Declaration of leaf types
  Int_t          accum_level[1500][50];
  Int_t          reaction;
  Int_t          cutBranch=-999;
  Int_t          mectopology;
  Float_t        dptTrue;
  Float_t        dphitTrue;
  Float_t        dalphatTrue;
  Float_t        pMomTrue;
  Float_t        pThetaTrue;
  Float_t        muMomTrue;
  Float_t        muThetaTrue;
  Float_t        muCosThetaTrue;
  Float_t        pCosThetaTrue;
  Float_t        RecoNuEnergy=0;
  Float_t        TrueNuEnergy=0;
  Float_t        weight;

  Int_t          accum_level_T[50];
  Int_t          reaction_T;
  Int_t          mectopology_T;
  Float_t        dptTrue_T;
  Float_t        dphitTrue_T;
  Float_t        dalphatTrue_T;
  Float_t        muMomTrue_T;
  Float_t        pMomTrue_T;
  Float_t        muCosThetaTrue_T;
  Float_t        pCosThetaTrue_T;
  Float_t        TrueNuEnergy_T;

  Float_t        TrueNuEnergy_RW;
  Float_t        TrueNuEnergy_TW;


  // Float_t        xsRecWeight[1000];
  // Float_t        xsTrueWeight[1000];


  hl2Intree->SetBranchAddress("accum_level", &accum_level);
  hl2Intree->SetBranchAddress("reaction", &reaction);
  hl2Intree->SetBranchAddress("mectopology", &mectopology);
  hl2Intree->SetBranchAddress("trueDpT", &dptTrue);
  hl2Intree->SetBranchAddress("trueDphiT", &dphitTrue);
  hl2Intree->SetBranchAddress("trueDalphaT", &dalphatTrue);
  hl2Intree->SetBranchAddress("truep_truemom" ,&pMomTrue);
  hl2Intree->SetBranchAddress("truep_truecostheta" ,&pCosThetaTrue);
  hl2Intree->SetBranchAddress("truemu_mom", &muMomTrue);
  hl2Intree->SetBranchAddress("truemu_costheta", &muCosThetaTrue);
  hl2Intree->SetBranchAddress("nu_trueE", &TrueNuEnergy);
  hl2Intree->SetBranchAddress("weight", &weight);

  hl2Intree_T->SetBranchAddress("accum_level", &accum_level_T);
  hl2Intree_T->SetBranchAddress("reaction", &reaction_T);
  hl2Intree_T->SetBranchAddress("mectopology", &mectopology_T);
  hl2Intree_T->SetBranchAddress("trueDpT", &dptTrue_T);
  hl2Intree_T->SetBranchAddress("trueDphiT", &dphitTrue_T);
  hl2Intree_T->SetBranchAddress("trueDalphaT", &dalphatTrue_T);
  hl2Intree_T->SetBranchAddress("truep_truemom" ,&pMomTrue_T);
  hl2Intree_T->SetBranchAddress("truep_truecostheta" ,&pCosThetaTrue_T);
  hl2Intree_T->SetBranchAddress("truemu_mom", &muMomTrue_T);
  hl2Intree_T->SetBranchAddress("truemu_costheta", &muCosThetaTrue_T);
  hl2Intree_T->SetBranchAddress("nu_trueE", &TrueNuEnergy_T);

  TArrayF *weightsXsecRec = new TArrayF(500);
  weightsXsecRec=0;
  TBranch *b_weightsXsecRec;
  TBranch *b_TrueNuEnergy_RW_XsecRec;
  if(!protonFSIOnly){
    weightsXsRecTree->SetBranchAddress("nuEnergy", &TrueNuEnergy_RW, &b_TrueNuEnergy_RW_XsecRec);
    weightsXsRecTree->SetBranchAddress("weights", &weightsXsecRec, &b_weightsXsecRec);
  }

  TArrayF *weightsXsecTrue = new TArrayF(500);
  weightsXsecTrue=0;
  TBranch *b_weightsXsecTrue;
  TBranch *b_TrueNuEnergy_TW_XsecTrue;
  if(!protonFSIOnly){
    weightsXsTrueTree->SetBranchAddress("nuEnergy", &TrueNuEnergy_TW, &b_TrueNuEnergy_TW_XsecTrue);
    weightsXsTrueTree->SetBranchAddress("weights", &weightsXsecTrue, &b_weightsXsecTrue);
  }

  TArrayF *weightsFsiRec = new TArrayF(500);
  weightsFsiRec=0;
  TBranch *b_weightsFsiRec;
  //weightsFsiRecTree->SetBranchAddress("weights", &weightsFsiRec, &b_weightsFsiRec);
  //if(protonFSIOnly) weightsFsiRecTree->SetBranchAddress("nuEnergy", &TrueNuEnergy_RW, &b_TrueNuEnergy_RW_XsecRec);
  if(!protonFSIOnly){
    weightsFsiRecTree->SetBranchAddress("weights", &weightsFsiRec, &b_weightsFsiRec);
  }

  TArrayF *weightsFsiTrue = new TArrayF(500);
  weightsFsiTrue=0;
  TBranch *b_weightsFsiTrue;
  //weightsFsiTrueTree->SetBranchAddress("weights", &weightsFsiTrue, &b_weightsFsiTrue);
  //if(protonFSIOnly) weightsFsiTrueTree->SetBranchAddress("nuEnergy", &TrueNuEnergy_TW, &b_TrueNuEnergy_TW_XsecTrue);
  if(!protonFSIOnly){
    weightsFsiTrueTree->SetBranchAddress("weights", &weightsFsiTrue, &b_weightsFsiTrue);
  }


  TArrayF *weightsOtherRec = new TArrayF(500);
  TArrayF *dialOtherRec = new TArrayF(500);
  weightsOtherRec=0;
  dialOtherRec=0;
  TBranch *b_weightsOtherRec;
  TBranch *b_dialOtherRec;
  if(protonFSIOnly) weightsOtherRecTree->SetBranchAddress("weights", &weightsOtherRec, &b_weightsOtherRec);
  if(protonFSIOnly) weightsOtherRecTree->SetBranchAddress("nuEnergy", &TrueNuEnergy_RW, &b_TrueNuEnergy_RW_XsecRec);
  if(protonFSIOnly) weightsOtherRecTree->SetBranchAddress("brNINuke_MFP_N", &dialOtherRec, &b_dialOtherRec);


  TArrayF *weightsOtherTrue = new TArrayF(500);
  TArrayF *dialOtherTrue = new TArrayF(500);
  weightsOtherTrue=0;
  dialOtherTrue=0;
  TBranch *b_weightsOtherTrue;
  TBranch *b_dialOtherTrue;
  if(protonFSIOnly) weightsOtherTrueTree->SetBranchAddress("weights", &weightsOtherTrue, &b_weightsOtherTrue);
  if(protonFSIOnly) weightsOtherTrueTree->SetBranchAddress("nuEnergy", &TrueNuEnergy_TW, &b_TrueNuEnergy_TW_XsecTrue);
  if(protonFSIOnly) weightsOtherTrueTree->SetBranchAddress("brNINuke_MFP_N", &dialOtherTrue, &b_dialOtherTrue);

  TArrayF *weightsFluxRec = new TArrayF(500);
  weightsFluxRec=0;
  TBranch *b_weightsFluxRec;
  if(!protonFSIOnly){
    weightsFluxRecTree->SetBranchAddress("weights", &weightsFluxRec, &b_weightsFluxRec);
  }

  TArrayF *weightsFluxTrue = new TArrayF(500);
  weightsFluxTrue=0;
  TBranch *b_weightsFluxTrue;
  if(!protonFSIOnly){
    weightsFluxTrueTree->SetBranchAddress("weights", &weightsFluxTrue, &b_weightsFluxTrue);
  }

  const int nbins = 9; 
  const int ndphitbins = 8; 
  const int ndalphatbins = 8; 
  double dptbins[nbins+1] = { 0.0, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 1.1, 100};
  double dphitbins[ndphitbins+1] = { 0.0, 0.067, 0.14, 0.225, 0.34, 0.52, 0.85, 1.5, 3.14159};
  double dalphatbins[ndalphatbins+1] = { 0.0, 0.47, 1.02, 1.54, 1.98, 2.34, 2.64, 2.89, 3.14159};

  TMatrixDSym covar_dpt(8);
  TMatrixDSym covar_mean_dpt(8);
  TMatrixDSym covar_norm_dpt(8);
  TMatrixDSym covar_mean_norm_dpt(8);

  TMatrixDSym covar_dphit(8);
  TMatrixDSym covar_mean_dphit(8);
  TMatrixDSym covar_norm_dphit(8);
  TMatrixDSym covar_mean_norm_dphit(8);

  TMatrixDSym covar_dat(8);
  TMatrixDSym covar_mean_dat(8);
  TMatrixDSym covar_norm_dat(8);
  TMatrixDSym covar_mean_norm_dat(8);

  for(Int_t i=0; i<8; i++){
      for(Int_t j=0; j<8; j++){
          covar_dpt[i][j]=0;
          covar_mean_dpt[i][j]=0;
          covar_norm_dpt[i][j]=0;
          covar_mean_norm_dpt[i][j]=0;
          covar_dphit[i][j]=0;
          covar_mean_dphit[i][j]=0;
          covar_norm_dphit[i][j]=0;
          covar_mean_norm_dphit[i][j]=0;
          covar_dat[i][j]=0;
          covar_mean_dat[i][j]=0;
          covar_norm_dat[i][j]=0;
          covar_mean_norm_dat[i][j]=0;
      }
  }


  TH1D* dpt_default = new TH1D("dpt_default","dpt_default",nbins,dptbins);
  TH1D* dphit_default = new TH1D("dphit_default","dphit_default",ndphitbins,dphitbins);
  TH1D* dalphat_default = new TH1D("dalphat_default","dalphat_default",ndalphatbins,dalphatbins);
  TH1D* oops_default = new TH1D("oops_default","oops_default",1,0,1);

  TH1D* dpt_SigWeights_rec = new TH1D("dpt_SigWeights_rec","dpt_SigWeights_rec",100,-5, 5);
  TH1D* dphit_SigWeights_rec = new TH1D("dphit_SigWeights_rec","dphit_SigWeights_rec",100,-5, 5);
  TH1D* dalphat_SigWeights_rec = new TH1D("dalphat_SigWeights_rec","dalphat_SigWeights_rec",100,-5, 5);

  TH1D* dpt_truth = new TH1D("dpt_truth","dpt_truth",nbins,dptbins);
  TH1D* dphit_truth = new TH1D("dphit_truth","dphit_truth",ndphitbins,dphitbins);
  TH1D* dalphat_truth = new TH1D("dalphat_truth","dalphat_truth",ndalphatbins,dalphatbins);

  TH1D* eff = new TH1D("eff","eff",nbins,dptbins);
  TH1D* eff_dphit = new TH1D("eff_dphit","eff_dphit",ndphitbins,dphitbins);
  TH1D* eff_dalphat = new TH1D("eff_dalphat","eff_dalphat",ndalphatbins,dalphatbins);

  vector<TH1D*> eff_allToys;
  vector<TH1D*> eff_allToys_dphit;
  vector<TH1D*> eff_allToys_dalphat;

  TH1D* effRecVar = new TH1D("effRecVar","effRecVar",nbins,dptbins);
  TH1D* effRecVar_dphit = new TH1D("effRecVar_dphit","effRecVar_dphit",ndphitbins,dphitbins);
  TH1D* effRecVar_dalphat = new TH1D("effRecVar_dalphat","effRecVar_dalphat",ndalphatbins,dalphatbins);
  TH1D* effRecVarOOPS = new TH1D("effRecVarOOPS","effRecVarOOPS",1,0,1);

  TH1D* dpt_default_nom = new TH1D("dpt_default_nom","dpt_default_nom",nbins,dptbins);
  TH1D* dphit_default_nom = new TH1D("dphit_default_nom","dphit_default_nom",ndphitbins,dphitbins);
  TH1D* dalphat_default_nom = new TH1D("dalphat_default_nom","dalphat_default_nom",ndalphatbins,dalphatbins);

  TH1D* dpt_SigWeights_true = new TH1D("dpt_SigWeights_true","dpt_SigWeights_true",100,-5, 5);
  TH1D* dphit_SigWeights_true = new TH1D("dphit_SigWeights_true","dphit_SigWeights_true",100,-5, 5);
  TH1D* dalphat_SigWeights_true = new TH1D("dalphat_SigWeights_true","dalphat_SigWeights_true",100,-5, 5);

  TH1D* dpt_truth_nom = new TH1D("dpt_truth_nom","dpt_truth_nom",nbins,dptbins);
  TH1D* dphit_truth_nom = new TH1D("dphit_truth_nom","dphit_truth_nom",ndphitbins,dphitbins);
  TH1D* dalphat_truth_nom = new TH1D("dalphat_truth_nom","dalphat_truth_nom",ndalphatbins,dalphatbins);
  TH1D* oops_truth_nom = new TH1D("oops_truth_nom","oops_truth_nom",1,0,1);

  TH1D* eff_nom = new TH1D("eff_nom","eff_nom",nbins,dptbins);
  TH1D* eff_nom_dphit = new TH1D("eff_nom_dphit","eff_nom_dphit",ndphitbins,dphitbins);
  TH1D* eff_nom_dalphat = new TH1D("eff_nom_dalphat","eff_nom_dalphat",ndalphatbins,dalphatbins);

  TH1D* relError = new TH1D("relError","relError",nbins,dptbins);
  TH1D* relError_dphit = new TH1D("relError_dphit","relError_dphit",ndphitbins,dphitbins);
  TH1D* relError_dalphat = new TH1D("relError_dalphat","relError_dalphat",ndalphatbins,dalphatbins);

  TH1D* relErrorAB = new TH1D("relErrorAB","relErrorAB",nbins+1,0,nbins+1);
  TH1D* relErrorAB_dphit = new TH1D("relErrorAB_dphit","relErrorAB_dphit",ndphitbins+1,0,ndphitbins+1);
  TH1D* relErrorAB_dalphat = new TH1D("relErrorAB_dalphat","relErrorAB_dalphat",ndalphatbins+1,0,ndalphatbins+1);

  TH1D* relError_corr = new TH1D("relError_corr","relError_corr",nbins,dptbins);
  TH1D* relError_dphit_corr = new TH1D("relError_dphit_corr","relError_dphit_corr",ndphitbins,dphitbins);
  TH1D* relError_dalphat_corr = new TH1D("relError_dalphat_corr","relError_dalphat_corr",ndalphatbins,dalphatbins);

  TH1D* relErrorAB_corr = new TH1D("relErrorAB_corr","relErrorAB_corr",nbins+1,0,nbins+1);
  TH1D* relErrorAB_dphit_corr = new TH1D("relErrorAB_dphit_corr","relErrorAB_dphit_corr",ndphitbins+1,0,ndphitbins+1);
  TH1D* relErrorAB_dalphat_corr = new TH1D("relErrorAB_dalphat_corr","relErrorAB_dalphat_corr",ndalphatbins+1,0,ndalphatbins+1);

  TH1D* psystEffVar = new TH1D("psystEffVar","psystEffVar",nbins+1,0,nbins+1);
  TH1D* psystEffVar_dphit = new TH1D("psystEffVar_dphit","psystEffVar_dphit",ndphitbins+1,0,ndphitbins+1);
  TH1D* psystEffVar_dalphat = new TH1D("psystEffVar_dalphat","psystEffVar_dalphat",ndalphatbins+1,0,ndalphatbins+1);


  TH1D* effRecVarBinVarOOPS = new TH1D("effRecVarBinOOPS","effRecVarBinOOPS",5000,0,1.0);
  vector<TH1D*> effBinVar;
  vector<TH1D*> effBinVar_dphit;
  vector<TH1D*> effBinVar_dalphat;
  vector<TH1D*> effRecVarBinVar;
  vector<TH1D*> effRecVarBinVar_dphit;
  vector<TH1D*> effRecVarBinVar_dalphat;
  for(int i=0;i<nbins;i++){
    string effBinHistName = Form("effBin%d", i);
    TH1D* effBin = new TH1D(effBinHistName.c_str(),effBinHistName.c_str(),5000,0,1.0);
    effBinVar.push_back(effBin);

    string effRecVarBinHistName = Form("effRecVarBin%d", i);
    TH1D* effRecVarBin = new TH1D(effRecVarBinHistName.c_str(),effRecVarBinHistName.c_str(),5000,0,1.0);
    effRecVarBinVar.push_back(effRecVarBin);
  }

  for(int i=0;i<ndphitbins;i++){
    string effBinHistName = Form("effBin_dphit%d", i);
    TH1D* effBin = new TH1D(effBinHistName.c_str(),effBinHistName.c_str(),5000,0,1.0);
    effBinVar_dphit.push_back(effBin);

    string effRecVarBinHistName = Form("effRecVarBin_dphit%d", i);
    TH1D* effRecVarBin = new TH1D(effRecVarBinHistName.c_str(),effRecVarBinHistName.c_str(),5000,0,1.0);
    effRecVarBinVar_dphit.push_back(effRecVarBin);
  }

  for(int i=0;i<ndalphatbins;i++){
    string effBinHistName = Form("effBin_dalphat%d", i);
    TH1D* effBin = new TH1D(effBinHistName.c_str(),effBinHistName.c_str(),5000,0,1.0);
    effBinVar_dalphat.push_back(effBin);

    string effRecVarBinHistName = Form("effRecVarBin_dalphat%d", i);
    TH1D* effRecVarBin = new TH1D(effRecVarBinHistName.c_str(),effRecVarBinHistName.c_str(),5000,0,1.0);
    effRecVarBinVar_dalphat.push_back(effRecVarBin);
  }

  Float_t totEvtWght = 0;
  Float_t dialValue = 0;

  // Signal definition phase space contraints
  Float_t pmul = 250;
  Float_t cthmul = -0.6;
  Float_t ppl = 450;
  Float_t pph = 1000;
  Float_t cthpl = 0.4;
  bool passEvent=false;
  bool passOOPSEvent=false;
  bool printPassedEvent=true;


  Long64_t nentries = hl2Intree->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  Long64_t nbytesw = 0, nbw = 0;
  Long64_t nbyteswfsi = 0, nbwfsi = 0;
  Long64_t nbyteswflux = 0, nbwflux = 0;

  Long64_t nentries_T = hl2Intree_T->GetEntriesFast();
  Long64_t nbytes_T = 0, nb_T = 0;
  Long64_t nbytesw_T = 0, nbw_T = 0;
  Long64_t nbyteswfsi_T = 0, nbwfsi_T = 0;
  Long64_t nbyteswflux_T = 0, nbwflux_T = 0;

  for(Int_t toy=0; toy<ntoys; toy++){
    dpt_default->Reset();
    dpt_default_nom->Reset();
    dphit_default->Reset();
    dphit_default_nom->Reset();
    dalphat_default->Reset();
    dalphat_default_nom->Reset();
    dpt_truth->Reset();
    dpt_truth_nom->Reset();
    dphit_truth->Reset();
    dphit_truth_nom->Reset();
    dalphat_truth->Reset();
    dalphat_truth_nom->Reset();
    std::cout << "Processing Toy: " << toy << std::endl;
    printPassedEvent=true;
    if(protonFSIOnly){
      nbwfsi = weightsOtherRecTree->GetEntry(0);
      dialValue=((*dialOtherRec)[toy]);
      std::cout << "Dial value is: " << dialValue << std::endl;
      if((dialValue<dialLow) || (dialValue>dialHigh) ) continue;
    }
    for(Long64_t jentry=0; jentry<nentries;jentry++) {
      nb = hl2Intree->GetEntry(jentry); nbytes += nb;
      if(protonFSIOnly){ nbwfsi = weightsOtherRecTree->GetEntry(jentry); nbyteswfsi += nbwfsi;}
      else{
        nbw = weightsXsRecTree->GetEntry(jentry); nbytesw += nbw;
        nbwfsi = weightsFsiRecTree->GetEntry(jentry); nbyteswfsi += nbwfsi;
        nbwflux = weightsFluxRecTree->GetEntry(jentry); nbyteswflux += nbwflux;
      }

      passEvent=false;
      passOOPSEvent=false;

      if(protonFSIOnly) dialValue=((*dialOtherRec)[toy]);
      if(protonFSIOnly) totEvtWght=((*weightsOtherRec)[toy]);
      else totEvtWght=((*weightsXsecRec)[toy]) * ((*weightsFsiRec)[toy]) * ((*weightsFluxRec)[toy]);
      //std::cout << "Weight: " << totEvtWght << std::endl; 


      if(totEvtWght>truncWeight) totEvtWght=truncWeight;

      if( ((mectopology==1) || (mectopology==2))  &&
            muMomTrue>pmul && muCosThetaTrue>cthmul &&
            pMomTrue>ppl && pMomTrue<pph && pCosThetaTrue>cthpl) passEvent=true;
      else if((mectopology==1) || (mectopology==2)) passOOPSEvent=true;

      if(passEvent && printPassedEvent){
        cout << "Current toy first passed rec event number: " << jentry << endl;
        cout << "dpT true is: " << dptTrue << endl;
        cout << "dphiT true is: " << dphitTrue << endl;
        cout << "dalphaT true is: " << dalphatTrue << endl;
        cout << "mu mom true is: " << muMomTrue << endl;
        cout << "mu Cos Theta true is: " << muCosThetaTrue << endl;
        cout << "Event weight is " << totEvtWght << endl;
        cout << "Neutrino Energy from HL2: " << TrueNuEnergy << endl;
        cout << "Neutrino Energy from Rec Weights: " << TrueNuEnergy_RW << endl;
        if(protonFSIOnly) cout << "Dial value is " << dialValue << endl;
        printPassedEvent=false;
      } 
      //if(passEvent && totEvtWght!=1) cout << "Reco weight is:" << totEvtWght << endl;
  
      for(int i=0; i<nbranches; i++){
        if(accum_level[0][i]>accumToCut[i] && passEvent==true) {
          if(i==0 || i==4) break;
          //std::cout << "Signal Event, trueDpT read from tree: " << dptTrue<< std::endl;
          dpt_SigWeights_rec->Fill(totEvtWght);
          dpt_default->Fill(dptTrue, totEvtWght);
          dpt_default_nom->Fill(dptTrue);

          dphit_SigWeights_rec->Fill(totEvtWght);
          dphit_default->Fill(dphitTrue, totEvtWght);
          dphit_default_nom->Fill(dphitTrue);

          dalphat_SigWeights_rec->Fill(totEvtWght);
          dalphat_default->Fill(dalphatTrue, totEvtWght);
          dalphat_default_nom->Fill(dalphatTrue);
          break;
        } 
        if(accum_level[0][i]>accumToCut[i] && passOOPSEvent==true) {
          if(i==0 || i==4) break;
          oops_default->Fill(0.5, totEvtWght);
          break;
        } 
      }
    }
    std::cout << "Processed Reco Events " << std::endl;
    printPassedEvent=true;


    for (Long64_t jentry=0; jentry<nentries_T;jentry++) {
      nb_T = hl2Intree_T->GetEntry(jentry); nbytes_T += nb_T;
      if(protonFSIOnly) {nbwfsi_T = weightsOtherTrueTree->GetEntry(jentry); nbyteswfsi_T += nbwfsi_T;}
      else{
        nbw_T = weightsXsTrueTree->GetEntry(jentry); nbytesw_T += nbw_T;
        nbwfsi_T = weightsFsiTrueTree->GetEntry(jentry); nbyteswfsi_T += nbwfsi_T;
        nbwflux_T = weightsFluxTrueTree->GetEntry(jentry); nbyteswflux_T += nbwflux_T;
      }
      passEvent=false;
      passOOPSEvent=false;
      // std::cout << "true entry is: " << jentry<< std::endl;
      // std::cout << "trueDpT read from tree: " << dptTrue_T<< std::endl;
      // std::cout << "Weight read from tree: " << (*weightsXsecTrue)[0] << std::endl;
      //std::cout << "Finding Weight " << std::endl; 
      if(protonFSIOnly) totEvtWght=((*weightsOtherTrue)[toy]);
      if(protonFSIOnly) dialValue=((*dialOtherRec)[toy]);
      else totEvtWght=((*weightsXsecTrue)[toy]) * ((*weightsFsiTrue)[toy]) * ((*weightsFluxTrue)[toy]);

      if(protonFSIOnly && ((dialValue<dialLow) || (dialValue>dialHigh)) ) continue;


      if(totEvtWght>truncWeight) totEvtWght=truncWeight;


      //std::cout << "Weight: " << totEvtWght << std::endl; 
      if( ((mectopology_T==1) || (mectopology_T==2))  &&
          muMomTrue_T>pmul && muCosThetaTrue_T>cthmul &&
          pMomTrue_T>ppl && pMomTrue_T<pph && pCosThetaTrue_T>cthpl) passEvent=true;
      else if((mectopology_T==1) || (mectopology_T==2)) passOOPSEvent=true;

      if(passEvent && printPassedEvent){
        cout << "Current toy first passed true event number: " << jentry << endl;
        cout << "dpT true is: " << dptTrue_T << endl;
        cout << "dphiT true is: " << dphitTrue_T << endl;
        cout << "dalphaT true is: " << dalphatTrue_T << endl;
        cout << "mu mom true is: " << muMomTrue_T << endl;
        cout << "mu Cos Theta true is: " << muCosThetaTrue_T << endl;
        cout << "Event weight is " << totEvtWght << endl;
        cout << "Neutrino Energy from HL2: " << TrueNuEnergy_T << endl;
        cout << "Neutrino Energy from True Weights: " << TrueNuEnergy_TW << endl;
        if(protonFSIOnly) cout << "Dial value is " << dialValue << endl;

        printPassedEvent=false;          
      }  
      //if(passEvent && totEvtWght!=1) cout << "True weight is:" << totEvtWght << endl;


      if(passEvent==true) {
        //std::cout << "Signal Event, trueDpT read from tree: " << dptTrue_T<< std::endl;
        dpt_truth->Fill(dptTrue_T, totEvtWght);
        dpt_SigWeights_true->Fill(totEvtWght);
        dpt_truth_nom->Fill(dptTrue_T);

        dphit_truth->Fill(dphitTrue_T, totEvtWght);
        dphit_SigWeights_true->Fill(totEvtWght);
        dphit_truth_nom->Fill(dphitTrue_T);

        dalphat_truth->Fill(dalphatTrue_T, totEvtWght);
        dalphat_SigWeights_true->Fill(totEvtWght);
        dalphat_truth_nom->Fill(dalphatTrue_T);
      }
      if(passOOPSEvent==true) {
        oops_truth_nom->Fill(0.5);
      } 
      // for(int i=0; i<nbranches; i++){
      //   if(accum_level_T[i]>accumToCut[i] && passEvent==true) {
      //     if(i==0 || i==4) break;
      //     dpt_default->Fill(dptTrue_T, totEvtWght);
      //     dpt_default_nom->Fill(dptTrue_T);
      //     break;
      //   } 
      // }
    }
    eff->Divide(dpt_default,dpt_truth);
    effRecVar->Divide(dpt_default,dpt_truth_nom);
    eff_nom->Divide(dpt_default_nom,dpt_truth_nom);

    eff_dphit->Divide(dphit_default,dphit_truth);
    effRecVar_dphit->Divide(dphit_default,dphit_truth_nom);
    eff_nom_dphit->Divide(dphit_default_nom,dphit_truth_nom);

    eff_dalphat->Divide(dalphat_default,dalphat_truth);
    effRecVar_dalphat->Divide(dalphat_default,dalphat_truth_nom);
    eff_nom_dalphat->Divide(dalphat_default_nom,dalphat_truth_nom);

    for(Int_t i=0; i<8; i++){
        for(Int_t j=0; j<8; j++){
            covar_dpt[i][j]+=(eff->GetBinContent(i+1)-eff_nom->GetBinContent(i+1))*(eff->GetBinContent(j+1)-eff_nom->GetBinContent(j+1)); 
            covar_dphit[i][j]+=(eff_dphit->GetBinContent(i+1)-eff_nom_dphit->GetBinContent(i+1))*(eff_dphit->GetBinContent(j+1)-eff_nom_dphit->GetBinContent(j+1)); 
            covar_dat[i][j]+=(eff_dalphat->GetBinContent(i+1)-eff_nom_dalphat->GetBinContent(i+1))*(eff_dalphat->GetBinContent(j+1)-eff_nom_dalphat->GetBinContent(j+1)); 
        }
    }

    eff_allToys.push_back(eff);
    eff_allToys_dphit.push_back(eff_dphit);
    eff_allToys_dalphat.push_back(eff_dalphat);

    effRecVarOOPS->Divide(oops_default,oops_truth_nom);
    effRecVarBinVarOOPS->Fill(effRecVarOOPS->GetBinContent(1));
    for(int i=0; i<nbins; i++){
      effBinVar[i]->Fill(eff->GetBinContent(i+1));
      effRecVarBinVar[i]->Fill(effRecVar->GetBinContent(i+1));
    }
    for(int i=0; i<ndphitbins; i++){
      effBinVar_dphit[i]->Fill(eff_dphit->GetBinContent(i+1));
      effRecVarBinVar_dphit[i]->Fill(effRecVar_dphit->GetBinContent(i+1));
    }
    for(int i=0; i<ndalphatbins; i++){
      effBinVar_dalphat[i]->Fill(eff_dalphat->GetBinContent(i+1));
      effRecVarBinVar_dalphat[i]->Fill(effRecVar_dalphat->GetBinContent(i+1));
    }
    if(protonFSIOnly && protonFSISystPlot && (dialValue>dialLow && dialValue<dialHigh) ) {
      cout << "Saving dpt dist with dial value of " << dialValue << endl;
      break;
    }
  }

  for(Int_t t=0; t<ntoys; t++){
      for(Int_t i=0; i<8; i++){
          for(Int_t j=0; j<8; j++){
              covar_mean_dpt[i][j]+=(eff_allToys[t]->GetBinContent(i+1)-effBinVar[i]->GetMean())*(eff_allToys[t]->GetBinContent(j+1)-effBinVar[j]->GetMean()); 
              covar_mean_dphit[i][j]+=(eff_allToys_dphit[t]->GetBinContent(i+1)-effBinVar_dphit[i]->GetMean())*(eff_allToys_dphit[t]->GetBinContent(j+1)-effBinVar_dphit[j]->GetMean()); 
              covar_mean_dat[i][j]+=(eff_allToys_dalphat[t]->GetBinContent(i+1)-effBinVar_dalphat[i]->GetMean())*(eff_allToys_dalphat[t]->GetBinContent(j+1)-effBinVar_dalphat[j]->GetMean()); 
          }
      }
  }
  
  for(int i=0; i<nbins; i++){
    relError->SetBinContent(i+1, (effRecVarBinVar[i]->GetRMS())/(effRecVarBinVar[i]->GetMean()));
    relErrorAB->SetBinContent(i+1, (effRecVarBinVar[i]->GetRMS())/(effRecVarBinVar[i]->GetMean()));
  }
  for(int i=0; i<ndphitbins; i++){
    relError_dphit->SetBinContent(i+1, (effRecVarBinVar_dphit[i]->GetRMS())/(effRecVarBinVar_dphit[i]->GetMean()));
    relErrorAB_dphit->SetBinContent(i+1, (effRecVarBinVar_dphit[i]->GetRMS())/(effRecVarBinVar_dphit[i]->GetMean()));
  }
  for(int i=0; i<ndalphatbins; i++){
    relError_dalphat->SetBinContent(i+1, (effRecVarBinVar_dalphat[i]->GetRMS())/(effRecVarBinVar_dalphat[i]->GetMean()));
    relErrorAB_dalphat->SetBinContent(i+1, (effRecVarBinVar_dalphat[i]->GetRMS())/(effRecVarBinVar_dalphat[i]->GetMean()));
  }
  relErrorAB->SetBinContent(nbins+1, (effRecVarBinVarOOPS->GetRMS())/(effRecVarBinVarOOPS->GetMean()));


  for(int i=0; i<nbins; i++){
    relError_corr->SetBinContent(i+1, (effBinVar[i]->GetRMS())/(effBinVar[i]->GetMean()));
    relErrorAB_corr->SetBinContent(i+1, (effBinVar[i]->GetRMS())/(effBinVar[i]->GetMean()));
  }
  for(int i=0; i<ndphitbins; i++){
    relError_dphit_corr->SetBinContent(i+1, (effBinVar_dphit[i]->GetRMS())/(effBinVar_dphit[i]->GetMean()));
    relErrorAB_dphit_corr->SetBinContent(i+1, (effBinVar_dphit[i]->GetRMS())/(effBinVar_dphit[i]->GetMean()));
  }
  for(int i=0; i<ndalphatbins; i++){
    relError_dalphat_corr->SetBinContent(i+1, (effBinVar_dalphat[i]->GetRMS())/(effBinVar_dalphat[i]->GetMean()));
    relErrorAB_dalphat_corr->SetBinContent(i+1, (effBinVar_dalphat[i]->GetRMS())/(effBinVar_dalphat[i]->GetMean()));
  }
  //relErrorAB_corr->SetBinContent(nbins+1, (effBinVarRecVarBinVarOOPS->GetRMS())/(effBinVarRecVarBinVarOOPS->GetMean()));

  if(protonFSIOnly && protonFSISystPlot && (dialValue>dialLow && dialValue<dialHigh) ) {
    for(int i=0; i<nbins; i++){
      psystEffVar->SetBinContent(i+1, (eff->GetBinContent(i+1)-eff_nom->GetBinContent(i+1))/eff_nom->GetBinContent(i+1));
    }
    for(int i=0; i<ndphitbins; i++){
      psystEffVar_dphit->SetBinContent(i+1, (eff_dphit->GetBinContent(i+1)-eff_nom_dphit->GetBinContent(i+1))/eff_nom_dphit->GetBinContent(i+1));
    }
    for(int i=0; i<ndalphatbins; i++){
      psystEffVar_dalphat->SetBinContent(i+1, (eff_dalphat->GetBinContent(i+1)-eff_nom_dalphat->GetBinContent(i+1))/eff_nom_dalphat->GetBinContent(i+1));
    }
    psystEffVar->Write();
    psystEffVar_dphit->Write();
    psystEffVar_dalphat->Write();
  }

  covar_dpt*=1.0/(Float_t)ntoys;
  covar_dphit*=1.0/(Float_t)ntoys;
  covar_dat*=1.0/(Float_t)ntoys;

  covar_mean_dpt*=1.0/(Float_t)ntoys;
  covar_mean_dphit*=1.0/(Float_t)ntoys;
  covar_mean_dat*=1.0/(Float_t)ntoys;

  for(Int_t i=0; i<8; i++){
      for(Int_t j=0; j<8; j++){
          covar_norm_dpt[i][j]+=covar_dpt[i][j]/(eff_nom->GetBinContent(i+1)*eff_nom->GetBinContent(j+1)); 
          covar_norm_dphit[i][j]+=covar_dphit[i][j]/(eff_nom_dphit->GetBinContent(i+1)*eff_nom_dphit->GetBinContent(j+1)); 
          covar_norm_dat[i][j]+=covar_dat[i][j]/(eff_nom_dalphat->GetBinContent(i+1)*eff_nom_dalphat->GetBinContent(j+1));

          covar_mean_norm_dpt[i][j]+=covar_mean_dpt[i][j]/(effRecVarBinVar[i]->GetMean()*effRecVarBinVar[j]->GetMean()); 
          covar_mean_norm_dphit[i][j]+=covar_mean_dphit[i][j]/(effRecVarBinVar_dphit[i]->GetMean()*effRecVarBinVar_dphit[j]->GetMean()); 
          covar_mean_norm_dat[i][j]+=covar_mean_dat[i][j]/(effRecVarBinVar_dalphat[i]->GetMean()*effRecVarBinVar_dalphat[j]->GetMean());  
      }
  }


  outfile->cd();

  covar_dpt.Write("covar_dpt");
  covar_norm_dpt.Write("covar_norm_dpt");
  covar_dphit.Write("covar_dphit");
  covar_norm_dphit.Write("covar_norm_dphit");
  covar_dat.Write("covar_dat");
  covar_norm_dat.Write("covar_norm_dat");

  covar_mean_dpt.Write("covar_mean_dpt");
  covar_mean_norm_dpt.Write("covar_mean_norm_dpt");
  covar_mean_dphit.Write("covar_mean_dphit");
  covar_mean_norm_dphit.Write("covar_mean_norm_dphit");
  covar_mean_dat.Write("covar_mean_dat");
  covar_mean_norm_dat.Write("covar_mean_norm_dat");

  relError->Write();
  relError_dphit->Write();
  relError_dalphat->Write();

  relErrorAB->Write();
  relErrorAB_dphit->Write();
  relErrorAB_dalphat->Write();

  relError_corr->Write();
  relError_dphit_corr->Write();
  relError_dalphat_corr->Write();

  relErrorAB_corr->Write();
  relErrorAB_dphit_corr->Write();
  relErrorAB_dalphat_corr->Write();


  dpt_SigWeights_rec->Write();
  dpt_SigWeights_true->Write();

  dphit_SigWeights_rec->Write();
  dphit_SigWeights_true->Write();

  dalphat_SigWeights_rec->Write();
  dalphat_SigWeights_true->Write();


  for(int i=0; i<nbins; i++){
    effBinVar[i]->Write();
    effRecVarBinVar[i]->Write();
  }
  for(int i=0; i<ndphitbins; i++){
    effBinVar_dphit[i]->Write();
    effRecVarBinVar_dphit[i]->Write();
  }
  for(int i=0; i<ndalphatbins; i++){
    effBinVar_dalphat[i]->Write();
    effRecVarBinVar_dalphat[i]->Write();
  }

  dpt_truth->Write();
  dpt_default->Write();

  dphit_truth->Write();
  dphit_default->Write();

  dalphat_truth->Write();
  dalphat_default->Write();

  eff->Write();
  eff_dphit->Write();
  eff_dalphat->Write();

  effRecVar->Write();
  effRecVar_dphit->Write();
  effRecVar_dalphat->Write();

  dpt_truth_nom->Write();
  dpt_default_nom->Write();

  dphit_truth_nom->Write();
  dphit_default_nom->Write();

  dalphat_truth_nom->Write();
  dalphat_default_nom->Write();

  eff_nom->Write();
  eff_nom_dphit->Write();
  eff_nom_dalphat->Write();

  return 0;
}
