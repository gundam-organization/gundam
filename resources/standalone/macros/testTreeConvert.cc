/******************************************************

Code to convert a HL2 tree into the format required 
for the fitting code. C

Can't simply read HL2 tree directly since we don't
know what variables will be the tree

Author: Stephen Dolan
Date Created: November 2015

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
#include <TGraph.h>
#include <TTree.h>
#include <TString.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TMath.h>


using namespace std;

Int_t EvtStart = 0;
Int_t EvtEnd = 0;
Double_t EvtFrac = 0.0;

int treeConvert(TString inFileName, TString outFileName, TString inTreeName="default",  TString inTreeName_T="truth", bool isRealData=false)
{
  // You need to provide the number of branches in your HL2 tree
  // And the accum_level you want to cut each one at to get your selected events
  // i.e choosing n in accum_level[0][branch_i]>n
  const int nbranches = 10;
  //const int accumToCut[nbranches] =   {7,8,9,8,7,5,4,7,8,7};
  const int accumToCut[nbranches] =     {7,8,9,8,7,99,99,7,8,7};

  TFile *infile = new TFile(inFileName);
  TTree *intree = (TTree*)infile->Get(inTreeName);
  TTree *intree_T = (TTree*)infile->Get(inTreeName_T);

  TFile *outfile = new TFile(outFileName,"recreate");
  TTree *outtree = new TTree("selectedEvents", "selectedEvents");
  TTree *outtree_T = new TTree("trueEvents", "trueEvents");


  // Declaration of leaf types
  Int_t          accum_level[1500][50];
  Int_t          reaction;
  Int_t          cutBranch=-999;
  Int_t          mectopology;
  Float_t        muMomRec;
  Float_t        muMomRecRange;
  Float_t        muThetaRec;
  Float_t        muMomTrue;
  Float_t        muThetaTrue;
  Float_t        muCosThetaRec;
  Float_t        muCosThetaTrue;
  Float_t        RecoNuEnergy=0;
  Float_t        TrueNuEnergy=0;
  Float_t        weight;

  Int_t          reaction_T;
  Int_t          mectopology_T; 
  Float_t        muMomTrue_T;
  Float_t        muCosThetaTrue_T;
  Float_t        TrueNuEnergy_T;
  Float_t        weight_T=1.0;


  intree->SetBranchAddress("accum_level", &accum_level);
  intree->SetBranchAddress("reaction", &reaction);
  intree->SetBranchAddress("mectopology", &mectopology);
  intree->SetBranchAddress("selmu_mom", &muMomRec);
  intree->SetBranchAddress("selmu_mom_range_oarecon", &muMomRecRange);
  intree->SetBranchAddress("selmu_theta", &muThetaRec);
  intree->SetBranchAddress("truemu_mom", &muMomTrue);
  intree->SetBranchAddress("truemu_costheta", &muCosThetaTrue);
  intree->SetBranchAddress("nu_trueE", &TrueNuEnergy);
  intree->SetBranchAddress("weight", &weight);

  outtree->Branch("reaction", &reaction, "reaction/I");
  outtree->Branch("cutBranch", &cutBranch, "cutBranch/I");
  outtree->Branch("mectopology", &mectopology, "mectopology/I");
  outtree->Branch("muMomRec", &muMomRec, ("muMomRec/F"));
  outtree->Branch("muMomTrue", &muMomTrue, ("muMomTrue/F"));
  outtree->Branch("muCosThetaRec", &muCosThetaRec, ("muCosThetaRec/F"));
  outtree->Branch("muCosThetaTrue", &muCosThetaTrue, ("muCosThetaTrue/F"));
  outtree->Branch("Enureco", &RecoNuEnergy, "Enureco/F");
  outtree->Branch("Enutrue", &TrueNuEnergy, "Enutrue/F");
  outtree->Branch("weight", &weight, "weight/F");

  intree_T->SetBranchAddress("reaction", &reaction_T);
  intree_T->SetBranchAddress("mectopology", &mectopology_T);
  intree_T->SetBranchAddress("truemu_mom", &muMomTrue_T);
  intree_T->SetBranchAddress("truemu_costheta", &muCosThetaTrue_T);
  intree_T->SetBranchAddress("nu_trueE", &TrueNuEnergy_T);

  outtree_T->Branch("reaction", &reaction_T, "reaction/I");
  outtree_T->Branch("mectopology", &mectopology_T, "mectopology/I");
  outtree_T->Branch("muMomTrue", &muMomTrue_T, ("muMomTrue/F"));
  outtree_T->Branch("muCosThetaTrue", &muCosThetaTrue_T, ("muCosThetaTrue/F"));
  outtree_T->Branch("Enutrue", &TrueNuEnergy_T, "Enutrue/F");
  outtree_T->Branch("weight", &weight_T, "weight/F");


  Long64_t nentries = intree->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  if(EvtEnd!=0) nentries=EvtEnd;
  if(EvtFrac>0.0001) nentries=nentries*EvtFrac;
  int passCount=0;
  for (Long64_t jentry=EvtStart; jentry<nentries;jentry++) {
    nb = intree->GetEntry(jentry); nbytes += nb;
    passCount=0;
    RecoNuEnergy=TrueNuEnergy;
    muCosThetaRec  = TMath::Cos(muThetaRec);
    int branches_passed[10]={0};
    for(int i=0; i<nbranches; i++){
      if(accum_level[0][i]>accumToCut[i]){
        cutBranch=i; passCount++; 
        if(cutBranch==3) muMomRec=muMomRecRange;
        branches_passed[i]++;
        outtree->Fill();
      }
    }
    if(passCount>1){
      printf("***Warning: More than one cut branch passed***\n");
      for(int j=0;j<10;j++){
        if(branches_passed[j]==1) printf("branch %d passed ...",j);
      }
      printf("\n");
    }
  }

  Long64_t nentries_T = intree_T->GetEntriesFast();
  Long64_t nbytes_T = 0, nb_T = 0;
  if(EvtFrac>0.0001) nentries_T=nentries_T*EvtFrac;
  for (Long64_t jentry=0; jentry<nentries_T;jentry++) {
    weight_T=1;
    nb_T = intree_T->GetEntry(jentry); nbytes_T += nb_T;
    outtree_T->Fill();
  }
  
  printf("***Output Rec Tree: ***\n");
  outtree->Print();
  printf("***Output True Tree: ***\n");
  outtree_T->Print();
  outfile->Write();

  delete infile;
  delete outfile;
  return 0;
}
