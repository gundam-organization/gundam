#ifndef anyTreeMC_h
#define anyTreeMC_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

#include <vector>

#include "AnaEvent.hh"
#include "AnaSample.hh"

class anyTreeMC 
{
public :
  TChain         *fChain; //!pointer to the analyzed TTree or TChain
 
  // Declaration of leaf types
  Int_t          evtTopology;
  Int_t          evtReaction;
  Float_t        trueD1;
  Float_t        trueD2;
  Int_t          qesampleFinal;
  Float_t        MainD1Glb;
  Float_t        MainD2;
  Float_t        MainRecEneGlb;
  Float_t        TrueEnergy;
  Float_t        weight;


  // New kinematic variables always included for phase space cuts
  Float_t        pMomRec;
  Float_t        pMomTrue;
  Float_t        muMomRec;
  Float_t        muMomTrue;
  Float_t        muCosThetaRec;
  Float_t        muCosThetaTrue;
  Float_t        pCosThetaRec;
  Float_t        pCosThetaTrue;


  // List of branches
  TBranch        *b_evtTopology;   //!
  TBranch        *b_evtReaction;   //!
  TBranch        *b_trueD1;   //!
  TBranch        *b_trueD2;   //!
  TBranch        *b_qesampleFinal;   //!
  TBranch        *b_MainD1Glb;   //!
  TBranch        *b_MainD2;   //!
  TBranch        *b_MainRecEneGlb;   //!
  TBranch        *b_TrueEnergy;   //!
  TBranch        *b_weight;   //!

  // New kinematic variables always included for phase space cuts
  TBranch        *b_pMomRec;   //!
  TBranch        *b_pMomTrue;   //!
  TBranch        *b_muMomRec;   //!
  TBranch        *b_muMomTrue;   //!
  TBranch        *b_muCosThetaRec;   //!
  TBranch        *b_muCosThetaTrue;   //!
  TBranch        *b_pCosThetaRec;   //!
  TBranch        *b_pCosThetaTrue;   //!



  anyTreeMC(const char *fname);
  virtual ~anyTreeMC();
  virtual Int_t GetEntry(Long64_t entry);
  virtual void  Init();
  virtual void  GetEvents(std::vector<AnaSample*> ana_samples);
};

#endif

#ifdef anyTreeMC_cxx
anyTreeMC::anyTreeMC(const char *fname)
{
  const char *trname = "selectedEvents";
  //init TChain object
  fChain = new TChain(trname); 
  //add files to TChain
  fChain->Add(fname);
  //init tree leaf pntrs
  Init();
}

anyTreeMC::~anyTreeMC()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t anyTreeMC::GetEntry(Long64_t entry)
{
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

void anyTreeMC::Init()
{
  // Set branch addresses and branch pointers
  fChain->SetBranchAddress("mectopology", &evtTopology, &b_evtTopology);
  fChain->SetBranchAddress("reaction", &evtReaction, &b_evtReaction);
  fChain->SetBranchAddress("D1True", &trueD1, &b_trueD1);
  fChain->SetBranchAddress("D2True", &trueD2, &b_trueD2);
  fChain->SetBranchAddress("cutBranch", &qesampleFinal, &b_qesampleFinal);
  fChain->SetBranchAddress("D1Rec", &MainD1Glb, &b_MainD1Glb);
  fChain->SetBranchAddress("D2Rec", &MainD2, &b_MainD2);
  fChain->SetBranchAddress("Enureco", &MainRecEneGlb, &b_MainRecEneGlb);
  fChain->SetBranchAddress("Enutrue", &TrueEnergy, &b_TrueEnergy);
  fChain->SetBranchAddress("weight", &weight, &b_weight);

  // New kinematic variables always included for phase space cuts
  fChain->SetBranchAddress("pMomRec", &pMomRec, &b_pMomRec);
  fChain->SetBranchAddress("pMomTrue", &pMomTrue, &b_pMomTrue);
  fChain->SetBranchAddress("pCosThetaRec", &pCosThetaRec, &b_pCosThetaRec);
  fChain->SetBranchAddress("pCosThetaTrue", &pCosThetaTrue, &b_pCosThetaTrue);
  fChain->SetBranchAddress("muMomRec", &muMomRec, &b_muMomRec);
  fChain->SetBranchAddress("muMomTrue", &muMomTrue, &b_muMomTrue);
  fChain->SetBranchAddress("muCosThetaRec", &muCosThetaRec, &b_muCosThetaRec);
  fChain->SetBranchAddress("muCosThetaTrue", &muCosThetaTrue, &b_muCosThetaTrue);
}

#endif
