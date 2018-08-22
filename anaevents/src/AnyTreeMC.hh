#ifndef AnyTreeMC_h
#define AnyTreeMC_h

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

#include "AnaEvent.hh"
#include "AnaSample.hh"

class AnyTreeMC
{
    public:
        TChain *fChain; //!pointer to the analyzed TTree or TChain

        // Declaration of leaf types
        int   nutype;
        int   cutBranch;
        int   evtTopology;
        int   evtReaction;
        float D1True;
        float D2True;
        float D1Reco;
        float D2Reco;
        float EnuReco;
        float EnuTrue;
        float weight;


        // New kinematic variables always included for phase space cuts
        float pMomRec;
        float pMomTrue;
        float muMomRec;
        float muMomTrue;
        float muCosThetaRec;
        float muCosThetaTrue;
        float pCosThetaRec;
        float pCosThetaTrue;


        // List of branches
        TBranch *b_nutype;
        TBranch *b_cutBranch;
        TBranch *b_evtTopology;
        TBranch *b_evtReaction;
        TBranch *b_D1True;
        TBranch *b_D2True;
        TBranch *b_D1Reco;
        TBranch *b_D2Reco;
        TBranch *b_EnuReco;
        TBranch *b_EnuTrue;
        TBranch *b_weight;

        // New kinematic variables always included for phase space cuts
        TBranch *b_pMomRec;
        TBranch *b_pMomTrue;
        TBranch *b_muMomRec;
        TBranch *b_muMomTrue;
        TBranch *b_muCosThetaRec;
        TBranch *b_muCosThetaTrue;
        TBranch *b_pCosThetaRec;
        TBranch *b_pCosThetaTrue;

        AnyTreeMC(const std::string& file_name, const std::string& tree_name);
        ~AnyTreeMC();
        long int GetEntry(long int entry) const;
        void SetBranches();
        void GetEvents(std::vector<AnaSample*>& ana_samples, const std::vector<int>& sig_topology, const bool evt_type);
};

#endif
