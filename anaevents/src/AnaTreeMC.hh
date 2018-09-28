#ifndef ANATREEMC_HH
#define ANATREEMC_HH

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

class AnaTreeMC
{
    private:
        TChain *fChain; //!pointer to the analyzed TTree or TChain

        // Declaration of leaf types
        int   nutype;
        int   cutBranch;
        int   evtTopology;
        int   evtReaction;
        float D1True;
        float D1Reco;
        float D2True;
        float D2Reco;
        float Q2True;
        float Q2Reco;
        float EnuTrue;
        float EnuReco;
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

    public:
        AnaTreeMC(const std::string& file_name, const std::string& tree_name);
        ~AnaTreeMC();
        long int GetEntry(long int entry) const;
        void SetBranches();
        void GetEvents(std::vector<AnaSample*>& ana_samples, const std::vector<int>& sig_topology, const bool evt_type);
};

#endif
