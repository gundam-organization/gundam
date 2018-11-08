#ifndef ANATREEMC_HH
#define ANATREEMC_HH

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TROOT.h>
#include <TTree.h>

#include "AnaEvent.hh"
#include "AnaSample.hh"
#include "ColorOutput.hh"
#include "ProgressBar.hh"

class AnaTreeMC
{
private:
    TChain* fChain; //! pointer to the analyzed TTree or TChain

    // Declaration of leaf types
    int nutype;
    int cutBranch;
    int evtTopology;
    int evtReaction;
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

    bool read_extra_var;
    const std::string TAG = color::GREEN_STR + "[AnaTreeMC]: " + color::RESET_STR;

public:
    AnaTreeMC(const std::string& file_name, const std::string& tree_name, bool extra_var = false);
    ~AnaTreeMC();
    long int GetEntry(long int entry) const;
    void SetBranches();
    void GetEvents(std::vector<AnaSample*>& ana_samples, const std::vector<int>& sig_topology,
                   const bool evt_type);
};

#endif
