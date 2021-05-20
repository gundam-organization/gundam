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
#include "../../utils/include/OptParser.hh"

class AnaTreeMC
{
private:
    TChain* fChain; //! pointer to the analyzed TTree or TChain

    // Declaration of leaf types
    int nutype;
    int beammode;
    int sample;
    int topology;
    int reaction;
    int target;
    float D1True;
    float D1Reco;
    float D2True;
    float D2Reco;
    float q2_true;
    float q2_reco;
    float enu_true;
    float enu_reco;
    float weight;

    bool read_extra_var;
    const std::string TAG = color::GREEN_STR + "[AnaTreeMC]: " + color::RESET_STR;

    std::string _file_name_;

public:
    AnaTreeMC(const std::string& file_name, const std::string& tree_name, bool extra_var = false);
    ~AnaTreeMC();

    long int GetEntry(long int entry) const;
    void SetBranches();
    void GetEvents(std::vector<AnaSample*>& ana_samples, const std::vector<SignalDef>& v_signal,
                   const bool evt_type);

};

#endif
