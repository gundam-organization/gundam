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
        int   evtTopology;
        int   evtReaction;
        float trueD1;
        float trueD2;
        int   qesampleFinal;
        float MainD1Glb;
        float MainD2;
        float MainRecEneGlb;
        float TrueEnergy;
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
        TBranch *b_evtTopology;   //!
        TBranch *b_evtReaction;   //!
        TBranch *b_trueD1;   //!
        TBranch *b_trueD2;   //!
        TBranch *b_qesampleFinal;   //!
        TBranch *b_MainD1Glb;   //!
        TBranch *b_MainD2;   //!
        TBranch *b_MainRecEneGlb;   //!
        TBranch *b_TrueEnergy;   //!
        TBranch *b_weight;   //!

        // New kinematic variables always included for phase space cuts
        TBranch *b_pMomRec;   //!
        TBranch *b_pMomTrue;   //!
        TBranch *b_muMomRec;   //!
        TBranch *b_muMomTrue;   //!
        TBranch *b_muCosThetaRec;   //!
        TBranch *b_muCosThetaTrue;   //!
        TBranch *b_pCosThetaRec;   //!
        TBranch *b_pCosThetaTrue;   //!

        AnyTreeMC(const std::string& file_name);
        virtual ~AnyTreeMC();
        virtual long int GetEntry(long int entry);
        virtual void SetBranches();
        virtual void GetEvents(std::vector<AnaSample*> ana_samples);
};

#endif
