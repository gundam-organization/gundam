#ifndef __XsecFitter_hh__
#define __XsecFitter_hh__

#include <algorithm>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <sstream>
#include <string>

#include <TFile.h>
#include <TGraph.h>
#include <TMath.h>
#include <TMatrixT.h>
#include <TMatrixTSym.h>
#include <TRandom3.h>
#include <TVectorT.h>

#include "Math/Factory.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"

#include "AnaFitParameters.hh"
#include "AnaSample.hh"
#include "ColorOutput.hh"

enum FitType
{
    kAsimovFit,
    kExternalFit,
    kDataFit,
    kToyFit
};

class XsecFitter
{
public:
    XsecFitter(TDirectory* dirout, const int seed);
    XsecFitter(TDirectory* dirout, const int seed, const int num_threads);
    ~XsecFitter();
    double CalcLikelihood(const double* par);
    void InitFitter(std::vector<AnaFitParameters*>& fitpara);
    void FixParameter(const std::string& par_name, const double& value);
    void Fit(const std::vector<AnaSample*>& samples, int fit_type, bool stat_fluc);

    void SetSeed(int seed);
    void SetZeroSyst(bool flag) { m_zerosyst = flag; }
    void SetTopology(const std::vector<std::string> v_topology) { topology = v_topology; }
    void SetNumThreads(const unsigned int num) { m_threads = num; }
    void SetSaveFreq(int freq, bool flag = true)
    {
        m_freq = freq;
        m_save = flag;
    }
    void SetPOTRatio(double val) { m_potratio = val; }

    TTree* outtree;

    // Declaration of leaf types
    Int_t cutBranch;
    Float_t D1true;
    Float_t D2true;
    Int_t nutype;
    Int_t mectopology;
    Int_t reaction;
    Int_t target;
    Float_t D1Reco;
    Float_t D2Reco;
    Float_t weight;
    Float_t weightNom;
    Float_t weightMC;

    // New kinematic variables always included for phase space cuts
    Float_t pMomRec;
    Float_t pMomTrue;
    Float_t muMomRec;
    Float_t muMomTrue;
    Float_t muCosThetaRec;
    Float_t muCosThetaTrue;
    Float_t pCosThetaRec;
    Float_t pCosThetaTrue;

    void InitOutputTree()
    {
        // Set branches
        outtree->Branch("nutype", &nutype, "nutype/I");
        outtree->Branch("reaction", &reaction, "reaction/I");
        outtree->Branch("target", &target, "target/I");
        outtree->Branch("cutBranch", &cutBranch, "cutBranch/I");
        outtree->Branch("mectopology", &mectopology, "mectopology/I");
        outtree->Branch("D1True", &D1true, ("D1True/F"));
        outtree->Branch("D1Rec", &D1Reco, ("D1Rec/F"));
        outtree->Branch("D2True", &D2true, ("D2True/F"));
        outtree->Branch("D2Rec", &D2Reco, ("D2Rec/F"));
        outtree->Branch("weight", &weight, "weight/F");
        outtree->Branch("weightNom", &weightNom, "weightNom/F");
        outtree->Branch("weightMC", &weightMC, "weightMC/F");

        outtree->Branch("muMomRec", &muMomRec, ("muMomRec/F"));
        outtree->Branch("muMomTrue", &muMomTrue, ("muMomTrue/F"));
        outtree->Branch("muCosThetaRec", &muCosThetaRec, ("muCosThetaRec/F"));
        outtree->Branch("muCosThetaTrue", &muCosThetaTrue, ("muCosThetaTrue/F"));
        outtree->Branch("pMomRec", &pMomRec, ("pMomRec/F"));
        outtree->Branch("pMomTrue", &pMomTrue, ("pMomTrue/F"));
        outtree->Branch("pCosThetaRec", &pCosThetaRec, ("pCosThetaRec/F"));
        outtree->Branch("pCosThetaTrue", &pCosThetaTrue, ("pCosThetaTrue/F"));
    }

private:
    void GenerateToyData(int toy_type, bool stat_fluc = false);
    double FillSamples(std::vector<std::vector<double>>& new_pars, int datatype = 0);
    void SaveParams(const std::vector<std::vector<double>>& new_pars);
    void SaveEvents(int fititer);
    void SaveFinalEvents(int fititer, std::vector<std::vector<double>>& parresults);
    void SaveChi2();
    void SaveResults(const std::vector<std::vector<double>>& parresults,
                     const std::vector<std::vector<double>>& parerrors);

    ROOT::Math::Minimizer* m_fitter;
    ROOT::Math::Functor* m_fcn;

    TRandom3* rng;
    TDirectory* m_dir;
    bool m_save;
    bool m_zerosyst;
    double m_potratio;
    int m_threads;
    int m_npar, m_calls, m_freq;
    std::string paramVectorFileName;
    std::vector<std::string> topology;
    std::vector<std::string> par_names;
    std::vector<double> par_prefit;
    std::vector<double> vec_chi2_stat;
    std::vector<double> vec_chi2_sys;
    std::vector<double> vec_chi2_reg;
    std::vector<AnaFitParameters*> m_fitpara;
    std::vector<AnaSample*> m_samples;

    const std::string TAG = color::YELLOW_STR + "[XsecFitter]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;
};
#endif
