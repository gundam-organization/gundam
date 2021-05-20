#ifndef __XsecFitter_hh__
#define __XsecFitter_hh__

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

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

#include "../../fitparam/include/AnaFitParameters.hh"
#include "../../anaevents/include/AnaSample.hh"
#include "ColorOutput.hh"
#include "../../utils/include/OptParser.hh"

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
    bool Fit(const std::vector<AnaSample*>& samples, int fit_type, bool stat_fluc);
    void ParameterScans(const std::vector<int>& param_list, unsigned int nsteps);

    TMatrixD GetPriorCovarianceMatrix();
    TMatrixD GetPosteriorCovarianceMatrix();

    void WriteCovarianceMatrices();

    void SetMinSettings(const MinSettings& ms);
    void SetSeed(int seed);
    void SetZeroSyst(bool flag) { m_zerosyst = flag; }
    void SetTopology(const std::vector<std::string> vec_top) { v_topology = vec_top; }
    void SetNumThreads(const unsigned int num) { m_threads = num; }
    void SetSaveFreq(int freq, bool flag = true)
    {
        m_freq = freq;
        m_save = flag;
    }
    void SetSaveEvents(bool flag = true) { m_save_events = flag; };
    void SetPOTRatio(double val) { m_potratio = val; }

    TTree* outtree;

    // Declaration of leaf types
    Int_t sample;
    Float_t D1true;
    Float_t D2true;
    Int_t nutype;
    Int_t topology;
    Int_t reaction;
    Int_t target;
    Int_t sigtype;
    Float_t D1Reco;
    Float_t D2Reco;
    Float_t weight;
    Float_t weightNom;
    Float_t weightMC;

    void InitOutputTree()
    {
        outtree->Branch("nutype", &nutype, "nutype/I");
        outtree->Branch("reaction", &reaction, "reaction/I");
        outtree->Branch("target", &target, "target/I");
        outtree->Branch("sample", &sample, "cut_branch/I");
        outtree->Branch("sigtype", &sigtype, "signal/I");
        outtree->Branch("topology", &topology, "topology/I");
        outtree->Branch("D1True", &D1true, ("D1True/F"));
        outtree->Branch("D1Reco", &D1Reco, ("D1Reco/F"));
        outtree->Branch("D2True", &D2true, ("D2True/F"));
        outtree->Branch("D2Reco", &D2Reco, ("D2Reco/F"));
        outtree->Branch("weight", &weight, "weight/F");
        outtree->Branch("weightMC", &weightMC, "weightMC/F");
    }

private:
    void GenerateToyData(int toy_type, bool stat_fluc = false);
    double FillSamples(std::vector<std::vector<double>>& new_pars, int datatype = 0);
    void SaveParams(const std::vector<std::vector<double>>& new_pars);
    void SaveEventHist(int fititer, bool is_final = false);
    void SaveEventTree(std::vector<std::vector<double>>& par_results);
    void SaveChi2();
    void SaveResults(const std::vector<std::vector<double>>& parresults,
                     const std::vector<std::vector<double>>& parerrors);

    ROOT::Math::Minimizer* m_fitter;
    ROOT::Math::Functor* m_fcn;

    TRandom3* rng;
    TDirectory* m_dir;
    bool m_save;
    bool m_save_events;
    bool m_zerosyst;
    double m_potratio;
    int m_threads;
    int m_npar, m_calls, m_freq;
    std::string paramVectorFileName;
    std::vector<std::string> v_topology;
    std::vector<std::string> par_names;
    std::vector<double> par_prefit;
    std::vector<double> vec_chi2_stat;
    std::vector<double> vec_chi2_sys;
    std::vector<double> vec_chi2_reg;
    std::vector<AnaFitParameters*> m_fitpara;
    std::vector<AnaSample*> m_samples;

    MinSettings min_settings;
    const std::string TAG = color::YELLOW_STR + "[XsecFitter]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;
};
#endif
