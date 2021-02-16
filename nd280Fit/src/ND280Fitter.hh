#ifndef __ND280FITTER_HH__
#define __ND280FITTER_HH__

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
#include <future>

#include "Math/Factory.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"

#include "AnaFitParameters.hh"
#include "AnaSample.hh"
#include "ColorOutput.hh"
#include "OptParser.hh"


class ND280Fitter {

public:

    ND280Fitter();
    ~ND280Fitter();

    void Reset();

    // Pre-initialization Methods
    void SetOutputTDirectory(TDirectory* output_tdirectory_);
    void SetPrngSeed(int PRNG_seed_);
    void SetNbThreads(int nb_threads_);
    void SetMcNormalizationFactor(double MC_normalization_factor_);
    void SetMinimizationSettings(const MinSettings& minimization_settings_);
    void SetDisableSystFit(bool disable_syst_fit_);
    void SetSaveEventTree(bool save_event_tree_);
    void SetSaveFitParams(bool save_fit_params_);
    void SetApplyStatisticalFluctuationsOnSamples(bool apply_statistical_fluctuations_on_samples_);
    void SetSaveFitParamsFrequency(int save_fit_params_frequency_);
    void SetAnaFitParametersList(std::vector<AnaFitParameters*> AnaFitParameters_list_);
    void SetAnaSamplesList(std::vector<AnaSample*> AnaSample_list_);
    void SetSelectedDataType(int selected_data_type_);

    void Initialize();

    // Post-initialization Methods
    void WritePrefitData();
    void MakeOneSigmaChecks();
    bool Fit();
    void WritePostFitData();

    double EvalFit(const double* par);
    void UpdateFitParameterValues(const double* par);
    void PropagateSystematics();
    void ComputeChi2();

    void InitFitter(std::vector<AnaFitParameters*>& fitpara);
    void FixParameter(const std::string& par_name, const double& value);
    void ParameterScans(const std::vector<int>& param_list, unsigned int nsteps);

    TMatrixD* GeneratePriorCovarianceMatrix();
    TMatrixD* GeneratePosteriorCovarianceMatrix();

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

protected:
    void InitializeThreadsParameters();
    void InitializeFitParameters();
    void InitializeDataSamples();
    void InitializeFitter();

    void PassIfInitialized(const std::string& method_name_) const;
    void PassIfNotInitialized(const std::string& method_name_) const;

    void GenerateToyData(int toy_type, bool stat_fluc = false);
    void SaveParams(const std::vector<std::vector<double>>& new_pars);
    void SaveEventHist(int fititer, bool is_final = false);
    void SaveEventTree(std::vector<std::vector<double>>& par_results);
    void SaveResults(const std::vector<std::vector<double>>& parresults,
                     const std::vector<std::vector<double>>& parerrors);

    // Parallel methods
    void ReWeightEvents(int iThread_ = -1);
    void ReFillSampleMcHistograms(int iThread_ = -1);


private:

    bool _is_initialized_;
    bool _fitHasBeenDone_;
    bool _fit_has_converged_;
    bool _save_fit_params_;
    bool _save_event_tree_;
    bool _disableChi2Pulls_;
    bool _apply_statistical_fluctuations_on_samples_;
    bool _printFitState_;

    int _PRNG_seed_;
    int _nb_threads_;
    int _nb_fit_parameters_;
    int _nbFitCalls_;
    int _saveFitParamsFrequency_;
    int _nbTotalEvents_;

    double _MC_normalization_factor_;

    double _chi2Buffer_;
    double _chi2StatBuffer_;
    double _chi2PullsBuffer_;
    double _chi2RegBuffer_;

    std::vector<std::string> _parameter_names_;
    std::vector<bool> _fixParameterStatusList_;
    std::vector<double> _parameterPriorValues_;
    std::vector<double> _parameterSteps_;
    std::vector<double> _parameter_low_edges_;
    std::vector<double> _parameter_high_edges_;
    std::vector<AnaFitParameters*> _fitParametersGroupList_;
    std::vector<AnaSample*> _samplesList_;
    std::vector<std::vector<double>> _newParametersList_;

    std::vector<std::vector<std::vector<long long>>> _durationReweightParameters_; // [SystGroup][Thread][Iteration] = time
    std::map<int, std::vector<double>> _durationHistoryHandler_;

    DataType _selected_data_type_;
    MinSettings _minimization_settings_;

    TRandom3* _PRNG_;
    TDirectory* _outputTDirectory_;
    ROOT::Math::Minimizer* _minimizer_;
    ROOT::Math::Functor* _functor_;

    std::chrono::high_resolution_clock::time_point _startTime_;
    std::chrono::high_resolution_clock::time_point _endTime_;

    std::vector<double> _chi2StatHistory_;
    std::vector<double> _chi2PullsHistory_;
    std::vector<double> _chi2RegHistory_;

    // Multi-thread
    std::mutex _threadMutex_;
    int _counterThread_;
    bool _stopThreads_;
    std::vector<bool> _triggerReweightThreads_;
    std::vector<bool> _triggerReFillMcHistogramsThreads_;
    std::vector<std::future<void>> _asyncFitThreads_;
    std::vector<std::vector<std::vector<TH1D*>>> _histThreadHandlers_;

};

#endif // __ND280FITTER_HH__
