#ifndef __AnaSample_hh__
#define __AnaSample_hh__

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <OptParser.hh>
#include <TDirectory.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TRandom3.h>
#include <TTree.h>

#include "AnaEvent.hh"
#include "ColorOutput.hh"
#include "FitStructs.hh"
#include "Likelihoods.hh"
using xsllh::FitBin;

enum DataType
{
    kReset    = -1,
    kMC       = 0,
    kAsimov   = 1,
    kExternal = 2,
    kData     = 3
};

class AnaSample
{

public:

    AnaSample(int sample_id, const std::string& name, const std::string& detector,
              const std::string& binning, TTree* t_data);
    AnaSample(const SampleOpt& sample, TTree* t_data);
    ~AnaSample();

    void Reset();

    int GetN() const;
    AnaEvent* GetEvent(int evnum);
    std::vector<AnaEvent>& GetEventList();
    void ClearEvents();
    void AddEvent(AnaEvent& event);
    void ResetWeights();

    void PrintStats() const;
    void SetNorm(const double val) { m_norm = val; }
    void SetData(TObject* data);
    void MakeHistos();

    void SaveMcEventsSnapshot();
    void SaveHistogramsSnapshot();

    void CompareHistogramsWithSnapshot();
    void CompareMcEventsWeightsWithSnapshot();

    void SetBinning(const std::string& binning);

    // Function to map the Highland topology codes to consecutive integers:
    void SetTopologyHLCode(const std::vector<int>& HLTopologyCodes);

    int GetBinIndex(const double D1, const double D2) const;
    std::vector<FitBin> GetBinEdges() const { return m_bin_edges; }

    void SetLLHFunction(const std::string& func_name);
    void SetDataType(DataType dataType_);
    DataType GetDataType() { return m_hdata_type; }

    double CalcLLH() const;
    double CalcChi2() const;
    double CalcEffLLH() const;

    void FillEventHist(int datatype, bool stat_fluc = false);
    void FillEventHist(DataType datatype, bool stat_fluc = false);

    void FillMcHistograms(int iThread_ = -1);
    void MergeMcHistogramsThread();

    void FillDataEventList();

    void PropagateMcWeightsToAsimovDataEvents();

    void Write(TDirectory* dirout, const std::string& bsname, int fititer);
    void GetSampleBreakdown(TDirectory* dirout, const std::string& tag,
                            const std::vector<std::string>& topology, bool save);

    double GetNorm() const { return m_norm; }
    int GetSampleID() const { return m_sample_id; }
    std::string GetName() const { return m_name; }
    std::string GetDetector() const { return m_detector; }
    std::string GetDetBinning() const { return m_binning; }
    std::string GetAdditionalCuts() const { return m_additional_cuts; }
    std::vector<AnaEvent>& GetMcEvents() { return m_mc_events; }
    std::vector<AnaEvent>& GetDataEvents() { return m_data_events; }

    // Return the const ref to avoid external modifications
    const TH1D& GetPredHisto() const { return *m_hpred; }
    const TH1D& GetDataHisto() const { return *m_hdata; }
    const TH1D& GetMCHisto() const { return *m_hmc; }
    const TH1D& GetMCTruthHisto() const { return *m_hmc_true; }
    const TH1D& GetSignalHisto() const { return *m_hsig; }

    TTreeFormula* GetAdditionalCutsFormulae() {return m_additional_cuts_formulae;}

    // Deprecated
    std::vector<AnaEvent>& GetEvents() { return m_mc_events; }

protected:

    DataType m_hdata_type;

    int m_sample_id;
    int m_nbins;
    double m_norm;

    std::string m_name;
    std::string m_detector;
    std::string m_binning;
    std::string m_additional_cuts;
    double m_data_POT;
    double m_mc_POT;
    std::vector<AnaEvent> m_mc_events;
    std::vector<AnaEvent> m_mc_events_snap;
    std::vector<AnaEvent> m_data_events;
    std::vector<FitBin> m_bin_edges;

    // Mapping between Highland topology codes and consecutive integers
    std::map<int, int> topology_HL_code;

    CalcLLHFunc* m_llh;

    TTree* m_data_tree;

//    std::vector<std::map<TH1D*, TH1D*>> _histThreadHandlers_; // [iThread][iHist]
    std::vector<std::vector<TH1D*>> _histThreadHandlers_; // [iThread][iHist]
    std::vector<bool> debugThreadBools;
    TH1D* m_hmc_true;
    TH1D* m_hmc;
    TH1D* m_hpred;
    TH1D* m_hdata;
    TH1D* m_hsig;

    TTreeFormula* m_additional_cuts_formulae;

    const std::string TAG = color::GREEN_STR + "[AnaSample]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + "[ERROR]: " + color::RESET_STR;

    std::vector<std::vector<TH1D*>> _histThreadHandlersSnap_;
    TH1D* m_hmc_true_snap;
    TH1D* m_hmc_snap;
    TH1D* m_hpred_snap;
    TH1D* m_hdata_snap;
    TH1D* m_hsig_snap;



};

#endif
