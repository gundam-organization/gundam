#ifndef __AnaSample_hh__
#define __AnaSample_hh__

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <TDirectory.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TRandom3.h>
#include <TTree.h>

#include "AnaEvent.hh"
#include "ColorOutput.hh"
#include "FitStructs.hh"
using xsllh::FitBin;

enum DataType
{
    kMC,
    kAsimov,
    kExternal
};

class AnaSample
{
public:
    AnaSample(int sample_id, const std::string& name, const std::string& detector,
              const std::string& binning, TTree* t_data);
    ~AnaSample();

    int GetN() const;
    AnaEvent* GetEvent(int evnum);
    void ClearEvents();
    void AddEvent(const AnaEvent& event);
    void ResetWeights();

    void PrintStats() const;
    void SetNorm(const double val) { m_norm = val; }
    void SetData(TObject* data);
    void MakeHistos();

    void SetBinning(const std::string& binning);
    int GetBinIndex(const double D1, const double D2) const;
    std::vector<FitBin> GetBinEdges() const { return m_bin_edges; }

    double CalcChi2() const;
    void FillEventHist(int datatype, bool stat_fluc = false);
    void Write(TDirectory* dirout, const std::string& bsname, int fititer);
    void GetSampleBreakdown(TDirectory* dirout, const std::string& tag,
                            const std::vector<std::string>& topology, bool save);

    double GetNorm() const { return m_norm; }
    int GetSampleID() const { return m_sample_id; }
    std::string GetName() const { return m_name; }
    std::string GetDetector() const { return m_detector; }
    std::string GetDetBinning() const { return m_binning; }

    TH1D* GetPredHisto() const { return m_hpred; }
    TH1D* GetDataHisto() const { return m_hdata; }
    TH1D* GetMCHisto() const { return m_hmc; }
    TH1D* GetMCTruthHisto() const { return m_hmc_true; }
    TH1D* GetSignalHisto() const { return m_hsig; }

protected:
    int m_sample_id;
    int m_nbins;
    double m_norm;

    std::string m_name;
    std::string m_detector;
    std::string m_binning;
    std::vector<AnaEvent> m_events;
    std::vector<FitBin> m_bin_edges;

    TTree* m_data_tree;
    TH1D* m_hmc_true;
    TH1D* m_hmc;
    TH1D* m_hpred;
    TH1D* m_hdata;
    TH1D* m_hsig;

    std::string TAG;
    std::string ERR;
};

#endif
