#ifndef FITOBJ_HH
#define FITOBJ_HH

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <TAxis.h>
#include <TFile.h>
#include <TH1D.h>
#include <TMatrixT.h>

using TMatrixDSym = TMatrixTSym<double>;

#include "AnaSample.hh"
#include "AnaTreeMC.hh"
#include "BinManager.hh"
#include "ColorOutput.hh"
#include "DetParameters.hh"
#include "FitParameters.hh"
#include "FluxParameters.hh"
#include "OptParser.hh"
#include "XsecParameters.hh"

class FitObj
{
public:
    FitObj(const std::string& json_config, const std::string& event_tree, bool is_true_tree,
           bool read_data_file = false);
    ~FitObj();

    void InitSignalHist(const std::vector<SignalDef>& v_signal);
    void ReweightEvents(const std::vector<double>& parameters);

    void ReweightNominal();
    void ResetHist();

    double ReweightFluxHist(const std::vector<double>& input_par, TH1D& flux_hist,
                            const std::string& det);

    TH1D GetHistCombined(const std::string& suffix = "") const;
    std::vector<TH1D> GetSignalHist() const { return signal_hist; };
    TH1D GetSignalHist(const int signal_id) const { return signal_hist.at(signal_id); }

    std::vector<SignalDef> GetSignalDef() const { return signal_def; };
    BinManager& GetBinManager(const int signal_id) { return signal_bins.at(signal_id); };
    unsigned int GetNumSignalBins() const { return total_signal_bins; };
    unsigned int GetNumSignalBins(const int signal_id) const
    {
        return signal_bins.at(signal_id).GetNbins();
    };

    int GetFitType() const { return m_fit_type; };
    TMatrixDSym GetPrefitCov();
    TMatrixDSym CalcTemplateCov();
    unsigned int GetNpar() const { return npar; };

private:
    FitParameters* m_fit_par;
    FluxParameters* m_flux_par;

    std::vector<AnaFitParameters*> fit_par;
    std::vector<AnaSample*> samples;

    std::vector<TH1D> signal_hist;
    std::vector<BinManager> signal_bins;
    std::vector<SignalDef> signal_def;

    int m_threads;
    int m_fit_type;
    double m_norm;
    unsigned int npar;
    unsigned int total_signal_bins;
    std::string m_tree_type;

    const std::string TAG = color::YELLOW_STR + "[FitObj]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + "[ERROR]: " + color::RESET_STR;
};

#endif
