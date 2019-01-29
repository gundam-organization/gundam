#ifndef FITOBJ_HH
#define FITOBJ_HH

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <TAxis.h>
#include <TFile.h>
#include <TH1D.h>

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
        FitObj(const std::string& json_config, const std::string& event_tree, bool is_true_tree);
        ~FitObj();

        void InitSignalHist(const std::vector<SignalDef>& v_signal);
        void ReweightEvents(const std::vector<double>& parameters);

        void ReweightNominal();
        void ResetHist();

        std::vector<TH1D> GetSignalHist() {return signal_hist;};
        TH1D GetSignalHist(const int signal_id) {return signal_hist.at(signal_id);}

        std::vector<SignalDef> GetSignalDef() {return signal_def;};

    private:
        std::vector<AnaFitParameters*> fit_par;
        std::vector<AnaSample*> samples;

        std::vector<TH1D> signal_hist;
        std::vector<BinManager> signal_bins;
        std::vector<SignalDef> signal_def;

        int m_threads;
        std::string m_tree_type;

        const std::string TAG = color::YELLOW_STR + "[FitObj]: " + color::RESET_STR;
        const std::string ERR = color::RED_STR + "[ERROR]: " + color::RESET_STR;
};

#endif
