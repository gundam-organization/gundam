#ifndef XSECCALC_HH
#define XSECCALC_HH

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <TFile.h>
#include <TH1D.h>

#include "ColorOutput.hh"
#include "FitObj.hh"
#include "OptParser.hh"
#include "ToyThrower.hh"

struct SigNorm
{
    std::string name;
    std::string detector;
    std::string flux_file;
    std::string flux_hist;
    double flux_int;
    double flux_err;
    bool use_flux_fit;

    double num_targets_val;
    double num_targets_err;
};

class XsecCalc
{
    public:
        XsecCalc(const std::string& json_config);
        ~XsecCalc();

        void ReweightNominal();

        void GenerateToys();
        void GenerateToys(const int ntoys);

        std::vector<TH1D> GetSelSignal() {return selected_events -> GetSignalHist();};
        TH1D GetSelSignal(const int signal_id) {return selected_events -> GetSignalHist(signal_id);};

        std::vector<TH1D> GetTruSignal() {return true_events -> GetSignalHist();};
        TH1D GetTruSignal(const int signal_id) {return true_events -> GetSignalHist(signal_id);};

        std::string GetOutputFileName() {return output_file;};
        std::string GetInputFileName() {return input_file;};

    private:
        void InitNormalization(const nlohmann::json& j);

        FitObj* selected_events;
        FitObj* true_events;

        //ToyThrower toy_thrower;

        std::vector<TH1D> toys_sel_events;
        std::vector<TH1D> toys_tru_events;
        std::vector<TH1D> toys_eff;
        std::vector<SigNorm> v_normalization;

        std::string input_file;
        std::string output_file;

        unsigned int num_toys;
        unsigned int rng_seed;

        const std::string TAG = color::GREEN_STR + "[XsecExtract]: " + color::RESET_STR;
        const std::string ERR = color::RED_STR + "[ERROR]: " + color::RESET_STR;
};

#endif
