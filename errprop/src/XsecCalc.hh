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

#include "FitObj.hh"
#include "OptParser.hh"
#include "ToyThrower.hh"

class XsecCalc
{
    public:
        XsecCalc(const std::string& json_config);
        ~XsecCalc();

        void GenerateToys();
        void GenerateToys(const int ntoys);

        std::vector<TH1D> GetSelSignal() {return selected_events -> GetSignalHist();};
        TH1D GetSelSignal(const int signal_id) {return selected_events -> GetSignalHist(signal_id);};

        std::vector<TH1D> GetTruSignal() {return true_events -> GetSignalHist();};
        TH1D GetTruSignal(const int signal_id) {return true_events -> GetSignalHist(signal_id);};

    private:
        FitObj* selected_events;
        FitObj* true_events;

        //ToyThrower toy_thrower;

        std::vector<TH1D> toys_sel_events;
        std::vector<TH1D> toys_tru_events;
        std::vector<TH1D> toys_eff;

        unsigned int num_toys;

        const std::string TAG = color::GREEN_STR + "[XsecExtract]: " + color::RESET_STR;
        const std::string ERR = color::RED_STR + "[ERROR]: " + color::RESET_STR;
};

#endif
