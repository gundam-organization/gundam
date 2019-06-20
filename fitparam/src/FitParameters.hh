#ifndef __FitParameters_hh__
#define __FitParameters_hh__

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <TRandom3.h>

#include "AnaFitParameters.hh"
#include "FitStructs.hh"
#include "OptParser.hh"

class FitParameters : public AnaFitParameters
{
    public:
        FitParameters(const std::string& par_name);
        ~FitParameters();

        void InitParameters();
        void InitEventMap(std::vector<AnaSample*> &sample, int mode);
        void ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent,
                      std::vector<double>& params);
        bool SetBinning(const std::string& file_name, std::vector<xsllh::FitBin>& bins);
        void AddDetector(const std::string& det, const std::string& f_binning);
        void AddDetector(const std::string& det, const std::vector<SignalDef>& v_input);
        int GetBinIndex(const int sig, double D1, double D2) const;
        double CalcRegularisation(const std::vector<double>& params) const;
        double CalcRegularisation(const std::vector<double>& params, double strength,
                                  RegMethod flag = kL2Reg) const;

    private:
        std::map<std::string, std::vector<xsllh::FitBin>> m_fit_bins;
        std::map<std::string, int> m_det_offset;
        std::vector<std::string> v_detectors;
        std::map<int, std::vector<xsllh::FitBin>> m_signal_bins;
        std::map<int, int> m_sig_offset;
        std::vector<int> v_signals;
        unsigned int signal_id;

        const std::string TAG = color::GREEN_STR + "[FitParameters]: " + color::RESET_STR;
};

#endif
