#ifndef XSECEXTRACTOR_HH
#define XSECEXTRACTOR_HH

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "TH1D.h"
#include "TRandom3.h"

enum class ThrowType: bool
{
    kThrow = true,
    kNom   = false
};

class XsecExtractor
{
    private:
        std::string m_name;
        std::string m_binning;
        std::vector<double> bin_widths;
        double num_targets_val;
        double num_targets_err;
        TH1D h_weighted_flux;
        double flux_integral;
        double flux_error;

        TRandom3 RNG;

    public:
        XsecExtractor(const std::string& name, unsigned int seed);
        XsecExtractor(const std::string& name, const std::string& binning, unsigned int seed);

        void SetBinning(const std::string& binning);
        void SetFluxHist(const TH1D& h_flux);
        void SetNumTargets(double ntargets, double nerror);

        void ApplyBinWidths(TH1D& h_event_rate);
        void ApplyFluxInt(TH1D& h_event_rate, bool do_throw);
        void ApplyNumTargets(TH1D& h_event_rate, bool do_throw);

        double ThrowNtargets();
};

#endif
