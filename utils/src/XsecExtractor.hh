#ifndef XSECEXTRACTOR_HH
#define XSECEXTRACTOR_HH

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "TH1D.h"
#include "TRandom3.h"

#include "FitStructs.hh"

enum class ThrowType: bool
{
    kThrow = true,
    kNom   = false
};

enum class ErrorType
{
    kAbs,
    kRel,
    kSig
};

class XsecExtractor
{
    private:
        std::string m_name;
        std::string m_binning;
        std::vector<double> bin_widths;
        std::vector<xsllh::FitBin> bin_edges;
        unsigned int m_nbins;
        double num_targets_val;
        double num_targets_err;
        TH1D h_weighted_flux;
        double flux_integral;
        double flux_error;

        TRandom3 RNG;

    public:
        XsecExtractor(const std::string& name, unsigned int seed);
        XsecExtractor(const std::string& name, const std::string& binning, unsigned int seed);

        std::string GetName() const;
        unsigned int GetNbins() const;

        int GetAnyBinIndex(const double D1, const double D2) const;

        void SetBinning(const std::string& binning);
        void SetFluxHist(const TH1D& h_flux);
        void SetFluxHist(const TH1D& h_flux, double err, ErrorType type = ErrorType::kRel);
        void SetFluxVar(double nom, double err, ErrorType type = ErrorType::kRel);
        void SetNumTargets(double ntargets, double nerror, ErrorType type = ErrorType::kRel);

        void ApplyBinWidths(TH1D& h_event_rate);
        void ApplyFluxInt(TH1D& h_event_rate, bool do_throw = false);
        void ApplyNumTargets(TH1D& h_event_rate, bool do_throw = false);

        double ThrowNtargets();
        double ThrowFlux();
};

#endif
