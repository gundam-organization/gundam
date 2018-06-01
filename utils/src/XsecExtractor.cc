#include "XsecExtractor.hh"

XsecExtractor::XsecExtractor(const std::string& name, unsigned int seed)
    : m_name(name), RNG(seed)
{
}


XsecExtractor::XsecExtractor(const std::string& name, const std::string& binning,
                             unsigned int seed)
    : m_name(name), m_binning(binning), RNG(seed)
{
}

void XsecExtractor::SetNumTargets(double ntargets, double nerror)
{
    num_targets_val = ntargets;
    num_targets_err = nerror * ntargets;
}

void XsecExtractor::SetFluxHist(const TH1D& h_flux)
{
    h_weighted_flux = h_flux;
    flux_integral = 0;
    for(int i = 1; i <= h_weighted_flux.GetNbinsX(); ++i)
        flux_integral += h_weighted_flux.GetBinContent(i) * h_weighted_flux.GetBinWidth(i) / 0.05;
}

void XsecExtractor::SetBinning(const std::string& binning)
{
    m_binning = binning;
}

void XsecExtractor::ApplyBinWidths(TH1D& h_event_rate)
{
}

void XsecExtractor::ApplyNumTargets(TH1D& h_event_rate, bool do_throw)
{
    if(do_throw)
        h_event_rate.Scale(1.0 / RNG.Gaus(num_targets_val, num_targets_err));
    else
        h_event_rate.Scale(1.0 / num_targets_val);
}

void XsecExtractor::ApplyFluxInt(TH1D& h_event_rate, bool do_throw)
{
    if(do_throw)
        h_event_rate.Scale(1.0 / RNG.Gaus(flux_integral, flux_error));
    else
        h_event_rate.Scale(1.0 / flux_integral);
}

double XsecExtractor::ThrowNtargets()
{
    return RNG.Gaus(num_targets_val, num_targets_err);
}
