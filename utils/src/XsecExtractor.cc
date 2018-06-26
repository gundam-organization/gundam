#include "XsecExtractor.hh"
using xsllh::FitBin;

XsecExtractor::XsecExtractor(const std::string& name, unsigned int seed)
    : m_name(name), RNG(seed), m_nbins(0)
{
}


XsecExtractor::XsecExtractor(const std::string& name, const std::string& binning,
                             unsigned int seed)
    : m_name(name), RNG(seed), m_nbins(0)
{
    SetBinning(binning);
}

std::string XsecExtractor::GetName() const
{
    return m_name;
}

unsigned int XsecExtractor::GetNbins() const
{
    return m_nbins;
}

int XsecExtractor::GetAnyBinIndex(const double D1, const double D2) const
{
    for(int i = 0; i < bin_edges.size(); ++i)
    {
        if(D1 >= bin_edges[i].D1low && D1 < bin_edges[i].D1high &&
           D2 >= bin_edges[i].D2low && D2 < bin_edges[i].D2high)
        {
            return i;
        }
    }
    return -1;
}

void XsecExtractor::SetNumTargets(double ntargets, double nerror, ErrorType type)
{
    num_targets_val = ntargets;

    if(type == ErrorType::kRel)
        num_targets_err = nerror * ntargets;
    else if(type == ErrorType::kAbs)
        num_targets_err = nerror;
    else
        std::cout << "[XsecExtractor]: Invalid error type." << std::endl;
}

void XsecExtractor::SetFluxHist(const TH1D& h_flux)
{
    h_weighted_flux = h_flux;
    flux_integral = 0;
    for(int i = 1; i <= h_weighted_flux.GetNbinsX(); ++i)
        flux_integral += h_weighted_flux.GetBinContent(i) * h_weighted_flux.GetBinWidth(i) / 0.05;
}

void XsecExtractor::SetFluxHist(const TH1D& h_flux, double err, ErrorType type)
{
    h_weighted_flux = h_flux;
    flux_integral = 0;
    for(int i = 1; i <= h_weighted_flux.GetNbinsX(); ++i)
        flux_integral += h_weighted_flux.GetBinContent(i) * h_weighted_flux.GetBinWidth(i) / 0.05;

    if(type == ErrorType::kRel)
        flux_error = err * flux_integral;
    else if(type == ErrorType::kAbs)
        flux_error = err;
    else
        std::cout << "[XsecExtractor]: Invalid error type." << std::endl;
}
void XsecExtractor::SetFluxVar(double nom, double err, ErrorType type)
{
    flux_integral = nom;

    if(type == ErrorType::kRel)
        flux_error = err * nom;
    else if(type == ErrorType::kAbs)
        flux_error = err;
    else
        std::cout << "[XsecExtractor]: Invalid error type." << std::endl;
}

void XsecExtractor::SetBinning(const std::string& binning)
{
    m_binning = binning;
    m_nbins = 0;

    std::ifstream fin(m_binning, std::ios::in);
    if(!fin.is_open())
    {
        std::cerr << "[ERROR]: In XsecExtractor::SetBinning().\n"
                  << "[ERROR]: Failed to open binning file: " << m_binning << std::endl;
    }
    else
    {
        bin_widths.clear();
        std::string line;
        while(std::getline(fin, line))
        {
            std::stringstream ss(line);
            double D1_1, D1_2, D2_1, D2_2;
            if(!(ss>>D2_1>>D2_2>>D1_1>>D1_2))
            {
                std::cerr << "[XsecExtractor]: Bad line format: " << line << std::endl;
                continue;
            }
            double bw = std::abs(D1_2 - D1_1) * std::abs(D2_2 - D2_1);
            bw = bw / 1000.0;
            std::cout << "Bin Width: " << bw << std::endl;
            bin_widths.push_back(bw);
            bin_edges.emplace_back(FitBin(D1_1, D1_2, D2_1, D2_2));
        }
        m_nbins = bin_widths.size();
    }
}

void XsecExtractor::ApplyBinWidths(TH1D& h_event_rate)
{
    for(int i = 1; i <= h_event_rate.GetNbinsX(); ++i)
    {
        double bin = h_event_rate.GetBinContent(i) / bin_widths.at(i-1);
        h_event_rate.SetBinContent(i, bin);
    }
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

double XsecExtractor::ThrowFlux()
{
    return RNG.Gaus(flux_integral, flux_error);
}
