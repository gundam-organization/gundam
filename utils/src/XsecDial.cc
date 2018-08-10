#include "XsecDial.hh"

XsecDial::XsecDial(const std::string& dial_name)
    : m_name(dial_name), TAG("\033[92m[XsecDial]: \033[0m")
{
}

XsecDial::XsecDial(const std::string& dial_name, const std::string& fname_binning,
                   const std::string& fname_splines)
    : m_name(dial_name), TAG("\033[92m[XsecDial]: \033[0m")
{
    SetBinning(fname_binning);
    ReadSplines(fname_splines);
}

void XsecDial::SetBinning(const std::string& fname_binning)
{
    bin_manager.SetBinning(fname_binning);
    nbins = bin_manager.GetNbins();
}

void XsecDial::ReadSplines(const std::string& fname_splines)
{
    v_splines.clear();

    TFile* file_splines = TFile::Open(fname_splines.c_str(), "READ");
    if(file_splines == 0)
    {
        std::cout << "[ERROR]: Failed to open " << fname_splines << std::endl;
        return;
    }

    TIter key_list(file_splines -> GetListOfKeys());
    TKey* key = nullptr;
    while((key = (TKey*)key_list.Next()))
    {
        TGraph* spline = (TGraph*)key -> ReadObj();
        v_splines.emplace_back(*spline);
    }
    file_splines -> Close();
    delete file_splines;

    v_splines.shrink_to_fit();
}

int XsecDial::GetSplineIndex(int topology, int reaction, double q2) const
{
    const int b = bin_manager.GetBinIndex(std::vector<double>{q2});
    return topology * nreac * nbins + reaction * nbins + b;
}

double XsecDial::GetSplineValue(int index, double dial_value) const
{
    if(index >= 0)
        return v_splines.at(index).Eval(dial_value);
    else
        return 1.0;
}

std::string XsecDial::GetSplineName(int index) const
{
    return std::string(v_splines.at(index).GetName());
}

void XsecDial::SetVars(double nominal, double step, double limit_lo, double limit_hi)
{
    m_nominal = nominal;
    m_step = step;
    m_limit_lo = limit_lo;
    m_limit_hi = limit_hi;
}

void XsecDial::SetDimensions(int num_top, int num_reac)
{
    ntop = num_top;
    nreac = num_reac;
}

void XsecDial::Print(bool print_bins) const
{
    std::cout << TAG << "Name: " <<  m_name << std::endl
              << TAG << "Nominal: " << m_nominal << std::endl
              << TAG << "Step: " << m_step << std::endl
              << TAG << "Limits: [" << m_limit_lo << "," << m_limit_hi << "]" << std::endl
              << TAG << "Splines: " << v_splines.size() << std::endl;

    if(print_bins)
        bin_manager.Print();
}
