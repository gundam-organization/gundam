#include "FitParameters.hh"
using xsllh::FitBin;

FitParameters::FitParameters(const std::string& par_name, bool random_priors)
{
    m_name = par_name;
    m_rng_priors = random_priors;
}

FitParameters::~FitParameters()
{;}

bool FitParameters::SetBinning(const std::string& file_name, std::vector<FitBin>& bins)
{
    std::ifstream fin(file_name, std::ios::in);
    if(!fin.is_open())
    {
        std::cerr << "[ERROR]: In FitParameters::SetBinning()\n"
                  << "[ERROR]: Failed to open binning file: " << file_name << std::endl;
        return false;
    }

    else
    {
        std::string line;
        while(getline(fin, line))
        {
            std::stringstream ss(line);
            double D1_1, D1_2, D2_1, D2_2;
            if(!(ss>>D2_1>>D2_2>>D1_1>>D1_2))
            {
                std::cout << "[WARNING]: In FitParameters::SetBinning()\n"
                          << "[WARNING]: Bad line format: " << line << std::endl;
                continue;
            }
            bins.emplace_back(FitBin(D1_1, D1_2, D2_1, D2_2));
        }
        fin.close();

        std::cout << "[FitParameters]: Fit binning: \n";
        for(std::size_t i = 0; i < bins.size(); ++i)
        {
            std::cout << std::setw(3) << i
                      << std::setw(5) << bins[i].D2low
                      << std::setw(5) << bins[i].D2high
                      << std::setw(5) << bins[i].D1low
                      << std::setw(5) << bins[i].D1high << std::endl;
        }

        return true;
    }
}

int FitParameters::GetBinIndex(const std::string& det, double D1, double D2) const
{
    int bin = BADBIN;
    const std::vector<FitBin> &temp_bins = m_fit_bins.at(det);

    for(int i = 0; i < temp_bins.size(); ++i)
    {
        if(D1 >= temp_bins[i].D1low && D1 < temp_bins[i].D1high &&
           D2 >= temp_bins[i].D2low && D2 < temp_bins[i].D2high)
        {
            bin = i;
            break;
        }
    }
    return bin;
}


// initEventMap
void FitParameters::InitEventMap(std::vector<AnaSample*> &sample, int mode)
{
    for(const auto& s : sample)
    {
        if(m_fit_bins.count(s -> GetDetector()) == 0)
        {
            std::cerr << "[ERROR] In FitParameters::InitEventMap\n"
                      << "[ERROR] Detector " << s -> GetDetector() << " not part of fit parameters.\n"
                      << "[ERROR] Not building event map." << std::endl;
            return;
        }
    }

    InitParameters();
    m_evmap.clear();

    for(std::size_t s=0; s < sample.size(); s++)
    {
        std::vector<int> sample_map;
        for(int i=0; i < sample[s] -> GetN(); i++)
        {
            AnaEvent* ev = sample[s] -> GetEvent(i);

            // SIGNAL DEFINITION TIME
            // Warning, important hard coding up ahead:
            // This is where your signal is actually defined, i.e. what you want to extract an xsec for
            // N.B In Sara's original code THIS WAS THE OTHER WAY AROUND i.e. this if statement asked what was NOT your signal
            // Bare that in mind if you've been using older versions of the fitter.

            if(ev -> isSignalEvent())
            {
                double D1 = ev -> GetTrueD1();
                double D2 = ev -> GetTrueD2();
                int bin = GetBinIndex(sample[s] -> GetDetector(), D1, D2);
                if(bin == BADBIN)
                {
                    std::cout << "[WARNING]: " << m_name << ", Event: " << i << std::endl
                              << "[WARNING]: D1 = " << D1 << ", D2 = " << D2 << ", falls outside bin ranges." << std::endl
                              << "[WARNING]: This event will be ignored in the analysis." << std::endl;
                }
                sample_map.push_back(bin);
            }
            else
            {
                sample_map.push_back(PASSEVENT);
                continue;
            }

        }
        m_evmap.push_back(sample_map);
    }
}

void FitParameters::ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent, std::vector<double> &params)
{
    if(m_evmap.empty()) //need to build an event map first
    {
        std::cerr << "[ERROR]: In FitParameters::ReWeight()\n"
                  << "[ERROR]: Need to build event map index for " << m_name << std::endl;
        return;
    }

    const int bin = m_evmap[nsample][nevent];

    //skip event if not Signal
    if(bin == PASSEVENT) return;

    // If bin fell out of valid ranges, pretend the event just didn't happen:
    if(bin == BADBIN)
        event -> AddEvWght(0.0);
    else
    {
        if(bin > params.size())
        {
            std::cout << "[WARNING]: In FitParameters::ReWeight()\n"
                      << "[WARNING]: Number of bins in " << m_name << " does not match num of parameters.\n"
                      << "[WARNING]: Setting event weight to zero." << std::endl;
            event -> AddEvWght(0.0);
        }

        if(m_fit_bins.count(det) == true)
            event -> AddEvWght(params[bin + m_det_offset.at(det)]);
    }
}

void FitParameters::InitParameters()
{
    double rand_prior = 0.0;
    TRandom3 rng(0);

    unsigned int offset = 0;
    for(const auto& det : v_detectors)
    {
        m_det_offset.emplace(std::make_pair(det, offset));
        const int nbins = m_fit_bins.at(det).size();
        for(int i = 0; i < nbins; ++i)
        {
            pars_name.push_back(Form("%s_%s_%d", m_name.c_str(), det.c_str(), i));
            if(m_rng_priors == true)
            {
                //rand_prior = 2.0 * rng.Uniform(0.0, 1.0);
                rand_prior = rng.Gaus(1.0, 0.15);
                pars_prior.push_back(rand_prior);
            }
            else
                pars_prior.push_back(1.0); //all weights are 1.0 a priori

            pars_step.push_back(0.05);
            pars_limlow.push_back(0.0);
            pars_limhigh.push_back(10.0);
        }

        std::cout << "[FitParameters]: Total " << nbins << " parameters at "
                  << offset << " for " << det << std::endl;
        offset += nbins;
    }

    Npar = pars_name.size();
}

void FitParameters::AddDetector(const std::string& det, const std::string& f_binning)
{
    std::cout << "[FitParameters]: Adding detector " << det << " for " << m_name << std::endl;

    std::vector<FitBin> temp_vector;
    if(SetBinning(f_binning, temp_vector))
    {
        m_fit_bins.emplace(std::make_pair(det, temp_vector));
        v_detectors.emplace_back(det);
    }
    else
        std::cout << "[WARNING]: Adding detector failed." << std::endl;
}
