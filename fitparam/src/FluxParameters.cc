#include "FluxParameters.hh"

FluxParameters::FluxParameters(const std::string& name)
{
    m_name = name;
}

FluxParameters::~FluxParameters() { ; }

int FluxParameters::GetBinIndex(const std::string& det, double enu)
{
    int bin = BADBIN;
    const std::vector<double> temp_bins = m_det_bins.at(det);

    for(std::size_t i = 0; i < (temp_bins.size() - 1); ++i)
    {
        if(enu >= temp_bins[i] && enu < temp_bins[i + 1])
        {
            bin = i;
            break;
        }
    }
    return bin;
}

void FluxParameters::InitEventMap(std::vector<AnaSample*>& sample, int mode)
{
    for(const auto& s : sample)
    {
        if(m_det_bins.count(s->GetDetector()) == 0)
        {
            std::cerr << ERR << "In FluxParameters::InitEventMap\n"
                      << ERR << "Detector " << s->GetDetector() << " not part of fit parameters.\n"
                      << ERR << "Not building event map." << std::endl;
            return;
        }
    }

    InitParameters();
    m_evmap.clear();
    // loop over events to build index map
    for(std::size_t s = 0; s < sample.size(); ++s)
    {
        std::vector<int> sample_map;
        for(int i = 0; i < sample[s]->GetN(); ++i)
        {
            AnaEvent* ev = sample[s]->GetEvent(i);
            double enu   = ev->GetTrueEnu() / 1000.0; //MeV -> GeV
            int bin      = GetBinIndex(sample[s]->GetDetector(), enu);

            if(bin == BADBIN)
            {
                std::cout << WAR << "Event Enu " << enu << " falls outside bin range.\n"
                          << WAR << "This event will be ignored in the analysis." << std::endl;
                ev->Print();
            }
            // If event is signal let the c_i params handle the reweighting:
            if(mode == 1 && ev->isSignalEvent())
                bin = PASSEVENT;

            sample_map.push_back(bin);
        } // event loop
        m_evmap.push_back(sample_map);
    } // sample loop
}

void FluxParameters::ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent,
                              std::vector<double>& params)
{
    if(m_evmap.empty()) // need to build an event map first
    {
        std::cerr << ERR << "In FluxParameters::ReWeight()\n"
                  << ERR << "Need to build event map index for " << m_name << std::endl;
        return;
    }

    const int bin = m_evmap[nsample][nevent];

    // skip event if not Signal
    if(bin == PASSEVENT)
        return;

    // If bin fell out of valid ranges, pretend the event just didn't happen:
    if(bin == BADBIN)
        event->AddEvWght(0.0);
    else
    {
        if(bin > params.size())
        {
            std::cout << WAR << "In FluxParameters::ReWeight()\n"
                      << WAR << "Number of bins in " << m_name
                      << " does not match num of parameters.\n"
                      << WAR << "Setting event weight to zero." << std::endl;
            event->AddEvWght(0.0);
        }

        if(m_det_bins.count(det) == true)
        {
            event->AddEvWght(params[bin + m_det_offset.at(det)]);
            // std::cout << "Offset: " << m_det_offset.at(det) << std::endl;
        }
    }
}

void FluxParameters::InitParameters()
{
    unsigned int offset = 0;
    std::cout << TAG << "Flux binning " << std::endl;
    for(const auto& det : v_detectors)
    {
        std::cout << TAG << "Detector - " << det << std::endl;
        m_det_offset.insert(std::make_pair(det, offset));
        const int nbins = m_det_bins.at(det).size() - 1;
        for(int i = 0; i < nbins; ++i)
        {
            pars_name.push_back(Form("%s_%s_%d", m_name.c_str(), det.c_str(), i));
            pars_prior.push_back(1.0); // all weights are 1.0 a priori
            pars_step.push_back(0.1);
            pars_limlow.push_back(0.0);
            pars_limhigh.push_back(5.0);
            pars_fixed.push_back(false);

            std::cout << i << ": " << m_det_bins.at(det).at(i) << std::endl;
        }
        std::cout << nbins << ": " << m_det_bins.at(det).back() << std::endl;

        std::cout << TAG << "Total " << nbins << " parameters at "
                  << offset << " for " << det << std::endl;
        offset += nbins;
    }

    Npar = pars_name.size();
    pars_original = pars_prior;

    if(m_decompose)
    {
        pars_prior = eigen_decomp -> GetDecompParameters(pars_prior);
        pars_limlow = std::vector<double>(Npar, -100);
        pars_limhigh = std::vector<double>(Npar, 100);

        const int idx = eigen_decomp -> GetInfoFraction(m_info_frac);
        for(int i = idx; i < Npar; ++i)
            pars_fixed[i] = true;

        std::cout << TAG << "Decomposed parameters.\n"
                  << TAG << "Keeping the " << idx << " largest eigen values.\n"
                  << TAG << "Corresponds to " << m_info_frac * 100.0
                  << "\% total variance.\n";
    }
}

void FluxParameters::AddDetector(const std::string& det, const std::vector<double>& bins)
{
    std::cout << TAG << "Adding detector " << det << " for " << this->m_name
              << std::endl;
    m_det_bins.emplace(std::make_pair(det, bins));
    v_detectors.emplace_back(det);
}
