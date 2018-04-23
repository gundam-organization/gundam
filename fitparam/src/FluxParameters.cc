#include "FluxParameters.hh"

FluxParameters::FluxParameters(const std::string& name)
{
    m_name = name;
}

FluxParameters::~FluxParameters()
{;}

int FluxParameters::GetBinIndex(const std::string& det, double enu)
{
    int bin = BADBIN;
    std::vector<double> temp_bins = m_det_bins.at(det);

    for(std::size_t i = 0; i < (temp_bins.size()-1); ++i)
    {
        if(enu >= temp_bins[i] && enu < temp_bins[i+1])
        {
            bin = i;
            break;
        }
    }
    return bin;
}

void FluxParameters::InitEventMap(std::vector<AnaSample*> &sample, int mode)
{
    for(const auto& s : sample)
    {
        if(m_det_bins.count(s -> GetDetector()) == 0)
        {
            std::cerr << "[ERROR] In FluxParameters::InitEventMap\n"
                      << "[ERROR] Detector " << s -> GetDetector() << " not part of fit parameters.\n"
                      << "[ERROR] Not building event map." << std::endl;
            return;
        }
    }

    InitParameters();
    m_evmap.clear();
    //loop over events to build index map
    for(std::size_t s = 0; s < sample.size(); ++s)
    {
        std::vector<int> sample_map;
        for(int i = 0; i < sample[s] -> GetN(); ++i)
        {
            AnaEvent* ev = sample[s] -> GetEvent(i);
            double enu = ev -> GetTrueEnu();
            int bin = GetBinIndex(sample[s] -> GetDetector(), enu);

            if(bin == BADBIN)
            {
                std::cout << "[WARNING]: Event Enu " << enu << " falls outside bin range.\n"
                          << "[WARNING]: This event will be ignored in the analysis." << std::endl;
                ev -> Print();
            }
            //If event is signal let the c_i params handle the reweighting:
            if(mode == 1 && ev -> isSignalEvent())
                bin = PASSEVENT;

            sample_map.push_back(bin);
        }//event loop
        m_evmap.push_back(sample_map);
    }//sample loop
}

void FluxParameters::ReWeight(AnaEvent *event, const std::string& det, int nsample, int nevent,
        std::vector<double> &params)
{
    if(m_evmap.empty()) //need to build an event map first
    {
        std::cerr << "[ERROR]: In FluxParameters::ReWeight()\n"
                  << "[ERROR]: Need to build event map index for " << m_name << std::endl;
        return;
    }

    int bin = m_evmap[nsample][nevent];

    //skip event if not Signal
    if(bin == PASSEVENT) return;

    // If bin fell out of valid ranges, pretend the event just didn't happen:
    if(bin == BADBIN)
        event -> AddEvWght(0.0);
    else
    {
        if(bin > params.size())
        {
            std::cout << "[WARNING]: In FluxParameters::ReWeight()\n"
                      << "[WARNING]: Number of bins in " << m_name << " does not match num of parameters.\n"
                      << "[WARNING]: Setting event weight to zero." << std::endl;
            event -> AddEvWght(0.0);
        }

        if(m_det_bins.count(det) == true)
        {
            event -> AddEvWght(params[bin + m_det_offset.at(det)]);
            //std::cout << "Offset: " << m_det_offset.at(det) << std::endl;
        }
    }
}

void FluxParameters::InitParameters()
{
    if(m_det_bins.size() != m_det_offset.size())
    {
        std::cerr << "[ERROR]: In FluxParameters::InitParameters()\n"
                  << "[ERROR]: Detector bins and offset maps are not the same size!" << std::endl;
        return;
    }

    std::set<std::pair<std::string, int>, PairCompare> temp_set;
    for(const auto& kv : m_det_offset)
        temp_set.insert(kv);

    std::cout << "[FluxParameters]: Flux binning " << std::endl;
    for(const auto& pear : temp_set)
    {
        std::cout << "[FluxParameters]: Detector - " << pear.first << std::endl;
        const int nbins = m_det_bins.at(pear.first).size() - 1;
        for(int i = 0; i < nbins; ++i)
        {
            pars_name.push_back(Form("%s_%s_%d", m_name.c_str(), pear.first.c_str(), i));
            pars_prior.push_back(1.0); //all weights are 1.0 a priori
            pars_step.push_back(0.1);
            pars_limlow.push_back(0.0);
            pars_limhigh.push_back(5.0);

            std::cout << i << ": " << m_det_bins.at(pear.first).at(i) << std::endl;
        }
        std::cout << nbins << ": " << m_det_bins.at(pear.first).back() << std::endl;
    }

    Npar = pars_name.size();
}
