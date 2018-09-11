#include "DetParameters.hh"
using xsllh::FitBin;

// ctor
DetParameters::DetParameters(const std::string& name)
{
    m_name       = name;
    hasRegCovMat = false;
}

// dtor
DetParameters::~DetParameters() { ; }

bool DetParameters::SetBinning(const std::string& file_name, std::vector<FitBin>& bins)
{
    std::ifstream fin(file_name, std::ios::in);
    if(!fin.is_open())
    {
        std::cerr << "[ERROR]: In DetParameters::SetBinning()\n"
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
                std::cout << "[WARNING]: In DetParameters::SetBinning()\n"
                          << "[WARNING]: Bad line format: " << line << std::endl;
                continue;
            }
            bins.emplace_back(FitBin(D1_1, D1_2, D2_1, D2_2));
        }
        fin.close();

        std::cout << "[DetParameters]: Fit binning: \n";
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

int DetParameters::GetBinIndex(const int sam, double D1, double D2) const
{
    int bin = BADBIN;
    const std::vector<FitBin> &temp_bins = m_sample_bins.at(sam);

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

void DetParameters::InitEventMap(std::vector<AnaSample*>& sample, int mode)
{
    InitParameters();
    m_evmap.clear();

    for(std::size_t s = 0; s < sample.size(); ++s)
    {
        std::vector<int> sample_map;
        for(int i = 0; i < sample[s]->GetN(); ++i)
        {
            AnaEvent* ev = sample[s]->GetEvent(i);
            double D1 = ev->GetRecD1();
            double D2 = ev->GetRecD2();
            int bin   = GetBinIndex(sample[s]->GetSampleID(), D1, D2);
#ifdef DEBUG_MSG
            if(bin == BADBIN)
            {
                std::cout << "[WARNING]: " << m_name << ", Event: " << i << std::endl
                          << "[WARNING]: D1 = " << D1 << ", D2 = " << D2 << ", falls outside bin ranges." << std::endl
                          << "[WARNING]: This event will be ignored in the analysis." << std::endl;
            }
#endif
            // If event is signal let the c_i params handle the reweighting:
            if(mode == 1 && ev->isSignalEvent())
                bin = PASSEVENT;
            sample_map.push_back(bin);
        }
        m_evmap.push_back(sample_map);
    }
}

void DetParameters::ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent, std::vector<double>& params)
{
    if(m_evmap.empty()) // need to build an event map first
    {
        std::cerr << "[ERROR]: In DetParameters::ReWeight()\n"
                  << "[ERROR]: Need to build event map index for " << m_name << std::endl;
        return;
    }

    const int bin = m_evmap[nsample][nevent];

    if(bin == PASSEVENT)
        return;
    if(bin == BADBIN)
        event->AddEvWght(0.0); // skip!!!!
    else
    {
        if(bin > params.size())
        {
            std::cout << "[WARNING]: In DetParameters::ReWeight()\n"
                      << "[WARNING]: Number of bins in " << m_name << " does not match num of parameters.\n"
                      << "[WARNING]: Setting event weight to zero." << std::endl;
            event->AddEvWght(0.0);
        }

        event->AddEvWght(params[bin + m_sample_offset.at(event->GetSampleType())]);
    }
}

void DetParameters::InitParameters()
{
    unsigned int offset = 0;
    for(const auto& sam : v_samples)
    {
        m_sample_offset.emplace(std::make_pair(sam, offset));
        const int nbins = m_sample_bins.at(sam).size();
        for(int i = 0; i < nbins; ++i)
        {
            pars_name.push_back(Form("%s_sam%d_%d", m_name.c_str(), sam, i));
            pars_prior.push_back(1.0);
            pars_step.push_back(0.05);
            pars_limlow.push_back(0.0);
            pars_limhigh.push_back(2.0);
        }

        std::cout << "[DetParameters]: Total " << nbins << " parameters at "
                  << offset << " for sample ID " << sam << std::endl;
        offset += nbins;
    }

    Npar = pars_name.size();
}

void DetParameters::AddDetector(const std::string& det, std::vector<AnaSample*>& v_sample, bool match_bins)
{
    std::cout << "[DetParameters]: Adding detector " << det << " for " << m_name << std::endl;

    for(const auto& sample : v_sample)
    {
        if(sample->GetDetector() != det)
            continue;

        const int sample_id = sample->GetSampleID();
        v_samples.emplace_back(sample_id);

        std::cout << "[DetParameters]: Adding sample " << sample->GetName()
                  << " with ID " << sample_id << " to fit." << std::endl;

        if(match_bins)
            m_sample_bins.emplace(std::make_pair(sample_id, sample->GetBinEdges()));
        else
        {
            std::vector<FitBin> temp_vector;
            if(SetBinning(sample->GetDetBinning(), temp_vector))
            {
                m_sample_bins.emplace(std::make_pair(sample_id, temp_vector));
            }
            else
                std::cout << "[WARNING]: Adding sample binning for DetParameters failed." << std::endl;
        }
    }
}
