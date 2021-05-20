#include "../include/DetParameters.hh"
using xsllh::FitBin;

DetParameters::DetParameters(const std::string& name)
{
    m_name = name;
}

DetParameters::~DetParameters() { ; }

bool DetParameters::SetBinning(AnaSample* sample_, std::vector<GeneralizedFitBin>& bins)
{
    std::ifstream fin(sample_->GetDetBinning(), std::ios::in);
    if(!fin.is_open())
    {
        std::cerr << ERR << "In DetParameters::SetBinning()\n"
                  << ERR << "Failed to open binning file: " << sample_->GetDetBinning() << std::endl;
        return false;
    }
    else
    {
        std::string line;
        while(getline(fin, line))
        {
            std::stringstream ss(line);
            std::vector<double> lowEdges;
            std::vector<double> highEdges;

            double lowEdge, highEdge;
            while( ss >> lowEdge >> highEdge ){
                lowEdges.emplace_back( lowEdge );
                highEdges.emplace_back( highEdge );
            }

            if( sample_->GetFitPhaseSpace().size() != lowEdges.size() ){
                LogWarning << "Bad bin: \"" << line << "\"" << std::endl;
                continue;
            }

            bins.emplace_back(GeneralizedFitBin(lowEdges, highEdges));
        }
        fin.close();

        return true;
    }
}

int DetParameters::GetBinIndex(const int sampleIndex_, const std::vector<double>& eventVarList_) const
{
    const std::vector<GeneralizedFitBin> & sampleBinning = m_sample_bins.at(sampleIndex_);

    for( size_t iBin = 0 ; iBin < sampleBinning.size() ; iBin++ ){
        if( sampleBinning[iBin].isInBin(eventVarList_) ){
            return iBin;
        }
    }
    return BADBIN;
}

void DetParameters::InitEventMap(std::vector<AnaSample*>& samplesList, int mode)
{
    InitParameters();
    m_evmap.clear();

    if(mode == 2)
        std::cout << TAG << "Not using detector reweighting." << std::endl;

    for(auto & sample : samplesList)
    {
        std::vector<int> sample_map;
        for(int iEvent = 0; iEvent < sample->GetN(); ++iEvent)
        {
            AnaEvent* ev = sample->GetEvent(iEvent);

            std::vector<double> eventVarBuffer(sample->GetFitPhaseSpace().size(),0);
            for( size_t iVar = 0 ; iVar < sample->GetFitPhaseSpace().size() ; iVar++ ){
                eventVarBuffer[iVar] = double(ev->GetEventVarFloat(sample->GetFitPhaseSpace()[iVar]));
            }
            int bin   = GetBinIndex(sample->GetSampleID(), eventVarBuffer);

#ifndef NDEBUG
            if(bin == BADBIN)
            {
                std::cout << WAR << m_name << ", Event: " << iEvent << std::endl;
                std::cout << WAR;
                for( size_t iVar = 0 ; iVar < sample->GetFitPhaseSpace().size() ; iVar++ ){
                    if( iVar != 0 ) std::cout << ", ";
                    std::cout << sample->GetFitPhaseSpace()[iVar] << " = " << eventVarBuffer[iVar];
                }
                std::cout << ", falls outside bin ranges." << std::endl;
                std::cout << WAR << "This event will be ignored in the analysis." << std::endl;
            }
#endif
            // If event is signal let the c_i params handle the reweighting:
            if(mode == 1 and ev->isSignalEvent() ){
                bin = PASSEVENT;
            }
            else if(mode == 2){
                bin = PASSEVENT;
            }

            sample_map.push_back(bin);
        }
        m_evmap.push_back(sample_map);
    }
}

// Multiplies the current event weight for AnaEvent* event with the detector parameter for the sample and reco bin that this event falls in:
void DetParameters::ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent, std::vector<double>& params)
{
    // m_evmap is a vector containing vectors of which bin an event falls in for all samples. This event map needs to be built first, otherwise an error is thrown:
    if(m_evmap.empty())
    {
        std::cerr << ERR << "In DetParameters::ReWeight()\n"
                  << ERR << "Need to build event map index for " << m_name << std::endl;
        return;
    }

    // Get the bin that this event falls in:
    const int bin = m_evmap[nsample][nevent];

    // Event is skipped if it isn't signal (if bin = PASSEVENT = -1):
    if(bin == PASSEVENT)
        return;
    
    // If the bin fell out of the valid bin ranges (if bin = BADBIN = -2), we assign an event weight of 0 and pretend the event just didn't happen:
    if(bin == BADBIN)
        event->AddEvWght(0.0);

    // Otherwise, we multiply the event weight with the parameter for this sample, signal and reco bin:
    else
    {
        // If the bin number is larger than the number of parameters, we set the event weight to zero (this should not happen):
        if(bin > params.size())
        {
            std::cout << WAR << "In DetParameters::ReWeight()\n"
                      << WAR << "Number of bins in " << m_name << " does not match num of parameters.\n"
                      << WAR << "Setting event weight to zero." << std::endl;
            event->AddEvWght(0.0);
        }

        // Multiply the current event weight by the parameter for the reco bin and signal that this event falls in (defined in AnaEvent.hh):
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
            pars_name.emplace_back(Form("%s_sam%d_%d", m_name.c_str(), sam, i));
            pars_prior.push_back(1.0);
            pars_step.push_back(0.05);
            pars_limlow.push_back(0.0);
            pars_limhigh.push_back(2.0);
            pars_fixed.push_back(false);
        }

        std::cout << TAG << "Total " << nbins << " parameters at "
                  << offset << " for sample ID " << sam << std::endl;
        offset += nbins;
    }

    Npar = pars_name.size();
    pars_original = pars_prior;

    if(m_decompose)
    {
        pars_prior = eigen_decomp -> GetDecompParameters(pars_prior);
        pars_limlow = std::vector<double>(Npar, -100);
        pars_limhigh = std::vector<double>(Npar, 100);

        int idx = eigen_decomp -> GetInfoFraction(m_info_frac);
        if(idx > Npar - m_nb_dropped_dof){
            idx = int(Npar) - m_nb_dropped_dof;
        }
        for(int i = idx; i < Npar; ++i)
            pars_fixed[i] = true;

        std::cout << TAG << "Decomposed parameters.\n"
                  << TAG << "Keeping the " << idx << " largest eigen values.\n"
                  << TAG << "Corresponds to " << m_info_frac * 100.0
                  << "\% total variance.\n";
    }
}

void DetParameters::AddDetector(const std::string& det, std::vector<AnaSample*>& v_sample, bool match_bins)
{
    std::cout << TAG << "Adding detector " << det << " for " << m_name << std::endl;

    for(const auto& sample : v_sample)
    {
        if(sample->GetDetector() != det)
            continue;

        const int sample_id = sample->GetSampleID();
        v_samples.emplace_back(sample_id);

        std::cout << TAG << "Adding sample " << sample->GetName()
                  << " with ID " << sample_id << " to fit." << std::endl;

        if(match_bins)
            m_sample_bins.emplace(std::make_pair(sample_id, sample->GetBinEdges()));
        else
        {
            std::vector<GeneralizedFitBin> temp_vector;
            if(SetBinning(sample, temp_vector))
            {
                m_sample_bins.emplace(std::make_pair(sample_id, temp_vector));
            }
            else
                std::cout << WAR << "Adding sample binning for DetParameters failed." << std::endl;
        }
    }
}
