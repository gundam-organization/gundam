#include "../include/FluxParameters.hh"
#include "Logger.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[FluxParameters]");
} )

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
        //if(m_det_bins.count(s->GetDetector()) == 0)
        if(m_det_bm.count(s->GetDetector()) == 0)
        {
            LogError << "In FluxParameters::InitEventMap\n"
                     << "Detector " << s->GetDetector() << " not part of fit parameters.\n"
                     << "Not building event map." << std::endl;
            return;
        }
    }

    InitParameters();
    m_evmap.clear();
    // Loop over events to build index map:
    for(std::size_t s = 0; s < sample.size(); ++s)
    {
        std::vector<int> sample_map;
        for(int i = 0; i < sample[s]->GetN(); ++i)
        {
            AnaEvent* ev = sample[s]->GetEvent(i);
            //double enu   = ev->GetTrueEnu() / 1000.0; //MeV -> GeV
            double enu   = ev->GetTrueEnu();
            int nutype   = ev->GetFlavor();
            int beammode = ev->GetBeamMode();
            //int bin      = GetBinIndex(sample[s]->GetDetector(), enu);
            int bin      = m_det_bm.at(sample[s]->GetDetector()).GetBinIndex(std::vector<double>{enu}, nutype, beammode);

            if(bin == BADBIN)
            {
                LogWarning << "Event Enu " << enu << " falls outside bin range.\n"
                           << "This event will be ignored in the analysis." << std::endl;
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

// Multiplies the current event weight for AnaEvent* event with the correct flux parameter for the true neutrino energy bin that this event falls in:
void FluxParameters::ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent,
                              std::vector<double>& params)
{
    // m_evmap is a vector containing vectors of which bin an event falls in for all samples. This event map needs to be built first, otherwise an error is thrown:
    if(m_evmap.empty())
    {
        LogError << "In FluxParameters::ReWeight()\n"
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

    // Otherwise, we multiply the event weight with the parameter for this neutrino energy:
    else
    {
        // If the bin number is larger than the number of parameters, we set the event weight to zero (this should not happen):
        if(bin > params.size())
        {
            LogWarning << "In FluxParameters::ReWeight()\n"
                       << "Number of bins in " << m_name
                       << " does not match num of parameters.\n"
                       << "Setting event weight to zero." << std::endl;
            event->AddEvWght(0.0);
        }

        // If the detector key is present in the m_det_bins map, we multiply the event weight with the parameter for the energy bin that this event falls in:
        //if(m_det_bins.count(det) == true)
        if(m_det_bm.count(det) == true)
        {
            // Multiply the current event weight by the parameter for the energy bin that this event falls in (defined in AnaEvent.hh):
            event->AddEvWght(params[bin + m_det_offset.at(det)]);
            // std::cout << "Offset: " << m_det_offset.at(det) << std::endl;
        }
    }
}

void FluxParameters::InitParameters()
{
    unsigned int offset = 0;
    LogInfo << "Flux binning " << std::endl;

    for( size_t iDetector = 0 ; iDetector < v_detectors.size() ; iDetector++ ){
        LogInfo << "Detector: "<< v_detectors[iDetector] << std::endl;
        m_det_offset.insert(std::make_pair(v_detectors[iDetector], offset));
        //const int nbins = m_det_bins.at(v_detectors[iDetector]).size() - 1;
        const int nbins = m_det_bm.at(v_detectors[iDetector]).GetNbins();
        for( int iBin = 0 ; iBin < nbins ; iBin++ ){
            pars_name.emplace_back(
                Form("%s_%s_%i",
                     m_name.c_str(), v_detectors[iDetector].c_str(), iBin
//                     , m_det_bm.at(v_detectors[iDetector]).GetBinNuTypeList()[iDetector][iBin]
//                     , m_det_bm.at(v_detectors[iDetector]).GetBinBeamModeList()[iDetector][iBin]
                     )
                );
            pars_prior.push_back(1.0); // all weights are 1.0 a priori
            pars_step.push_back(0.1);
            pars_limlow.push_back(0.0);
            pars_limhigh.push_back(5.0);
            pars_fixed.push_back(false);
        }

        m_det_bm.at(v_detectors[iDetector]).Print();
        LogInfo << "Total " << nbins << " parameters at "
                << offset << " for " << v_detectors[iDetector] << std::endl;
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

        LogInfo << "Decomposed parameters.\n"
                  << TAG << "Keeping the " << idx << " largest eigen values.\n"
                  << TAG << "Corresponds to " << m_info_frac * 100.0
                  << "\% total variance.\n";
    }
}

void FluxParameters::AddDetector(const std::string& det, const std::vector<double>& bins)
{
    LogInfo << "Adding detector " << det << " for " << this->m_name
              << std::endl;
    m_det_bins.emplace(std::make_pair(det, bins));
    v_detectors.emplace_back(det);
}

void FluxParameters::AddDetector(const std::string& det, const std::string& binning_file)
{
    LogInfo << "Adding detector " << det << " for " << this->m_name
              << std::endl;
    BinManager temp(binning_file, true);
    m_det_bm.emplace(std::make_pair(det, std::move(temp)));
    v_detectors.emplace_back(det);
}

void FluxParameters::Print() {

    LogInfo << m_name << std::endl;
    for( size_t iDetector = 0 ; iDetector < v_detectors.size() ; iDetector++ ){
        LogInfo << "Detector: "<< v_detectors[iDetector] << std::endl;
        const int nbins = m_det_bm.at(v_detectors[iDetector]).GetNbins();
        for( int iBin = 0 ; iBin < nbins ; iBin++ ){
            this->PrintParameterInfo(iDetector, iBin);
        }
    }

}

void FluxParameters::PrintParameterInfo(int iPar_) {

    int detectorIndex = -1;
    int binIndex = -1;
    int parOffset     = 0;
    for( size_t iDetector = 0 ; iDetector < v_detectors.size() ; iDetector++ ){
        const int nbins = m_det_bm.at(v_detectors[iDetector]).GetNbins();
        if( iPar_ < parOffset + nbins ){
            detectorIndex = iDetector;
            binIndex = iPar_ - parOffset;
            break;
        }
        parOffset += nbins;
    }

    if( detectorIndex != -1 and binIndex != -1 ){
        this->PrintParameterInfo(detectorIndex, binIndex);
    }

}

void FluxParameters::PrintParameterInfo(int iDetector_, int iBin_){

    LogInfo << "Parameter Index: " << m_det_offset[v_detectors[iDetector_]] + iBin_ << std::endl;
    LogInfo << "Parameter Name: " << pars_name[m_det_offset[v_detectors[iDetector_]] + iBin_] << std::endl;
    LogInfo << "Kinematic Bin: " << m_det_bm.at(v_detectors[iDetector_]).GetEdgeVector(iDetector_)[iBin_].first
            << " - " << m_det_bm.at(v_detectors[iDetector_]).GetEdgeVector(iDetector_)[iBin_].second << std::endl;
    LogInfo << "NuType: " << m_det_bm.at(v_detectors[iDetector_]).GetBinNuTypeList()[iDetector_][iBin_] << std::endl;
    LogInfo << "BeamMode: " << m_det_bm.at(v_detectors[iDetector_]).GetBinBeamModeList()[iDetector_][iBin_] << std::endl;

}
