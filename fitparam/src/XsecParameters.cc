#include "XsecParameters.hh"
#include "GenericToolbox.h"
#include "Logger.h"

XsecParameters::XsecParameters(const std::string& name)
{
    m_name = name;
    Npar = 0;
    Logger::setUserHeaderStr("[XsecParameters]");
}

XsecParameters::~XsecParameters() { ; }

void XsecParameters::InitEventMap(std::vector<AnaSample*>& sample, int mode)
{
    LogWarning << "Initializing Event Map..." << std::endl;

    InitParameters();

    if(Npar == 0)
    {
        LogError << "In XsecParameters::InitEventMap\n"
                 << "No parameters delcared. Not building event map."
                 << std::endl;
    }

    m_dial_evtmap.clear();
    for(std::size_t iSample = 0; iSample < sample.size(); ++iSample)
    {
        LogInfo << "Mapping events in sample: " << sample[iSample]->GetName() << " (" << sample[iSample] -> GetN() << " events)" << std::endl;

        std::vector<std::vector<int>> sample_map;
//        #pragma omp parallel for num_threads(_nb_threads_)
        for(int iEvent = 0; iEvent < sample[iSample] -> GetN(); iEvent++)
        {
            GenericToolbox::displayProgressBar(iEvent, sample[iSample] -> GetN(), LogInfo.getPrefixString() + "Reading sample events");

            AnaEvent* anaEvent = sample[iSample] -> GetEvent(iEvent);
            std::vector<int> dial_index_map;

            std::vector<XsecDial>* v_dials = &m_dials.at(sample[iSample] -> GetDetector());
            int num_dials = v_dials->size();

            for(int iDial = 0; iDial < num_dials; ++iDial)
            {
                int idx;
                if(v_dials->at(iDial).GetUseSplineSplitMapping()){
                    // event mapping is done internaly
                    idx = v_dials->at(iDial).GetSplineIndex(anaEvent);
                }
                else{
                    double q2 = anaEvent-> GetQ2True() / 1.0E6; //MeV to GeV conversion.
                    //idx = v_dials.at(iDial).GetSplineIndex(anaEvent -> GetTopology(), anaEvent -> GetReaction(), q2);

                    //idx = v_dials.at(iDial).GetSplineIndex(std::vector<int>{anaEvent -> GetTopology(), anaEvent -> GetReaction()},
                    //                                       std::vector<double>{q2});
//                idx = v_dials.at(iDial).GetSplineIndex(std::vector<int>{anaEvent->GetTopology(), anaEvent->GetReaction()},
//                                                       std::vector<double>{anaEvent->GetTrueD2(), anaEvent->GetTrueD1()});
                    idx = v_dials->at(iDial).GetSplineIndex(std::vector<int>{anaEvent->GetSampleType(), anaEvent->GetReaction()},
                                                           std::vector<double>{anaEvent->GetTrueD2(), anaEvent->GetTrueD1()});

                    if(idx == BADBIN)
                    {
                        LogWarning << "Event falls outside spline range.\n"
                                   << "This event will be ignored in the analysis."
                                   << std::endl;
                        anaEvent-> AddEvWght(0.0);
                    }

                    if(mode == 1 && anaEvent-> isSignalEvent())
                        idx = PASSEVENT;
                }
                dial_index_map.emplace_back(idx);
            } // iDial

            #pragma omp critical
            {
            sample_map.emplace_back(dial_index_map);
            }
        } // iEvent

        m_dial_evtmap.emplace_back(sample_map);
    } // iSample

}

void XsecParameters::InitParameters()
{
    unsigned int offset = 0;
    for(const auto& det : v_detectors)
    {
        m_offset.insert(std::make_pair(det, offset));
        for(const auto& d : m_dials.at(det))
        {
            pars_name.push_back(Form("%s_%s", det.c_str(), d.GetName().c_str()));
            pars_prior.push_back(d.GetNominal());
            pars_step.push_back(d.GetStep());
            pars_limlow.push_back(d.GetLimitLow());
            pars_limhigh.push_back(d.GetLimitHigh());
            pars_fixed.push_back(false);

            LogInfo << "Added " << det << "_" << d.GetName()
                      << std::endl;
        }

        LogInfo << "Total " << m_dials.at(det).size() << " parameters at "
                  << offset << " for " << det << std::endl;

        offset += m_dials.at(det).size();
    }

    Npar = pars_name.size();
    pars_original = pars_prior;

    if(m_decompose) {
        LogDebug << "Decomposing covariance matrix..." << std::endl;
        pars_prior = eigen_decomp -> GetDecompParameters(pars_prior);
        pars_limlow = std::vector<double>(Npar, -100);
        pars_limhigh = std::vector<double>(Npar, 100);

        const int idx = eigen_decomp -> GetInfoFraction(m_info_frac);
        for(int i = idx; i < Npar; ++i)
            pars_fixed[i] = true;

        LogInfo << "Decomposed parameters.\n"
                << "Keeping the " << idx << " largest eigen values.\n"
                << "Corresponds to " << m_info_frac * 100.0
                << "\% total variance.\n";
    }
}

void XsecParameters::ReWeight(AnaEvent* event, const std::string& detectorName, int nsample, int nevent, std::vector<double>& params)
{
    if(m_dial_evtmap.empty()) // need to build an event map first
    {
        LogError << "In XsecParameters::ReWeight()\n"
                 << "Need to build event map index for " << m_name << std::endl;
        return;
    }

    std::vector<XsecDial> &v_dials = m_dials.at(detectorName);
    int num_dials = v_dials.size();
    double weight = 1.0;

    for(int iDial = 0; iDial < num_dials; ++iDial)
    {
        int idx = m_dial_evtmap[nsample][nevent][iDial];
        double dial_weight = v_dials[iDial].GetBoundedValue(idx, params[iDial + m_offset.at(detectorName)]);

        if(dial_weight == 0){
            LogFatal << "dial_weight is 0" << std::endl;
            throw std::logic_error("0");
        }

        weight *= dial_weight;

        /*
        if(dial_weight > 3.0)
        {
            std::cout << "--------------" << std::endl;
            std::cout << "Ev T: " << event -> GetTopology() << std::endl
                      << "Ev R: " << event -> GetReaction() << std::endl
                      << "Ev Q: " << event -> GetQ2() << std::endl;
            std::cout << "Ev I: " << idx << std::endl;
            std::cout << "Ev W: " << dial_weight << std::endl;
            std::cout << "Dl V: " << params[iDial + m_offset.at(detectorName)] << std::endl;
            std::cout << "Dl N: " << detectorName << "_" << v_dials[iDial].GetName() << std::endl;
            std::cout << "Sp N: " << v_dials[iDial].GetSplineName(idx) << std::endl;
        }
        */
    }
//    std::cout << "weight=" << weight << std::endl;

    if(m_do_cap_weights)
        weight = weight > m_weight_cap ? m_weight_cap : weight;

    event -> AddEvWght(weight);
}

void XsecParameters::AddDetector(const std::string& det, const std::string& config)
{
    LogInfo << "Adding detector " << det << " for " << m_name << std::endl;
    std::fstream f;
    LogInfo << "Opening config file " << config << std::endl;
    f.open(config, std::ios::in);

    json j;
    f >> j;

    std::string input_dir = std::string(std::getenv("XSLLHFITTER"))
                            + j["input_dir"].get<std::string>();
    LogInfo << "Adding the following dials." << std::endl;

    std::vector<int> global_dimensions;
    if(j.find("subject_id") != j.end()){
        std::vector<int> global_dimensions = j["dimensions"].get<std::vector<int>>();
    }

    std::vector<XsecDial> v_dials;
    for(const auto& dial : j["dials"])
    {
        if(dial["use"] == true)
        {


            std::string fname_splines;
            std::string fname_binning;
            if(dial.find("splines") != dial.end()){
                fname_binning = input_dir + dial["binning"].get<std::string>();
                fname_splines = input_dir + dial["splines"].get<std::string>();
            }

            XsecDial x(dial["name"], fname_binning, fname_splines);
            x.SetVars(dial["nominal"], dial["step"], dial["limit_lo"], dial["limit_hi"]);
            x.Print(false);
            if(not global_dimensions.empty()){
                std::vector<int> dimensions = dial.value("dimensions", global_dimensions);
                x.SetDimensions(dimensions);
            }

            v_dials.emplace_back(x);
        }
    }

    v_detectors.emplace_back(det);
    m_dials.insert(std::make_pair(det, v_dials));
}

void XsecParameters::SetNbThreads(int nbThreads_){
  _nb_threads_ = nbThreads_;
}
