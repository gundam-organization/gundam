#include "XsecParameters.hh"

// ctor
XsecParameters::XsecParameters(const std::string& name)
{
    m_name       = name;
    hasRegCovMat = false;
    Npar         = 0;
}

// dtor
XsecParameters::~XsecParameters() { ; }

// So much upsetting hard coding here, sorry!
// So below I've hacked in a fix to avoid the FSI params looking at cut branch (sample) 0 or 4
// (since these have no protons)
// and reaction 6 has been made to be reaaction 7 (since reaction 6 is not defined in mectopology
// HL2 cat)

// The dummy response function thing is also a bit of a hack. This allows the occasional job to fail
// when making resp functions and the fitter still runs but this is not good and should be delt with

// store response functions in vector of Xsec "bins" (Ereco, Etrue, reac, topo)
void XsecParameters::StoreResponseFunctions(std::vector<TFile*> respfuncs,
                                            std::vector<std::pair<double, double>> v_D1edges,
                                            std::vector<std::pair<double, double>> v_D2edges)
{
    double dummyx[7] = {-1, -0.66, -0.33, 0, 0.33, 0.66, 1};
    double dummyy[7] = {1, 1, 1, 1, 1, 1, 1};
    int dummyn       = 7;

    for(int stInt = 0; stInt < 8; stInt++)
    {
        if((stInt == 0) || (stInt == 4))
            continue; // Ignore branches with no proton
        SampleTypes sampletype = static_cast<SampleTypes>(stInt);
        for(int rtInt = 0; rtInt < 8; rtInt++)
        {
            ReactionTypes reactype = static_cast<ReactionTypes>(rtInt);
            if(rtInt == 6)
                continue; // Hack to deal with missing mectopo6
            // cout<<"reading response functions for topology "<<stInt<<"  reaction "<<rtInt<<endl;
            int nccqebins = v_D1edges.size();
            for(int br = 0; br < nccqebins; br++)
            { // reco kinematics bin
                // cout<<"reading rewighting function for reco bin "<<br<<endl;
                for(int bt = 0; bt < nccqebins; bt++)
                { // true kinematics bin
                    // cout<<"reading rewighting function for true bin "<<bt<<endl;
                    XsecBin bin;
                    bin.recoD1low  = v_D1edges[br].first;
                    bin.recoD1high = v_D1edges[br].second;
                    bin.trueD1low = v_D1edges[bt].first; // same binning for reco and true
                    bin.trueD1high = v_D1edges[bt].second;
                    bin.recoD2low  = v_D2edges[br].first;
                    bin.recoD2high = v_D2edges[br].second;
                    bin.trueD2low = v_D2edges[bt].first; // same binning for reco and true
                    bin.trueD2high = v_D2edges[bt].second;
                    bin.topology   = sampletype;
                    bin.reaction   = reactype;
                    if(fabs(br - bt) < 21)
                    { // save memory if reco bin and true bin very far away
                        for(uint i = 0; i < Npar; i++)
                        {
                            char name[200];
                            sprintf(name, "topology_%d/RecBin_%d_trueBin_%d_topology_%d_reac_%d",
                                    stInt, br, bt, stInt, rtInt);
                            // cout<<respfuncs[i]->GetName()<<" "<<name<<endl;
                            TGraph* g = (TGraph*)respfuncs[i]->Get(name);
                            // **** Temp remove effect of CA5 since highly corrolated with MARES
                            // **** if(i==0) g = new TGraph(dummyn, dummyx, dummyy);
                            // ********
                            if(!g)
                            {
                                sprintf(name, "topology_%d/trueBin_%d_topology_%d_reac_%d", stInt,
                                        bt, stInt, rtInt);
                                g = (TGraph*)respfuncs[i]->Get(name);
                            }
                            // cout<<g<<endl;
                            if(!g)
                            {
                                if(rtInt != 0 && rtInt < 6 && i < 9)
                                { // Ignore OOFV splines, don't bother with warnings for lack of
                                  // CCQE params
                                    std::cout << "WARNING: creating dummy respfunc, param: " << i
                                              << " " << name << std::endl;
                                    // cout << "getchar to cont" << endl;
                                    // getchar();
                                }
                                g = new TGraph(dummyn, dummyx, dummyy);
                            }
                            g->SetName(name);
                            if((g->GetY())[3] != 1.0)
                            {
                                std::cout << "WARNING: altering xsec nominal param: " << i << " "
                                          << name << std::endl;
                            }
                            bin.respfuncs.push_back(g);
                        }
                    }
                    m_bins.push_back(bin);
                }
            }
        }
    }

    /*for(size_t j=0; j<m_bins.size();j++){
      cout<<j<<" topology: "<<m_bins[j].topology<<"  reaction: "<<m_bins[j].reaction
      <<"  recoP: "<<m_bins[j].recoD1low<<"-"<<m_bins[j].recoD1high
      <<"  trueP: "<<m_bins[j].trueD1low<<"-"<<m_bins[j].trueD1high
      <<"  recoD2: "<<m_bins[j].recoD2low<<"-"<<m_bins[j].recoD2high
      <<"  trueD2: "<<m_bins[j].trueD2low<<"-"<<m_bins[j].trueD2high<<endl;
      if(m_bins[j].respfuncs.size()>0)
      cout<<" response function name "<<m_bins[j].respfuncs[0]->GetName()<<endl;
      else
      cout<<" no response function"<<endl;
      }*/
}

// --
int XsecParameters::GetBinIndex(SampleTypes sampletype, ReactionTypes reactype, double D1reco,
                                double D1true, double D2reco, double D2true)
{
    int binn = BADBIN;
    for(size_t i = 0; i < m_bins.size(); i++)
    {
        if(m_bins[i].topology == sampletype && m_bins[i].reaction == reactype
           && (D1reco > m_bins[i].recoD1low) && (D1reco < m_bins[i].recoD1high)
           && (D2reco > m_bins[i].recoD2low) && (D2reco < m_bins[i].recoD2high)
           && (D1true > m_bins[i].trueD1low) && (D1true < m_bins[i].trueD1high)
           && (D2true > m_bins[i].trueD2low) && (D2true < m_bins[i].trueD2high))
        {
            binn = (int)i;
            break;
        }
    }
    /*cout<<"topology "<<sampletype<<"  reaction "<<reactype<<endl;
      cout<<"recoP "<<D1reco<<"  trueP "<<D1true<<"    recoD2 "<<D2reco<<"  trueD2 "<<D2true<<endl;
      cout<<"BIN "<<binn<<endl<<endl;*/
    return binn;
}

// initEventMap
void XsecParameters::InitEventMap(std::vector<AnaSample*>& sample, int mode)
{
    InitParameters();
    if(Npar == 0)
    {
        std::cerr << "[ERROR]: In XsecParameters::InitEventMap\n"
                  << "[ERROR]: No parameters delcared. Not building event map."
                  << std::endl;
    }
    //m_evmap.clear();
    m_dial_evtmap.clear();

    for(std::size_t s = 0; s < sample.size(); ++s)
    {
        std::vector<std::vector<int>> sample_map;
        for(int i = 0; i < sample[s] -> GetN(); i++)
        {
            AnaEvent* ev = sample[s] -> GetEvent(i);
            std::vector<int> dial_index_map;

            std::vector<XsecDial> &v_dials = m_dials.at(sample[s] -> GetDetector());
            int num_dials = v_dials.size();

            for(int d = 0; d < num_dials; ++d)
            {
                int idx = v_dials.at(d).GetSplineIndex(ev -> GetTopology(), ev -> GetReaction(),
                                                       ev -> GetQ2());
                if(idx == BADBIN)
                {
                    std::cout << "[WARNING]: Event falls outside spline range.\n"
                              << "[WARNING]: This event will be ignored in the analysis."
                              << std::endl;
                    ev -> AddEvWght(0.0);
                }

                if(mode == 1 && ev -> isSignalEvent())
                    idx = PASSEVENT;

                dial_index_map.push_back(idx);
            }

            sample_map.emplace_back(dial_index_map);
        }

        m_dial_evtmap.emplace_back(sample_map);
    }
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

            std::cout << "[XsecParameters]: Added " << det << "_" << d.GetName()
                      << std::endl;
        }

        std::cout << "[XsecParameters]: Total " << m_dials.at(det).size() << " parameters at "
                  << offset << " for " << det << std::endl;

        offset += m_dials.at(det).size();
    }

    Npar = pars_name.size();
}

// ReWeight
void XsecParameters::ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent, std::vector<double>& params)
{
    if(m_dial_evtmap.empty()) // need to build an event map first
    {
        std::cerr << "[ERROR]: In XsecParameters::ReWeight()\n"
                  << "[ERROR]: Need to build event map index for " << m_name << std::endl;
        return;
    }

    std::vector<XsecDial> &v_dials = m_dials.at(det);
    int num_dials = v_dials.size();
    double weight = 1.0;


    for(int d = 0; d < num_dials; ++d)
    {
        int idx = m_dial_evtmap[nsample][nevent][d];
        double dial_weight = v_dials[d].GetSplineValue(idx, params[d + m_offset.at(det)]);
        weight *= dial_weight;

        /*
        if(dial_weight != 1.0 && det == "INGRID")
        {
            std::cout << "--------------" << std::endl;
            std::cout << "Ev T: " << event -> GetTopology() << std::endl
                      << "Ev R: " << event -> GetReaction() << std::endl
                      << "Ev Q: " << event -> GetQ2() << std::endl;
            std::cout << "Ev I: " << idx << std::endl;
            std::cout << "Ev W: " << dial_weight << std::endl;
            std::cout << "Dl V: " << params[d + m_offset.at(det)] << std::endl;
            std::cout << "Dl N: " << det << "_" << v_dials[d].GetName() << std::endl;
            std::cout << "Sp N: " << v_dials[d].GetSplineName(idx) << std::endl;
        }
        */
    }

    event -> AddEvWght(weight);
}

void XsecParameters::AddDetector(const std::string& det, const std::string& config)
{
    std::cout << "[XsecParameters]: Adding detector " << det << " for " << m_name << std::endl;
    std::fstream f;
    f.open(config, std::ios::in);

    json j;
    f >> j;

    std::string input_dir = std::string(std::getenv("XSLLHFITTER"))
                            + j["input_dir"].get<std::string>();

    std::cout << "[XsecParameters]: Adding the following dials." << std::endl;

    std::vector<XsecDial> v_dials;
    for(const auto& dial : j["dials"])
    {
        if(dial["use"] == true)
        {
            std::string fname_binning = input_dir + dial["binning"].get<std::string>();
            std::string fname_splines = input_dir + dial["splines"].get<std::string>();

            XsecDial x(dial["name"], fname_binning, fname_splines);
            x.SetVars(dial["nominal"], dial["step"], dial["limit_lo"], dial["limit_hi"]);
            x.SetDimensions(8, 10);
            x.Print(true);
            v_dials.emplace_back(x);
        }
    }

    v_detectors.emplace_back(det);
    m_dials.insert(std::make_pair(det, v_dials));
}
