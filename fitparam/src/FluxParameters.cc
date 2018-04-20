#include "FluxParameters.hh"

//ctor
FluxParameters::FluxParameters(const std::string& name)
{
    m_name = name;
}

FluxParameters::FluxParameters(std::vector<double> &enubins, const std::string& name)
{
    m_name    = name;
    m_enubins = enubins;

    //does one need to distinguish neutrino flavours???
    numu_flux = 11;

    //parameter names & prior values
    for(size_t i=0;i<enubins.size()-1;i++)
    {
        pars_name.push_back(Form("%s%d", m_name.c_str(), (int)i));
        pars_prior.push_back(1.0); //all weights are 1.0 a priori
        pars_step.push_back(0.1);
        pars_limlow.push_back(0.0);
        pars_limhigh.push_back(5.0);
    }

    Npar = pars_name.size();

    std::cout << "[FluxParameters]: Flux binning " << std::endl;
    for(std::size_t i = 0; i < enubins.size(); ++i)
    {
        std::cout << i << " " << enubins[i] << std::endl;
    }
}

//dtor
FluxParameters::~FluxParameters()
{;}

// --
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

// initEventMap
void FluxParameters::InitEventMap(std::vector<AnaSample*> &sample, int mode)
{
    InitParameters();
    m_evmap.clear();
    //loop over events to build index map
    for(size_t s=0;s<sample.size();s++)
    {
        vector<int> row;
        for(int i=0;i<sample[s]->GetN();i++)
        {
            AnaEvent *ev = sample[s]->GetEvent(i);
            //get true neutrino energy
            double enu = ev->GetTrueEnu();
            int binn = GetBinIndex(sample[s] -> GetDetector(), enu);
            if(binn == BADBIN)
            {
                cout<<"WARNING: "<<m_name<<" enu "<<enu<<" fall outside bin ranges"<<endl;
                cout<<"        This event will be ignored in analysis."<<endl;
                ev->Print();
            }
            //If event is signal let the c_i params handle the reweighting:
            if( mode==1 && ((ev->GetTopology()==1)||(ev->GetTopology()==2)) ) binn = PASSEVENT;
            row.push_back(binn);
        }//event loop
        m_evmap.push_back(row);
    }//sample loop
}

// EventWeghts
void FluxParameters::EventWeights(std::vector<AnaSample*> &sample,
        std::vector<double> &params)
{
    if(m_evmap.empty()) //build an event map
    {
        cout<<"******************************" <<endl;
        cout<<"WARNING: No event map specified for "<<m_name<<endl;
        cout<<"Need to build event map index for "<<m_name<<endl;
        cout<<"WARNING: initialising in mode 0" <<endl;
        cout<<"******************************" <<endl;
        InitEventMap(sample, 0);
    }

    for(size_t s=0;s<sample.size();s++)
    {
        for(int i=0;i<sample[s]->GetN();i++)
        {
            AnaEvent *ev = sample[s]->GetEvent(i);
            std::string det = sample[s] -> GetDetector();
            ReWeight(ev, det, s, i, params);
        }
    }
}

// ReWeight
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
            event -> AddEvWght(params[bin + m_det_offset.at(det)]);
        /*
        else
        {
            std::cout << "[WARNING]: In FitParameters::ReWeight()\n"
                      << "[WARNING]: Detector " << det << " has not been added to fit parameters.\n"
                      << "[WARNING]: Ignoring event reweight." << std::endl;
        }
        */
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
        for(int i = 0; i < m_det_bins.at(pear.first).size()-1; ++i)
        {
            pars_name.push_back(Form("%s_%s_%d", m_name.c_str(), pear.first.c_str(), i));
            pars_prior.push_back(1.0); //all weights are 1.0 a priori
            pars_step.push_back(0.1);
            pars_limlow.push_back(0.0);
            pars_limhigh.push_back(5.0);

            std::cout << i << ": " << m_det_bins.at(pear.first).at(i) << std::endl;
        }
    }

    Npar = pars_name.size();

    /*
    std::cout << "[FluxParameters]: Flux binning " << std::endl;
    for(std::size_t i = 0; i < enubins.size(); ++i)
    {
        std::cout << i << " " << enubins[i] << std::endl;
    }
    */
}
