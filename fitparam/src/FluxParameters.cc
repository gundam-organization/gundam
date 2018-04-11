#include "FluxParameters.hh"

//ctor
FluxParameters::FluxParameters(std::vector<double> &enubins,
        const char *name, bool addIngrid)
{
    m_name    = name;
    m_enubins = enubins;
    hasRegCovMat = false;


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
    if(addIngrid){
        for(size_t i=0;i<enubins.size()-1;i++)
        {
            pars_name.push_back(Form("%s%d_INGRID", m_name.c_str(), (int)i));
            pars_prior.push_back(1.0); //all weights are 1.0 a priori
            pars_step.push_back(0.1);
            pars_limlow.push_back(0.0);
            pars_limhigh.push_back(5.0);
        }
    }

    Npar = pars_name.size();

    cout<<endl<<"Flux binning:"<<endl;
    for(size_t i = 0;i<enubins.size();i++)
    {
        cout<<i<<" "<<enubins[i]<<endl;
    }
    cout<<endl;
}

//dtor
FluxParameters::~FluxParameters()
{;}

// --
int FluxParameters::GetBinIndex(double enu)
{
    int binn = BADBIN;
    for(size_t i=0;i<(m_enubins.size()-1);i++)
    {
        if(enu>=m_enubins[i] && enu<m_enubins[i+1])
        {
            binn = i;
            break;
        }
    }
    return binn;
}

// initEventMap
void FluxParameters::InitEventMap(std::vector<AnaSample*> &sample, int mode)
{
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
            int binn = GetBinIndex(enu);
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
            if(sample[s]->isIngrid()) ReWeightIngrid(ev, s, i, params);
            else ReWeight(ev, s, i, params);
        }
    }
}

// ReWeight
void FluxParameters::ReWeight(AnaEvent *event, int nsample, int nevent,
        std::vector<double> &params)
{
    if(m_evmap.empty()) //need to build an event map first
    {
        cout<<"Need to build event map index for "<<m_name<<endl;
        return;
    }

    int binn = m_evmap[nsample][nevent];

    if(binn == PASSEVENT) return;
    if(binn == BADBIN) event->AddEvWght(0.0); //skip!!!!
    else
    {
        if(binn>(int)params.size())
        {
            cerr<<"ERROR: number of bins "<<m_name<<" does not match num of param"<<endl;
            event->AddEvWght(0.0);
        }
        event->AddEvWght(params[binn]);
    }
}
// ReWeightIngrid
void FluxParameters::ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
        std::vector<double> &params)
{
    //cout << "ReWeighting INGRID sample" << endl;
    if(m_evmap.empty()) //need to build an event map first
    {
        cout<<"Need to build event map index for "<<m_name<<endl;
        return;
    }

    int binn = m_evmap[nsample][nevent];

    if(binn == PASSEVENT) return;
    if(binn == BADBIN) event->AddEvWght(0.0); //skip!!!!
    else
    {
        if(binn>(int)params.size())
        {
            cerr<<"ERROR: number of bins "<<m_name<<" does not match num of param"<<endl;
            event->AddEvWght(0.0);
        }
        //event->AddEvWght(params[binn+Npar]);
        event->AddEvWght(params.at(binn+m_enubins.size()-1));
    }
}



