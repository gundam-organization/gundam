#include "FitParameters.hh"

FitParameters::FitParameters(const std::string& file_name, const std::string& par_name, bool random_priors)
{
    m_name = par_name;

    //get the binning from a file
    SetBinning(file_name);

    double rand_prior = 0.0;
    TRandom3 rng(0);

    //parameter names & prior values
    for(std::size_t i = 0; i < Npar; ++i)
    {
        pars_name.push_back(Form("%s%d", m_name.c_str(), (int)i));
        if(random_priors == true)
        {
            rand_prior = 2.0 * rng.Uniform(0.0, 1.0);
            pars_prior.push_back(rand_prior);
        }
        else
            pars_prior.push_back(1.0); //all weights are 1.0 a priori

        pars_step.push_back(0.05);
        pars_limlow.push_back(0.0);
        pars_limhigh.push_back(10.0);
    }
}

FitParameters::~FitParameters()
{;}

void FitParameters::SetBinning(const std::string& file_name)
{
    std::ifstream fin(file_name, std::ios::in);
    if(!fin.is_open())
    {
        std::cerr << "[ERROR]: In FitParameters::SetBinning()\n"
                  << "[ERROR]: Failed to open binning file: " << file_name << std::endl;
        return;
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
                std::cerr << "[ERROR]: In FitParameters::SetBinning()\n"
                          << "[ERROR]: Bad line format: " << line << std::endl;
                continue;
            }
            m_bins.emplace_back(FitBin(D1_1, D1_2, D2_1, D2_2));
        }
        fin.close();
        Npar = m_bins.size();

        std::cout << "[FitParameters]: Fit binning: \n";
        for(std::size_t i = 0; i < m_bins.size(); ++i)
        {
            std::cout << std::setw(3) << i
                      << std::setw(5) << m_bins[i].D2low
                      << std::setw(5) << m_bins[i].D2high
                      << std::setw(5) << m_bins[i].D1low
                      << std::setw(5) << m_bins[i].D1high << std::endl;
        }
    }
}

int FitParameters::GetBinIndex(double D1, double D2)
{
    for(int i = 0; i < m_bins.size(); ++i)
    {
        if(D1 >= m_bins[i].D1low && D1 < m_bins[i].D1high &&
           D2 >= m_bins[i].D2low && D2 < m_bins[i].D2high)
        {
            return i;
        }
    }
    return BADBIN;
}


// initEventMap
void FitParameters::InitEventMap(std::vector<AnaSample*> &sample, int mode)
{
    m_evmap.clear();

    //loop over events to build index map
    for(std::size_t s=0; s < sample.size(); s++)
    {
        vector<int> row;
        for(int i=0; i < sample[s] -> GetN() ; i++)
        {
            AnaEvent *ev = sample[s] -> GetEvent(i);

            // SIGNAL DEFINITION TIME
            // Warning, important hard coding up ahead:
            // This is where your signal is actually defined, i.e. what you want to extract an xsec for
            // N.B In Sara's original code THIS WAS THE OTHER WAY AROUND i.e. this if statement asked what was NOT your signal
            // Bare that in mind if you've been using older versions of the fitter.

            //if((ev->GetTopology()==1)||(ev->GetTopology()==2))
            if(ev -> isSignalEvent())
            {
                double D1 = ev -> GetTrueD1();
                double D2 = ev -> GetTrueD2();
                int bin = GetBinIndex(D1, D2);
                if(bin == BADBIN)
                {
                    std::cout << "[WARNING]: " << m_name << ", Event: " << i << std::endl
                              << "[WARNING]: D1 = " << D1 << ", D2 = " << D2 << ", falls outside bin ranges." << std::endl
                              << "[WARNING]: This event will be ignored in the analysis." << std::endl;
                }
                row.push_back(bin);
            }
            else
            {
                row.push_back(PASSEVENT);
                continue;
            }

        }
        m_evmap.push_back(row);
    }
}

// EventWeghts
void FitParameters::EventWeights(std::vector<AnaSample*> &sample, std::vector<double> &params)
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
            ReWeight(ev, s, i, params);
        }
    }
}


void FitParameters::ReWeight(AnaEvent *event, int nsample, int nevent, std::vector<double> &params)
{
    if(m_evmap.empty()) //need to build an event map first
    {
        cout<<"Need to build event map index for "<<m_name<<endl;
        return;
    }

    int bin = m_evmap[nsample][nevent];

    //skip event if not Signal
    if(bin == PASSEVENT) return;

    // If bin fell out of valid ranges, pretend the event just didn't happen:
    if(bin == BADBIN) event->AddEvWght(0.0);
    else
    {
        if(bin > params.size())
        {
            cerr<<"ERROR: number of bins "<<m_name<<" does not match num of param"<<endl;
            event->AddEvWght(0.0);
        }
        event->AddEvWght(params[bin]);
        //cout << "ReWeight param " << binn << endl;
        //cout << "Weight is " << params[binn] << endl;
    }
}

void FitParameters::ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
        std::vector<double> &params)
{
    //Treat as all fit parameters
    ReWeight(event, nsample, nevent, params);
}
