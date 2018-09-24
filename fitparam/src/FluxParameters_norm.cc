#include "FluxParameters_norm.hh"

using namespace std;

//ctor
FluxParameters_norm::FluxParameters_norm(const char *name)
{
  m_name    = name;
  Npar      = 1;
  
  //parameter names & prior values
  for(size_t i=0;i<Npar;i++)
  {
    pars_name.push_back(Form("%s%d", m_name.c_str(), (int)i));
    pars_prior.push_back(1.0); //all weights are 1.0 a priori 
    pars_step.push_back(0.1);
    pars_limlow.push_back(0.0);
    pars_limhigh.push_back(5.0);
  }  
}

//dtor
FluxParameters_norm::~FluxParameters_norm()
{;}

void FluxParameters_norm::InitEventMap(std::vector<AnaSample*> &sample,int mode) 
{;}


//DoThrow (can I overwrite the mother class???)
/*void FluxParameters_norm::DoThrow(std::vector<double> &pars){
  pars.clear();
  pars[0]=gRandom->Gaus(1,0.11); //11% uncertainty should be input
  cout<<"using correct DoThrow "<<pars[0]<<endl;
}
*/

// EventWeghts
void FluxParameters_norm::EventWeights(std::vector<AnaSample*> &sample, 
          std::vector<double> &params)
{
  for(size_t s=0;s<sample.size();s++)
  {
    for(int i=0;i<sample[s]->GetN();i++)
    {
      AnaEvent *ev = sample[s]->GetEvent(i);
      ReWeight(ev, s, i, params);
    }
  }
}

// ReWeight
void FluxParameters_norm::ReWeight(AnaEvent *event, int nsample, int nevent,
            std::vector<double> &params)
{
  event->AddEvWght(params[0]);
}
void FluxParameters_norm::ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
            std::vector<double> &params)
{
  //Not implemented yet
  ReWeight(event, nsample, nevent, params);
}

