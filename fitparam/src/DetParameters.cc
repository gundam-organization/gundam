#include <fstream>
#include <sstream>
#include <iomanip>
#include <assert.h>

#include "DetParameters.hh"

using namespace std;

//ctor
DetParameters::DetParameters(const char *fname,TVectorD* detweights,
                             std::vector<AnaSample*> samples, const char *name)
{
  m_name    = name;
  hasRegCovMat = false;

  //get the binning from a file
  SetBinning(fname, samples);

  //parameter names & prior values
  cout<<"Det weight starting values "<<endl;
  for(size_t i=0;i<Npar;i++)
  {
    pars_name.push_back(Form("%s%d", m_name.c_str(), (int)i));
    double parprior = 1;//(*detweights)(i);  //nominal value coming from detsyst sampling (NEW samples with corrections applied)
    cout<<i<<" "<<parprior<<endl;
    pars_prior.push_back(parprior);
    pars_step.push_back(0.1);
    pars_limlow.push_back(0.5);
    pars_limhigh.push_back(1.5);
  }
}

//dtor
DetParameters::~DetParameters()
{;}

void DetParameters::SetBinning(const char *fname, std::vector<AnaSample*> &samples)
{
  string line;
  DetBin bin;
  //loop over the 6 samples
  for(int i=0; i<samples.size(); i++){
    bin.sample=samples[i]->GetSampleType();
    ifstream fin(fname);
    assert(fin.is_open());
    while (getline(fin, line)){
      stringstream ss(line);
      double D11, D12, D21, D22;
      if(!(ss>>D21>>D22>>D11>>D12)) {
        cerr<<"Bad line format: "<< endl
            <<"     " << line << endl;
        continue;
      }
      bin.D1low    = D11;
      bin.D1high   = D12;
      bin.D2low  = D21;
      bin.D2high = D22;
      m_bins.push_back(bin);
    }
    fin.close();
  }
  Npar = m_bins.size();
  
  cout<<endl<<"Det Syst binning:"<<endl;
  for(size_t i = 0;i<m_bins.size();i++)
    {
      cout<<setw(3)<<i
    <<setw(5)<<m_bins[i].D2low
    <<setw(5)<<m_bins[i].D2high
    <<setw(5)<<m_bins[i].D1low
    <<setw(5)<<m_bins[i].D1high<<endl;
    }
  cout<<endl<<"Det Syst # parameters: "<<Npar<<endl;
  cout<<endl;
}

// --
int DetParameters::GetBinIndex(double p, double D2, int sample_id)
{
  int binn = BADBIN;
  for(size_t i=0;i<m_bins.size();i++)
  {
    if(sample_id != m_bins[i].sample) continue;
    if(p>=m_bins[i].D1low && p<m_bins[i].D1high && D2>=m_bins[i].D2low && D2<m_bins[i].D2high) binn = (int)i;
  }
  return binn;
}

// initEventMap
void DetParameters::InitEventMap(std::vector<AnaSample*> &sample, int mode) 
{
  m_evmap.clear();
  
  //loop over events to build index map
  for(size_t s=0;s<sample.size();s++)
  {
    vector<int> row;
    for(int i=0;i<sample[s]->GetN();i++)
    {
      AnaEvent *ev = sample[s]->GetEvent(i);
      //get event true p and D2
      double p   = ev->GetRecD1();
      double D2 = ev->GetRecD2();
      int reaction = ev->GetTopology();
      int binn   = GetBinIndex(p, D2,sample[s]->GetSampleType());
      if(binn == BADBIN)
      {
        cout<<"WARNING: "<<m_name<<" p = "<<p<<" D2 = "<<D2<<" fall outside bin ranges"<<endl;
        cout<<" This event will be ignored in analysis."  <<endl;
      }
      //If event is signal let the c_i params handle the reweighting:
      if( mode==1 && ((ev->GetTopology()==1)||(ev->GetTopology()==2)) ) binn = PASSEVENT;
      row.push_back(binn);
    }
    m_evmap.push_back(row);
  }
}

// EventWeghts
void DetParameters::EventWeights(std::vector<AnaSample*> &sample, 
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
      //if(ev->GetReaction()==0) //detector systematics apply only to signal
      ReWeight(ev, s, i, params);
    }
  }
}

// ReWeight
void DetParameters::ReWeight(AnaEvent *event, int nsample, int nevent,
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
      cerr<<"ERROR: number of bins "<<binn<<" for "<<m_name<<" does not match num of param"<<endl;
      event->AddEvWght(0.0);
    }
    event->AddEvWght(params[binn]);
  }
}

void DetParameters::ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
            std::vector<double> &params)
{
  //Not implemented yet
  ReWeight(event, nsample, nevent, params);
}

