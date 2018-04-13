#include <fstream>
#include <sstream>
#include <iomanip>
#include <assert.h>

#include "NuclFSIParameters.hh"

using namespace std;


// Currently I'm not suing these, they are not adapted to work with my analysis at all

//ctor
NuclFSIParameters::NuclFSIParameters(const char *name)
{
  m_name = name;
  hasRegCovMat = false;

 
  //parameter number, names & prior values
  //(Npar, values and ordering should be in agreement with input TFiles)
  Npar=3;
  pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"MFP"));
  pars_prior.push_back(1);
  pars_step.push_back(0.2);
  pars_limlow.push_back(0);
  pars_limhigh.push_back(2);
  pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"FrElas"));
  pars_prior.push_back(1);
  pars_step.push_back(0.3);
  pars_limlow.push_back(0);
  pars_limhigh.push_back(2);
  pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"FrAbs"));
  pars_prior.push_back(1);
  pars_step.push_back(0.2);
  pars_limlow.push_back(0);
  pars_limhigh.push_back(2);

 }

//dtor
NuclFSIParameters::~NuclFSIParameters()
{;}

// store response functions in vector of NuclFSI "bins" (Ereco, Etrue, reac, topo)
void NuclFSIParameters::StoreResponseFunctions(vector<TFile*> respfuncs, std::vector<std::pair <double,double> > v_D1edges, 
                        std::vector<std::pair <double,double> > v_D2edges)
{
  for ( int stInt = muTPC; stInt != crDIS+1; stInt++ ){
    SampleTypes sampletype = static_cast <SampleTypes> (stInt);
    for ( int rtInt = cc0pi0p; rtInt != OutFGD+1; rtInt++){
      ReactionTypes reactype = static_cast<ReactionTypes>(rtInt);
      cout<<"reading response functions for topology "<<stInt<<"  reaction "<<rtInt<<endl;
      int nccqebins=v_D1edges.size();
      for(int br=0;br<nccqebins;br++){//reco kinematics bin
        //cout<<"reading rewighting function for reco bin "<<br<<endl;
        for(int bt=0;bt<nccqebins;bt++){//true kinematics bin
          //cout<<"reading rewighting function for true bin "<<bt<<endl;
          NuclFSIBin bin;
          bin.recoD1low = v_D1edges[br].first;
          bin.recoD1high = v_D1edges[br].second;
          bin.trueD1low = v_D1edges[bt].first; //same binning for reco and true
          bin.trueD1high = v_D1edges[bt].second;
          bin.recoD2low = v_D2edges[br].first;
          bin.recoD2high = v_D2edges[br].second;
          bin.trueD2low = v_D2edges[bt].first; //same binning for reco and true
          bin.trueD2high = v_D2edges[bt].second;
          bin.topology = sampletype; 
          bin.reaction = reactype; 
          if(fabs(br-bt)<21) {  //save memory if reco bin and true bin very far away
            for(uint i=0; i<Npar; i++){
              char name[200];
              sprintf(name,"topology_%d/RecBin_%d_trueBin_%d_topology_%d_reac_%d",stInt,br,bt,stInt,rtInt);
              TGraph* g=(TGraph*)respfuncs[i]->Get(name);
              g->SetName(name);
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
int NuclFSIParameters::GetBinIndex(SampleTypes sampletype, ReactionTypes reactype, 
                double D1reco, double D1true, double D2reco, double D2true)
{
  int binn = BADBIN;
  for(size_t i=0;i<m_bins.size();i++)
  {
    if(m_bins[i].topology == sampletype && m_bins[i].reaction == reactype &&
      (D1reco > m_bins[i].recoD1low) && (D1reco  < m_bins[i].recoD1high)  &&
      (D2reco  > m_bins[i].recoD2low) && (D2reco  < m_bins[i].recoD2high) &&
      (D1true > m_bins[i].trueD1low) && (D1true  < m_bins[i].trueD1high)  &&
      (D2true  > m_bins[i].trueD2low) && (D2true  < m_bins[i].trueD2high)){
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
void NuclFSIParameters::InitEventMap(std::vector<AnaSample*> &sample, int mode) 
{
  if(m_bins.empty()) 
    {
      cout<<"Need to build map of response functions for "<<m_name<<" ... exiting ..."<<endl;
      exit(-1);
    }
  m_evmap.clear();
  
  //loop over events to build index map
  for(size_t s=0;s<sample.size();s++)
  {
    vector<int> row;
    for(int i=0;i<sample[s]->GetN();i++)
    {
      AnaEvent *ev = sample[s]->GetEvent(i);
      //skip reactions not prepared in response function
      /*int code = PASSEVENT;
      if(ev->GetTopology() == AntiNu || 
         ev->GetTopology() == OutFGD) 
        {
          row.push_back(code);
          continue;
          }*/     
      //get event info
      int binn = GetBinIndex(static_cast<SampleTypes>(ev->GetSampleType()),
                             static_cast<ReactionTypes>(ev->GetTopology()),
                             ev->GetRecD1(),ev->GetTrueD1(),ev->GetRecD2(),ev->GetTrueD2());
      if(binn == BADBIN) 
      {
        cout<<"WARNING: "<<m_name<<" event "
        <<" fall outside bin ranges"<<endl;
        cout<<"        This event will be ignored in analysis."
        <<endl;
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
void NuclFSIParameters::EventWeights(std::vector<AnaSample*> &sample, 
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
void NuclFSIParameters::ReWeight(AnaEvent *event, int nsample, int nevent,
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
    vector <TGraph*> respfuncs = m_bins[binn].respfuncs;
    double weight=1;
    if(respfuncs.size()>0){ //needed because there are missing reponse functions when reco very different from true (to save memory)
      for(uint i=0; i<Npar; i++){
        weight = weight*(respfuncs[i]->Eval(params[i]));
        //if(weight!=1)
        //cout<<"reweighting using weight "<<weight<<"  from bin "<<binn<<endl;
      }
    }
     event->AddEvWght(weight);
  }
}
void NuclFSIParameters::ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
            std::vector<double> &params)
{
  //Not implemented yet
  ReWeight(event, nsample, nevent, params);
}


