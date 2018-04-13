#include <fstream>
#include <sstream>
#include <iomanip>
#include <assert.h>

#include "FSIParameters.hh"

using namespace std;

//ctor
FSIParameters::FSIParameters(const char *name)
{
  m_name = name;
  hasRegCovMat = false;


  //More hard coding:
  //Here we need to set the FSI params that we're going to use
  //and the step info for MINUIT.
  //One day this will be done in the main ...

  // Sara's old parameters for CC0Pi:
 
  //parameter number, names & prior values
  //(Npar, values and ordering should be in agreement with input TFiles)
  //Npar=5;
  // Npar=6;
  // pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"InelLow"));
  // pars_prior.push_back(1.0);
  // pars_step.push_back(0.412);
  // pars_limlow.push_back(-0.236);
  // pars_limhigh.push_back(2.236);
  // pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"InelHigh"));
  // pars_prior.push_back(1.0);
  // pars_step.push_back(0.338);
  // pars_limlow.push_back(-0.014);
  // pars_limhigh.push_back(2.014);
  // pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"PiAbs"));
  // pars_prior.push_back(1.0);
  // pars_step.push_back(0.412);
  // pars_limlow.push_back(-0.236);
  // pars_limhigh.push_back(2.236);
  // pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"PiProd"));
  // pars_prior.push_back(1.0);
  // pars_step.push_back(0.5);
  // pars_limlow.push_back(-0.5);
  // pars_limhigh.push_back(2.5);
  // pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"CExLow"));
  // pars_prior.push_back(1.0);
  // pars_step.push_back(0.567);
  // pars_limlow.push_back(-0.701);
  // pars_limhigh.push_back(2.701);
  // pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"CExHigh"));
  // pars_prior.push_back(1.0);
  // pars_step.push_back(0.278);
  // pars_limlow.push_back(0);
  // pars_limhigh.push_back(2);

  // My parameter:


  Npar=6;
  pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"InelLow"));
  pars_prior.push_back(1.0);
  pars_step.push_back(0.10);
  pars_limlow.push_back(0);
  pars_limhigh.push_back(2.236);
  pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"InelHigh"));
  pars_prior.push_back(1.0);
  pars_step.push_back(0.10);
  pars_limlow.push_back(0);
  pars_limhigh.push_back(2.014);
  pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"PiAbs"));
  pars_prior.push_back(1.0);
  pars_step.push_back(0.10);
  pars_limlow.push_back(0);
  pars_limhigh.push_back(2.236);
  pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"PiProd"));
  pars_prior.push_back(1.0);
  pars_step.push_back(0.10);
  pars_limlow.push_back(0);
  pars_limhigh.push_back(2.5);
  pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"CExLow"));
  pars_prior.push_back(1.0);
  pars_step.push_back(0.10);
  pars_limlow.push_back(0);
  pars_limhigh.push_back(2.701);
  pars_name.push_back(Form("%s%d%s", m_name.c_str(), 1,"CExHigh"));
  pars_prior.push_back(1.0);
  pars_step.push_back(0.10);
  pars_limlow.push_back(0);
  pars_limhigh.push_back(2);

 }

//dtor
FSIParameters::~FSIParameters()
{;}

// So much upsetting hard coding here, sorry!
// So below I've hacked in a fix to avoid the FSI params looking at cut branch (sample) 0 or 4
// (since these have no protons)
// and reaction 6 has been made to be reaaction 7 (since reaction 6 is not defined in mectopology HL2 cat)

// The dummy response function thing is also a bit of a hack. This allows the occasional job to fail when making 
// resp functions and the fitter still runs but this is not good and should be delt with

// The warning about altering the nominal params implies there's something wrong with your response function.
// You should have+/- 3 sigma relative to the nominal, if you're shifting the nominal make sure you're doing this intentionally

// store response functions in vector of FSI "bins" (Ereco, Etrue, reac, topo)
void FSIParameters::StoreResponseFunctions(vector<TFile*> respfuncs, std::vector<std::pair <double,double> > v_D1edges, 
              std::vector<std::pair <double,double> > v_D2edges)
{
  double dummyx[7]={-1,-0.66,-0.33,0,0.33,0.66,1};
  double dummyy[7]={1,1,1,1,1,1,1};
  int    dummyn=7;
 
  for ( int stInt = 0; stInt < 8; stInt++ ){
    if((stInt==0) || (stInt==4)) continue; // Ignore branches with no proton
    SampleTypes sampletype = static_cast <SampleTypes> (stInt);
    for ( int rtInt = 0; rtInt < 8; rtInt++){
      ReactionTypes reactype = static_cast<ReactionTypes>(rtInt);
      if(rtInt==6) continue; // Hack to deal with missing mectopo6
      //cout<<"reading response functions for topology "<<stInt<<"  reaction "<<rtInt<<endl;
      int nccqebins=v_D1edges.size();
      for(int br=0;br<nccqebins;br++){//reco kinematics bin
        //cout<<"reading rewighting function for reco bin "<<br<<endl;
        for(int bt=0;bt<nccqebins;bt++){//true kinematics bin
          //cout<<"reading rewighting function for true bin "<<bt<<endl;
          FSIBin bin;
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
              //cout<<respfuncs[i]->GetName()<<" "<<name<<endl;
              TGraph* g=(TGraph*)respfuncs[i]->Get(name);
              if(!g){ 
                sprintf(name,"topology_%d/trueBin_%d_topology_%d_reac_%d",stInt,bt,stInt,rtInt);
                g=(TGraph*)respfuncs[i]->Get(name);
              }              //cout<<g<<endl;
              if(!g){
                if(rtInt!=0 && rtInt<6){ // Ignore OOFV splines
                  cout << "Warning, creating dummy respfunc, param: " << i << " " << name << endl;
                  //cout << "getchar to cont" << endl;
                  //getchar();
                }
                g = new TGraph(dummyn, dummyx, dummyy);
              }            
              g->SetName(name);
              if((g->GetY())[3]!=1.0){
                cout << "Warning: altering fsi nominal param: " << i << " " << name << endl;
              }  
              bin.respfuncs.push_back(g);
            }
          }
          m_bins.push_back(bin);
        }
      }
    }
  }

  
}

// --
int FSIParameters::GetBinIndex(SampleTypes sampletype, ReactionTypes reactype, 
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
  return binn;
}

// initEventMap
void FSIParameters::InitEventMap(std::vector<AnaSample*> &sample, int mode) 
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
      //get event info
      int binn = GetBinIndex(static_cast<SampleTypes>(ev->GetSampleType()),
                             static_cast<ReactionTypes>(ev->GetTopology()),
                             ev->GetRecD1(),ev->GetTrueD1(),ev->GetRecD2(),ev->GetTrueD2());
      if(binn == BADBIN) 
      {
        cout<<"WARNING: "<<m_name<<" event "<<" fall outside bin ranges"<<endl;
        cout<<"This event will be ignored in analysis."<<endl;
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
void FSIParameters::EventWeights(std::vector<AnaSample*> &sample, 
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
void FSIParameters::ReWeight(AnaEvent *event, int nsample, int nevent,
            std::vector<double> &params)
{
  if(m_evmap.empty()) //need to build an event map first
  {
    cout<<"******************************" <<endl;
    cout<<"WARNING: No event map specified for "<<m_name<<endl;
    cout<<"Need to build event map index for "<<m_name<<endl;
    cout<<"******************************" <<endl;
    return;
  }

  int binn = m_evmap[nsample][nevent];

  if(binn == PASSEVENT) return;
  if(binn == BADBIN)event->AddEvWght(0.0); //skip!!!!
  else
  {
    vector <TGraph*> respfuncs = m_bins[binn].respfuncs;
    double weight=1;
    if(respfuncs.size()>0){ //needed because there are missing reponse functions when reco very different from true (to save memory)
      for(uint i=0; i<Npar; i++){
        weight = weight*(respfuncs[i]->Eval(params[i]));
      }
    }
    event->AddEvWght(weight);
  }
}
void FSIParameters::ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
            std::vector<double> &params)
{
  //Not implemented yet
  ReWeight(event, nsample, nevent, params);
}


