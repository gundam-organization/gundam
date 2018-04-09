//This is the code that actually reads int he MC tree and fills the event info. 
//The tree should be produced by feeding a HL2 microtree into the treeconvert macro.


#define anyTreeMC_cxx

#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "anyTreeMC.hh"

using namespace std;

void anyTreeMC::GetEvents(std::vector<AnaSample*> ana_samples)
{
  if (fChain == 0) return;
  if (ana_samples.empty()) return;
  
  Long64_t nentries = fChain->GetEntries();

  Long64_t nbytes = 0, nb = 0;
  //nentries = 500;
  for (Long64_t jentry=0; jentry<nentries;jentry++) 
  {
      if(jentry % (int)1e+5 == 0)
      cout<<"Processing "<<jentry<<" out of "<<nentries<<endl;
      nb = fChain->GetEntry(jentry); nbytes += nb;
      //create and fill event structure
      AnaEvent ev(jentry);
      ev.SetSampleType(qesampleFinal);
      int evtTopo=evtTopology; //For my analysis 0 CC0pi0p, 1 CC0pi1p, 2 CC0pinp, 3 CC1pi, 4 CCOther, 5 backg(NC+antinu), 7 OOFV
      //cout << "Evt Topology is " << evtTopo << endl;
      if(evtTopo>7) cout << "*** Warning: evtTopology>7, evt Topology is " << evtTopo << endl;
      ev.SetReaction(evtTopology); // mectopology (i.e. CC0Pi,CC1Pi etc)
      ev.SetReactionMode(evtReaction); // reaction (i.e. CCQE,CCRES etc)
      ev.SetTrueEnu(TrueEnergy/1000.0);   //MeV --> GeV
      ev.SetRecEnu(MainRecEneGlb/1000.0); //MeV --> GeV
      ev.SetTrueD1trk(trueD1);     
      ev.SetRecD1trk(MainD1Glb);   
      ev.SetTrueD2trk(trueD2);
      ev.SetRecD2trk(MainD2);
      ev.SetEvWght(weight);
      ev.SetEvWghtMC(weight);

      ev.SetpMomRec(pMomRec);
      ev.SetpMomTrue(pMomTrue);
      ev.SetmuMomRec(muMomRec);
      ev.SetmuMomTrue(muMomTrue);
      ev.SetmuCosThetaRec(muCosThetaRec);
      ev.SetmuCosThetaTrue(muCosThetaTrue);
      ev.SetpCosThetaRec(pCosThetaRec);
      ev.SetpCosThetaTrue(pCosThetaTrue);


      
      //Loop over AnaSample objs and fill appropriate ones
      for(size_t i=0;i<ana_samples.size();i++)
      {
        if(ana_samples[i]->GetSampleType() == qesampleFinal)
        {
          // cout << "DEBUG DEBUG Printing Event:" << endl;
          // if(ev.GetTrueD2trk()!=floor(ev.GetTrueD2trk())) ev.Print();
          // if(ev.GetRecD2trk()!=floor(ev.GetRecD2trk())) ev.Print();
          //DEBUG TIME
          //if( ((ev.GetReaction()==1) || (ev.GetReaction()==2) ) && ((ev.GetpMomTrue()<460)||(ev.GetmuMomTrue()<460)||(ev.GetpD2True()<0.4)||(ev.GetmuD2True()<0.4))) ev.Print();
          //
          ana_samples[i]->AddEvent(ev);
          break;
        }  
      }  
  } // jentry loop

  //Print some stats
  for(size_t i=0;i<ana_samples.size();i++)
    ana_samples[i]->PrintStats();
}
