//////////////////////////////////////////////////////////
//
//  A class for event samples for for CCQE analysis
//
//
//
//  Created: Thu Jun  6 12:01:10 CEST 2013   
//  Modified:
//
//////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>

#include "AnaSample.hh"

using namespace std;

// ctor
AnaSample::AnaSample()
{  
  m_sampleid = -99;    //unique id
  m_name     = "none"; //some comprehensible name
  m_norm     = 1.0;
  m_ingrid   = false;
}

// dtor
AnaSample::~AnaSample()
{;}

// ClearEvents -- clears all events from event vector
void AnaSample::ClearEvents()
{
  m_events.clear();
}

// GetN -- get number of events stored
int AnaSample::GetN()
{
  return (int)m_events.size();
}

// GetEvent 
AnaEvent* AnaSample::GetEvent(int evnum)
{
  if(m_events.empty() || evnum>=this->GetN())
  {
    cerr<<"No events are found in "<<m_name<<" sample"<<endl;
    return NULL;
  }
  
  return &m_events[evnum];
}

// AddEvent
void AnaSample::AddEvent(AnaEvent &event)
{
  m_events.push_back(event);
}

// ResetWghts -- set all the event weights in the sample to 1
void AnaSample::ResetWeights()
{
  for(size_t i=0;i<m_events.size();i++) m_events[i].SetEvWght(1.0);
}

// PrintStats
void AnaSample::PrintStats()
{
  cout<<"Sample "<<m_name<<" id = "<<m_sampleid<<endl;
  cout<<"   num of events = "<<m_events.size()<<endl;
  cout<<"   memory used   = "
      <<sizeof(m_events)*m_events.size()/1000.<<" kB"<<endl;
}

