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
#include "AnaSample.hh"

// ctor
AnaSample::AnaSample()
{
    m_sample_id = -99;    //unique id
    m_name     = "none"; //some comprehensible name
    m_detector = "none";
    m_norm     = 1.0;
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
    if(m_events.empty())
    {
        std::cerr << "[ERROR]: In AnaSample::GetEvent()" << std::endl;
        std::cerr << "[ERROR]: No events are found in " << m_name << " sample." << std::endl;
        return nullptr;
    }
    else if(evnum >= m_events.size())
    {
        std::cerr << "[ERROR]: In AnaSample::GetEvent()" << std::endl;
        std::cerr << "[ERROR]: Event number out of bounds in " << m_name << " sample." << std::endl;
        return nullptr;
    }

    return &m_events.at(evnum);
}

// AddEvent
void AnaSample::AddEvent(AnaEvent& event)
{
    m_events.push_back(event);
}

// ResetWghts -- set all the event weights in the sample to 1
void AnaSample::ResetWeights()
{
    for(auto& event : m_events)
        event.SetEvWght(1.0);
}

// PrintStats
void AnaSample::PrintStats()
{
    double mem_kb = sizeof(m_events) * m_events.size() / 1000.0;
    std::cout << "[AnaSample]: Sample " << m_name << " ID = " << m_sample_id << std::endl;
    std::cout << "[AnaSample]: Num of events = " << m_events.size() << std::endl;
    std::cout << "[AnaSample]: Memory used   = " << mem_kb << " kB." << std::endl;
}

